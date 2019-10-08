import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import LeakyReLU

from tensorflow.keras.layers import Average
from tensorflow.keras.layers import Layer


class DWPW2D(Layer):
    def __init__(self, filters, alpha, stride):
        super(DWPW2D, self).__init__()

        """ ConvDepthWise

        This function defines a basic bottleneck structure.
        # Arguments
            inputs: Tensor, input tensor of conv layer.
            kernel_size: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            filters: Integer, the dimensionality of the output space.

            alpha: Integer, expansion factor.
                t is always applied to the input size.
            stride: An integer or tuple/list of 2 integers,specifying the strides
                of the convolution along the width and height.Can be a single
                integer to specify the same value for all spatial dimensions.

        # Returns   
            Output tensor.
        """
        _channel_axis = 3
        self.alpha=  alpha# tf.cast(alpha, tf.float32)
        self.dw_conv = DepthwiseConv2D(
            data_format="channels_last",
                            kernel_size=(3, 3), 
                            strides=(stride, stride),
                            depth_multiplier=self.alpha,
                            padding='same')

        # x = BatchNormalization(axis=self._channel_axis)(x)
        # x = Activation(LeakyReLU)(x)

        # Point wise operation
        self.conv2d = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')

        self.bn = BatchNormalization(axis=_channel_axis)

        self.activation = LeakyReLU(alpha=0.1)

    def call(self, inputs):
        
        dw_conv = self.dw_conv(inputs)
        pw_conv = self.conv2d(dw_conv)
        bn = self.bn(pw_conv)
        activation = self.activation(bn)

        return activation
        


class Encoder(Layer):

    def __init__(self, alpha=1):
        super(Encoder, self).__init__()

        self.conv1 = DWPW2D(32, alpha, 2)

        # Conv2_dw / Conv2
        self.conv2 = DWPW2D(64, alpha, 2)
        
        # Conv3_dw / Conv3
        self.conv3 = DWPW2D(128, alpha, 2)

        # Conv4a_dw / Conv4a
        self.conv4a = DWPW2D(256, alpha, 2)

        # Conv4b_dw / Conv4b1
        self.conv4b = DWPW2D(256, alpha, 1)

        # Conv5a_dw / Conv5a
        self.conv5a = DWPW2D(512, alpha, 2)

        # Conv5b_dw / Conv5b
        self.conv5b = DWPW2D(512, alpha, 1)

        # Conv6a_dw / Conv6a
        self.conv6a = DWPW2D(1024, alpha, 2)

        # Conv6b_dw / Conv6b
        self.conv6b = DWPW2D(1024, alpha, 1)

    def call(self, x):
        # #################################################
        # ENCODER 
        # #################################################

        # Conv1_dw / Conv1
        conv1 = self.conv1(x)

        # Conv2_dw / Conv2
        conv2 = self.conv2(conv1)
        
        # Conv3_dw / Conv3
        conv3 = self.conv3(conv2)

        # Conv4a_dw / Conv4a
        conv4a = self.conv4a(conv3)

        # Conv4b_dw / Conv4b1
        conv4b = self.conv4b(conv4a)

        # Conv5a_dw / Conv5a
        conv5a = self.conv5a(conv4b)

        # Conv5b_dw / Conv5b
        conv5b = self.conv5b(conv5a)

        # Conv6a_dw / Conv6a
        conv6a = self.conv6a(conv5b)

        # Conv6b_dw / Conv6b
        conv6b = self.conv6b(conv6a)

        return (conv6b, conv5b, conv4b, conv3, conv2 )


class Decoder(Layer):
    def __init__(self, alpha=1):
        super(Decoder, self).__init__()
        
        #################################################
        # DECODER 
        #################################################
        self._concat_axis = 3
        # Conv7_dw / Conv7
        self.conv7 = DWPW2D( 256, alpha, 1)

        # Conv8_dw /Conv8
        self.conv8 = DWPW2D(128, alpha, 1)

        # Conv9_dw /Conv9
        self.conv9 = DWPW2D( 64, alpha, 1)

        # Conv10_dw / Conv10       
        self.conv10 =  DWPW2D( 32, alpha, 1)

        # Conv11_dw / Con11
  
        # Convolutions
        self.conv11 =  DWPW2D( 16, alpha, 1)

    def resize_like(self, input_tensor, ref_tensor, scale): # resizes input tensor wrt. ref_tensor
        H, W = ref_tensor.get_shape()[1], ref_tensor.get_shape()[2]
        return tf.compat.v1.image.resize_nearest_neighbor(input_tensor, [H*scale, W*scale])

    def concat(self, input_1, input_2, name):
        return tf.concat([input_1, input_2], axis=self._concat_axis, name=name)

    def call(self, args):

        #################################################
        # DECODER 
        #################################################
        conv2 = args['conv2']
        conv3 = args['conv3']
        conv4b = args['conv4b']
        conv5b = args['conv5b']
        conv6b = args['conv6b']

        # Conv7_dw / Conv7
        conv7 = self.conv7(conv6b)

        # Conv8_dw /Conv8
        # Operations
        conv7_x2 = self.resize_like(conv7, conv7, 2)
        concat_op1 = self.concat(conv7_x2, conv5b, 'concat_op1')
        
        # Convolutions
        conv8 = self.conv8(concat_op1)

        # Conv9_dw /Conv9
        # Operations
        conv8_x2 = self.resize_like(conv8, conv8, 2)
        concat_op2 = self.concat(conv8_x2, conv4b, 'concat_op2')
        
        #Convolutions   
        conv9 = self.conv9(concat_op2)

        # Conv10_dw / Conv10
        # Operations
        conv9_x2 = self.resize_like(conv9, conv9, 2)
        concat_op3 = self.concat(conv9_x2, conv3, 'concat_op3')
        
        #Convolutions
        conv10 =  self.conv10(concat_op3)

        # Conv11_dw / Con11
        # Operations
        conv10_x2 = self.resize_like(conv10, conv10, 2)
        concat_op4 = self.concat(conv10_x2, conv2, 'concat_op4')
        
        # Convolutions
        conv11 =  self.conv11(concat_op4)

        return  conv11, conv10, conv9, conv8, conv7


class LightFlow(Model):
    def __init__(self, alpha=1.0):

        super(LightFlow, self).__init__()

        self._concat_axis = 3
        self.alpha = 1
        self.average = Average(name='average_layer')
        
        self.encoder = Encoder()
        self.decoder = Decoder()

        ##################################################
        # Optical Flow Predictions
        ##################################################

        # Conv12_dw / conv12
        self.conv12 = DWPW2D( 2, self.alpha, 1)

        # Conv13_dw / conv13
        self.conv13 = DWPW2D( 2, self.alpha, 1)

        # Conv14_dw / conv14
        self.conv14 = DWPW2D( 2, self.alpha, 1)

        # Conv15_dw / con15
        self.conv15 =  DWPW2D( 2, self.alpha, 1)

        # Conv16_dw / conv16
        self.conv16 =  DWPW2D( 2, self.alpha, 1)
        

    def resize_like(self, input_tensor, ref_tensor, scale): # resizes input tensor wrt. ref_tensor
        H, W = ref_tensor.get_shape()[1], ref_tensor.get_shape()[2]
        return tf.compat.v1.image.resize_nearest_neighbor(input_tensor, [H*scale, W*scale])


    def call(self, x):
        # Expected INPUT_SHAPE = (384, 512,6)
        # 4D Tensor placeholder for input images

        # Encoder

        conv6b, conv5b, conv4b, conv3, conv2  = self.encoder(x)
        
        # Decoder
        kargs ={'conv2': conv2,'conv3': conv3, 'conv4b': conv4b, 
                'conv5b': conv5b, 'conv6b':conv6b}

        conv11, conv10, conv9, conv8, conv7 = self.decoder(inputs=kargs)

        ##################################################
        # Optical Flow Predictions
        ##################################################

        # Conv12_dw / conv12
        conv12 = self.conv12(conv7)

        # Conv13_dw / conv13
        conv13 = self.conv13(conv8)

        # Conv14_dw / conv14
        conv14 = self.conv14(conv9)

        # Conv15_dw / con15
        conv15 =  self.conv15(conv10)

        # Conv16_dw / conv16
        conv16 =  self.conv16(conv11)

        ###################################################
        # Multiple Optical Flow Predictions Fusion
        ###################################################
        conv12_x16 = self.resize_like(conv12, conv12, 16)
        conv13_x8  = self.resize_like(conv13, conv13, 8)
        conv14_x4  = self.resize_like(conv14, conv14, 4)
        conv15_x2  = self.resize_like(conv15, conv15, 2)
    

        # 96x128x2
        flow = self.average([conv12_x16, 
                            conv13_x8, 
                            conv14_x4,
                            conv15_x2, 
                            conv16])

        # Fuse groundtrunth resolution with prediction
        #flow = tf.image.resize_bilinear(average, tf.stack([height, width]), align_corners=True)
        #flow = Lambda(resize_like, arguments={'ref_tensor':average, 'scale': 4})(average)
        
        return flow