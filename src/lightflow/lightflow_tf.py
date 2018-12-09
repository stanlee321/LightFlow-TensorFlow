import tensorflow as tf
from ..utils import  average_endpoint_error, LeakyReLU
from tensorflow.keras.layers import Dense, Add, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, LeakyReLU
from tensorflow.keras.layers import BatchNormalization, Lambda, Average
from tensorflow.keras.layers import Concatenate, UpSampling2D 
from tensorflow.keras.layers import Input

import tensorflow.contrib.slim as slim
# Import Own Lib
from ..depthwise_conv2d import DepthwiseConvolution2D
from ..net import Net, Mode
from ..downsample import downsample

def _depthwise_convolution2D(input, _alpha, deepwise_filter_size, kernel_size, strides, padding='same', bias=False, training=True):
    x= DepthwiseConvolution2D(int(deepwise_filter_size * _alpha), 
                                kernel_size, 
                                strides=strides,
                                padding=padding, 
                                use_bias=bias)(input)
    x= BatchNormalization()(x, training=training)
    x = LeakyReLU(alpha=0.1)(x)
    return  x

def _convolution2D(input, _alpha,
                    deepwise_filter_size,
                    kernel_size, 
                    strides, 
                    padding='same', 
                    bias=False,
                    training=True):
    x = Conv2D(int(deepwise_filter_size * _alpha), 
                kernel_size, 
                strides=strides,
                padding=padding,
                use_bias=bias)(input)
    x = BatchNormalization()(x, training=training)
    x = LeakyReLU(alpha=0.1)(x)
    return x

def resize_like(input_tensor, ref_tensor, scale): # resizes input tensor wrt. ref_tensor
    H, W = ref_tensor.get_shape()[1], ref_tensor.get_shape()[2]
    return tf.image.resize_nearest_neighbor(input_tensor, [H.value*scale, W.value*scale])


class LightFlow(Net):

    def __init__(self, mode=Mode.TRAIN, debug=False):
        super(LightFlow, self).__init__(mode=mode, debug=debug)

    def model(self, inputs, training_schedule, trainable=True, build=False):
        _concat_axis = 3
        alpha = 1.0
        beta = 1.0
        if 'warped' in inputs and 'flow' in inputs and 'brightness_error' in inputs:
            concat_inputs = tf.concat([inputs['input_a'],
                                        inputs['input_b'],
                                        inputs['warped'],
                                        inputs['flow'],
                                        inputs['brightness_error']], axis=_concat_axis)
        else:
            concat_inputs = tf.concat([inputs['input_a'], inputs['input_b']], axis=_concat_axis)
        
        if build is True:
            #INPUT_SHAPE = #(384, 512,6)
            concat_inputs =  tf.concat([inputs['input_a'], inputs['input_b']], axis=_concat_axis) #Input(shape=INPUT_SHAPE)
        
        #################################################
        # ENCODER 
        #################################################

        # Conv1_dw / Conv1
        conv1_dw = self.depthwiseconv(concat_inputs, 6, beta, 2, (3,3), 'conv1_dw')
        conv1 =  self.conv_2d(conv1_dw, 32, alpha, 1, (1,1), 'conv1_dw')
 
        # Conv2_dw / Conv2
        conv2_dw = self.depthwiseconv(conv1, 32, beta, 2, (3,3), 'conv1_dw')
        conv2 = self.conv_2d(conv2_dw, 64, alpha, 1, (1,1), 'conv2')
       
        # Conv3_dw / Conv3
        conv3_dw = self.depthwiseconv(conv2, 64, beta, 2, (3,3), 'conv3_dw')
        conv3 = self.conv_2d(conv3_dw, 128, alpha, 1, (1,1), 'conv3')

 
        # Conv4a_dw / Conv4a
        conv4a_dw = self.depthwiseconv(conv3, 128, beta, 2, (3,3), 'conv4a_dw')
        conv4a = self.conv_2d(conv4a_dw, 256, alpha, 1, (1,1), 'conv4a')


        # Conv4b_dw / Conv4b1
        conv4b_dw = self.depthwiseconv(conv4a, 256, beta, 1, (3,3), 'conv4b_dw')
        conv4b = self.conv_2d(conv4b_dw, 256, alpha, 1, (1,1), 'conv4b')


        # Conv5a_dw / Conv5a
        conv5a_dw = self.depthwiseconv(conv4b, 256, beta, 2, (3,3), 'conv5a_dw')
        conv5a = self.conv_2d(conv5a_dw, 512, alpha, 1, (1,1), 'conv5a' )


        # Conv5b_dw / Conv5b
        conv5b_dw = self.depthwiseconv(conv5a, 512, beta, 1, (3,3), 'conv5b_dw')
        conv5b = self.conv_2d(conv5b_dw, 512, alpha, 1, (1,1), 'conv5b')

        # Conv6a_dw / Conv6a
        conv6a_dw = self.depthwiseconv(conv5b, 512, beta, 2, (3,3), 'conv6a_dw')
        conv6a = self.conv_2d(conv6a_dw, 1024, alpha, 1, (1,1), 'conv6a')

        # Conv6b_dw / Conv6b
        conv6b_dw = self.depthwiseconv(conv6a, 1024, beta, 1, (3,3), 'conv6b_dw')
        conv6b = self.conv_2d(conv6b_dw, 1024, alpha, 1, (1,1), 'conv6b')

        #################################################
        # DECODER 
        #################################################
        
        # Conv7_dw / Conv7
        conv7_dw = self.depthwiseconv(conv6b, 1024, beta, 1, (3,3), 'conv7_dw')
        conv7 = self.conv_2d(conv7_dw, 256, alpha, 1, (1,1), 'conv7')

        # Conv8_dw /Conv8
        # Operations
        conv7_resized_tensor = resize_like(conv7, conv7, 2)
        concat_op1 = tf.concat([conv7_resized_tensor, conv5b], axis=_concat_axis, name='concat_op1')
        # Convolutions
        conv8_dw = self.depthwiseconv(concat_op1, 768, beta, 1, (3,3), 'conv8_dw')
        conv8 = self.conv_2d(conv8_dw, 128, alpha, 1, (1,1), 'conv8')

        # Conv9_dw /Conv9
        # Operations
        conv8_resized_tensor = resize_like(conv8, conv8, 2)
        concat_op2 = tf.concat([conv8_resized_tensor, conv4b], axis=_concat_axis, name='concat_op2')
        #Convolutions
        conv9_dw = self.depthwiseconv(concat_op2, 384, beta, 1, (3,3), 'conv9_dw')
        conv9 = self.conv_2d(conv9_dw, 64, alpha, 1, (1,1), 'conv9')

        # Conv10_dw / Conv10
        # Operations
        coonv9_resized_tensor = resize_like(conv9, conv9, 2)
        concat_op3 = tf.concat([coonv9_resized_tensor, conv3], axis=_concat_axis, name='concat_op3')
        #Convolutions
        conv10_dw = self.depthwiseconv(concat_op3, 192, beta, 1, (3,3), 'conv10_dw')
        conv10 = self.conv_2d(conv10_dw, 32, alpha, 1, (1,1), 'conv10')

        # Conv11_dw / Con11
        # Operations
        conv10_resized_tensor = resize_like(conv10, conv10, 2)
        concat_op3 = tf.concat([conv10_resized_tensor, conv2], axis=_concat_axis, name='concat_op3')
        # Convolutions
        conv11_dw = self.depthwiseconv(concat_op3, 96, beta, 1, (3,3), 'conv11_dw')
        conv11 = self.conv_2d(conv11_dw, 16, alpha, 1, (1,1),'conv11')


        ##################################################
        # Optical Flow Predictions
        ##################################################

        # Conv12_dw / conv12
        conv12_dw = self.depthwiseconv(conv7, 256, beta, 1, (3,3), 'conv12_dw')
        conv12 = self.conv_2d(conv12_dw, 2, alpha, 1, (1,1), 'conv12')
        
        # Conv13_dw / conv13
        conv13_dw = self.depthwiseconv(conv8, 128, beta, 1, (3,3), 'conv13_dw')
        conv13 = self.conv_2d(conv13_dw, 2, alpha, 1, (1,1), 'conv13')

        # Conv14_dw / conv14
        conv14_dw = self.depthwiseconv(conv9, 64, beta, 1, (3,3), 'conv14_dw')
        conv14 = self.conv_2d(conv14_dw, 2, alpha, 1, (1,1), 'conv14')

        # Conv15_dw / con15
        conv15_dw = self.depthwiseconv(conv10, 32, beta, 1, (3,3), 'conv15_dw')
        covn15 = self.conv_2d(conv15_dw, 2, alpha, 1, (1,1), 'conv15')

        # Conv16_dw / conv16
        conv16_dw = self.depthwiseconv(conv11, 16, beta, 1, (3,3), 'conv16_dw')
        conv16 = self.conv_2d(conv16_dw, 2, alpha, 1, (1,1), 'conv16')


        ###################################################
        # Multiple Optical Flow Predictions Fusion
        ###################################################
        conv12_resized_tensor_x16 = resize_like(conv12, conv12, 16)
        conv13_resized_tensor_x8  = resize_like(conv13, conv13, 8)
        conv14_resized_tensor_x4  = resize_like(conv14, conv14, 4)
        conv15_resized_tensor_x2  = resize_like(conv15, conv15, 2)
        
        # 96x128x2
        flow = Average(name='average_layer')([conv12_resized_tensor_x16, 
                            conv13_resized_tensor_x8, 
                            conv14_resized_tensor_x4,
                            conv15_resized_tensor_x2, 
                            conv16])

        #flow = tf.image.resize_bilinear(average,
        #                tf.stack([height, width]), 
        #                align_corners=True)
        
        # Fuse groundtrunth resolution with prediction
        #flow = Lambda(resize_like, arguments={'ref_tensor':average, 'scale': 4})(average)
    
        return {
            'inputs': concat_inputs,
            'flow': flow
        }

    @staticmethod
    def depthwiseconv(inputs, num_pwc_filters, depth_multiplier, stride, kernel, sc):
        """ Helper function to build the depth-wise separable convolution layer.
        """
        # skip pointwise by setting num_outputs=None
        depthwise_conv = slim.separable_convolution2d(inputs,
                                                    num_outputs=None,
                                                    stride=stride,
                                                    depth_multiplier=depth_multiplier,
                                                    kernel_size=kernel,
                                                    scope=sc+'/depthwise_conv',
                                                    activation_fn=None)
        # Set BN
        bn = slim.batch_norm(depthwise_conv, scope=sc+'/dw_batch_norm')
        
        #depthwise_conv = tf.nn.leaky_relu(bn, alpha=0.1, name=sc + '/leaky_relu')
        depthwise_conv = LeakyReLU(bn, leak=0.1, name= sc + '/leaky_relu')
        return depthwise_conv

    @staticmethod
    def conv_2d(features, num_pwc_filters, width_multiplier, stride, kernel_size, sc):
        num_pwc_filters = round(num_pwc_filters * width_multiplier)

        _stride = stride #2 if downsample else 1

        conv = slim.convolution2d(features,
                                            num_pwc_filters,
                                            kernel_size=kernel_size,
                                            scope=sc+'/pointwise_conv',
                                            activation_fn=None)
        bn = slim.batch_norm(conv, scope=sc+'/pw_batch_norm')
        conv= LeakyReLU(bn, leak=0.1, name= sc + '/leaky_relu')
        return conv
        
    def loss(self, flow, predictions):
        # L2 loss between predict_flow, concat_input(img_a,img_b)
        predicted_flow = predictions['flow']
        size = [predicted_flow.shape[1], predicted_flow.shape[2]]
        downsampled_flow = downsample(flow, size)
        loss = average_endpoint_error(downsampled_flow, predicted_flow)
        tf.losses.add_loss(loss)

        # Return the 'total' loss: loss fns + regularization terms defined in the model
        return tf.losses.get_total_loss()
