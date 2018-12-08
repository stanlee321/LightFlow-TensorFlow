import tensorflow as tf
from ..utils import  average_endpoint_error
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
        conv7_dw =  _depthwise_convolution2D(conv6b, alpha, 1024,  (3, 3), strides=(1, 1), training=trainable)
        conv7 = _convolution2D(conv7_dw, alpha, 256, (1, 1), strides=(1, 1) , training=trainable)

        # Conv8_dw /Conv8
        conv7_resized_tensor = Lambda(resize_like, arguments={'ref_tensor':conv7, 'scale': 2})(conv7)
        concat_op1 = Concatenate(axis=_concat_axis)([conv7_resized_tensor, conv5b])
        
        conv8_dw = _depthwise_convolution2D(concat_op1, alpha, 768,  (3, 3), strides=(1, 1), training=trainable)
        conv8 = _convolution2D(conv8_dw, alpha, 128, (1, 1), strides=(1, 1), training=trainable )

        # Conv9_dw /Conv9
        conv8_resized_tensor = Lambda(resize_like, arguments={'ref_tensor':conv8, 'scale': 2})(conv8)
        concat_op2 = Concatenate(axis=_concat_axis)([conv8_resized_tensor, conv4b])
        
        conv9_dw = _depthwise_convolution2D(concat_op2, alpha, 384, (3,3), strides=(1,1), training=trainable)
        conv9 = _convolution2D(conv9_dw, alpha, 64, (1,1), strides=(1,1), training=trainable)

        # Conv10_dw / Conv10
        coonv9_resized_tensor = Lambda(resize_like, arguments={'ref_tensor':conv9, 'scale': 2})(conv9)
        concat_op3 = Concatenate(axis=_concat_axis)([coonv9_resized_tensor, conv3])

        conv10_dw = _depthwise_convolution2D(concat_op3, alpha, 192, (3,3), strides=(1,1), training=trainable)
        conv10 = _convolution2D(conv10_dw, alpha, 32, (1,1), strides=(1,1), training=trainable )

        # Conv11_dw / Con11
        conv10_resized_tensor = Lambda(resize_like, arguments={'ref_tensor':conv10, 'scale': 2})(conv10)
        concat_op3 = Concatenate(axis=_concat_axis)([conv10_resized_tensor, conv2])

        conv11_dw = _depthwise_convolution2D(concat_op3, alpha, 96, (3,3), strides=(1,1), training=trainable)
        conv11 = _convolution2D(conv11_dw, alpha, 16, (1,1), strides=(1,1) , training=trainable)


        ##################################################
        # Optical Flow Predictions
        ##################################################

        # Conv12_dw / conv12
        conv12_dw = _depthwise_convolution2D(conv7, alpha, 256, (3,3), strides = (1,1), training=trainable)
        conv12 = _convolution2D(conv12_dw, alpha, 2, (1,1), strides=(1,1), training=trainable )

        # Conv13_dw / conv13
        conv13_dw = _depthwise_convolution2D(conv8, alpha, 128, (3,3), strides = (1,1), training=trainable)
        conv13 = _convolution2D(conv13_dw, alpha, 2, (1,1), strides=(1,1), training=trainable )

        # Conv14_dw / conv14
        conv14_dw = _depthwise_convolution2D(conv9, alpha, 64, (3,3), strides=(1,1), training=trainable)
        conv14 = _convolution2D(conv14_dw, alpha, 2, (1,1), strides=(1,1), training=trainable )

        # Conv15_dw / con15
        conv15_dw = _depthwise_convolution2D(conv10, alpha, 32, (3,3), strides=(1,1), training=trainable)
        conv15 = _convolution2D(conv15_dw, alpha, 2, (1,1), strides=(1,1), training=trainable )

        # Conv16_dw / conv16
        conv16_dw = _depthwise_convolution2D(conv11, alpha, 16, (3,3), strides=(1,1), training=trainable)
        conv16 = _convolution2D(conv16_dw, alpha, 2, (1,1), strides=(1,1), training=trainable)


        ###################################################
        # Multiple Optical Flow Predictions Fusion
        ###################################################

        conv12_resized_tensor_x16 = Lambda(resize_like, arguments={'ref_tensor':conv12, 'scale': 16})(conv12)
        conv13_resized_tensor_x8 = Lambda(resize_like, arguments={'ref_tensor':conv13, 'scale': 8})(conv13)
        conv14_resized_tensor_x4 = Lambda(resize_like, arguments={'ref_tensor':conv14, 'scale': 4})(conv14)
        conv15_resized_tensor_x2 = Lambda(resize_like, arguments={'ref_tensor':conv15, 'scale': 2})(conv15)
        
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
