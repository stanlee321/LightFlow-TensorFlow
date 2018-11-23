import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Add, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, LeakyReLU
from tensorflow.keras.layers import BatchNormalization, Lambda
from tensorflow.keras.layers import Concatenate, UpSampling2D 
from tensorflow.image import resize_nearest_neighbor
from tensorflow.keras import Model
from keras.utils.vis_utils import plot_model
from keras.engine.topology import get_source_inputs
# Import Own Lib
from .depthwise_conv2d import DepthwiseConvolution2D


def _depthwise_convolution2D(input, alpha, deepwise_filter_size, kernel_size, strides, padding='same', bias=False):
    x= DepthwiseConvolution2D(int(deepwise_filter_size * alpha), kernel_size, strides=strides, padding=padding, use_bias=bias)(input)
    x= BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return  x

def _convolution2D(input, alpha, deepwise_filter_size, kernel_size, strides, padding='same', bias=False):
    x = Conv2D(int(deepwise_filter_size * alpha), kernel_size, strides=strides, padding=padding, use_bias=bias)(input)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
   
    return x

def resize_like(input_tensor, ref_tensor, scale): # resizes input tensor wrt. ref_tensor
    H, W = ref_tensor.get_shape()[1], ref_tensor.get_shape()[2]
    return tf.image.resize_nearest_neighbor(input_tensor, [H.value*scale, W.value*scale])

def average_endpoint_error(labels, predictions):
    """
    Given labels and predictions of size (N, H, W, 2), calculates average endpoint error:
        sqrt[sum_across_channels{(X - Y)^2}]
    """
    num_samples = predictions.shape.as_list()[0]
    with tf.name_scope(None, "average_endpoint_error", (predictions, labels)) as scope:
        predictions = tf.to_float(predictions)
        labels = tf.to_float(labels)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())

        squared_difference = tf.square(tf.subtract(predictions, labels))
        # sum across channels: sum[(X - Y)^2] -> N, H, W, 1
        loss = tf.reduce_sum(squared_difference, 3, keep_dims=True)
        loss = tf.sqrt(loss)
        return tf.reduce_sum(loss) / num_samples


class LightFlow:

    def __init__(self):
        pass

    @staticmethod
    def build(input_tensor = None, input_shape=None, plot=False):
       
        input_image = Input(shape=input_shape)
        concat_axis = 3
        alpha = 1.0

        #################################################
        # ENCODER 
        #################################################
       
        # Conv1_dw / Conv1

        conv1_dw = _depthwise_convolution2D(input_image, alpha, 6,  (3, 3), strides=(2, 2))
        conv1 = _convolution2D(conv1_dw, alpha, 32, (1, 1), strides=(1, 1))

        # Conv2_dw / Conv2
        conv2_dw = _depthwise_convolution2D(conv1, alpha, 32,  (3, 3), strides=(2, 2))
        conv2 = _convolution2D(conv2_dw, alpha, 64, (1, 1), strides=(1, 1) )
        
        # Conv3_dw / Conv3
        conv3_dw = _depthwise_convolution2D(conv2, alpha, 64,  (3, 3), strides=(2, 2))
        conv3 = _convolution2D(conv3_dw, alpha, 128, (1, 1), strides=(1, 1))

        # Conv4a_dw / Conv4a
        conv4a_dw = _depthwise_convolution2D(conv3, alpha, 128,  (3, 3), strides=(2, 2))
        conv4a = _convolution2D(conv4a_dw, alpha, 256, (1, 1), strides=(1, 1) )

        # Conv4b_dw / Conv4b
        conv4b_dw = _depthwise_convolution2D(conv4a, alpha, 256,  (3, 3), strides=(1, 1))
        conv4b = _convolution2D(conv4b_dw, alpha, 256, (1, 1), strides=(1, 1))

        # Conv5a_dw / Conv5a
        conv5a_dw = _depthwise_convolution2D(conv4b, alpha, 256,  (3, 3), strides=(2, 2))
        conv5a = _convolution2D(conv5a_dw, alpha, 512, (1, 1), strides=(1, 1))

        # Conv5b_dw / Conv5b
        conv5b_dw = _depthwise_convolution2D(conv5a, alpha, 512,  (3, 3), strides=(1, 1))
        conv5b = _convolution2D(conv5b_dw, alpha, 512, (1, 1), strides=(1, 1))

        # Conv6a_dw / Conv6a
        conv6a_dw = _depthwise_convolution2D(conv5b, alpha, 512,  (3, 3), strides=(2, 2))
        conv6a = _convolution2D(conv6a_dw, alpha, 1024, (1, 1), strides=(1, 1))

        # Conv6b_dw / Conv6b
        conv6b_dw = _depthwise_convolution2D(conv6a, alpha, 1024,  (3, 3), strides=(1, 1))
        conv6b =  _convolution2D(conv6b_dw, alpha, 1024, (1, 1), strides=(1, 1))

        #################################################
        # DECODER 
        #################################################
        
        # Conv7_dw / Conv7
        conv7_dw =  _depthwise_convolution2D(conv6b, alpha, 1024,  (3, 3), strides=(1, 1))
        conv7 = _convolution2D(conv7_dw, alpha, 256, (1, 1), strides=(1, 1))

        # Conv8_dw /Conv8
        conv7_resized_tensor = Lambda(resize_like, arguments={'ref_tensor':conv7, 'scale': 2})(conv7)
        concat_op1 = Concatenate(axis=concat_axis)([conv7_resized_tensor, conv5b])
        
        conv8_dw = _depthwise_convolution2D(concat_op1, alpha, 768,  (3, 3), strides=(1, 1))
        conv8 = _convolution2D(conv8_dw, alpha, 128, (1, 1), strides=(1, 1))

        # Conv9_dw /Conv9
        conv8_resized_tensor = Lambda(resize_like, arguments={'ref_tensor':conv8, 'scale': 2})(conv8)
        concat_op2 = Concatenate(axis=concat_axis)([conv8_resized_tensor, conv4b])
        
        conv9_dw = _depthwise_convolution2D(concat_op2, alpha, 384, (3,3), strides=(1,1))
        conv9 = _convolution2D(conv9_dw, alpha, 64, (1,1), strides=(1,1))

        # Conv10_dw / Conv10
        coonv9_resized_tensor = Lambda(resize_like, arguments={'ref_tensor':conv9, 'scale': 2})(conv9)
        concat_op3 = Concatenate(axis=concat_axis)([coonv9_resized_tensor, conv3])

        conv10_dw = _depthwise_convolution2D(concat_op3, alpha, 192, (3,3), strides=(1,1))
        conv10 = _convolution2D(conv10_dw, alpha, 32, (1,1), strides=(1,1))

        # Conv11_dw / Con11
        conv10_resized_tensor = Lambda(resize_like, arguments={'ref_tensor':conv10, 'scale': 2})(conv10)
        concat_op3 = Concatenate(axis=concat_axis)([conv10_resized_tensor, conv2])

        conv11_dw = _depthwise_convolution2D(concat_op3, alpha, 96, (3,3), strides=(1,1))
        conv11 = _convolution2D(conv11_dw, alpha, 16, (1,1), strides=(1,1))


        ##################################################
        # Optical Flow Predictions
        ##################################################

        # Conv12_dw / conv12
        conv12_dw = _depthwise_convolution2D(conv7, alpha, 256, (3,3), strides = (1,1))
        conv12 = _convolution2D(conv12_dw, alpha, 2, (1,1), strides=(1,1))

        # Conv13_dw / conv13
        conv13_dw = _depthwise_convolution2D(conv8, alpha, 128, (3,3), strides = (1,1))
        conv13 = _convolution2D(conv13_dw, alpha, 2, (1,1), strides=(1,1))

        # Conv14_dw / conv14
        conv14_dw = _depthwise_convolution2D(conv9, alpha, 64, (3,3), strides=(1,1))
        conv14 = _convolution2D(conv14_dw, alpha, 2, (1,1), strides=(1,1))

        # Conv15_dw / con15
        conv15_dw = _depthwise_convolution2D(conv10, alpha, 32, (3,3), strides=(1,1))
        conv15 = _convolution2D(conv15_dw, alpha, 2, (1,1), strides=(1,1))

        # Conv16_dw / conv16
        conv16_dw = _depthwise_convolution2D(conv11, alpha, 16, (3,3), strides=(1,1))
        conv16 = _convolution2D(conv16_dw, alpha, 2, (1,1), strides=(1,1))


        ###################################################
        # Multiple Optical Flow Predictions Fusion
        ###################################################

        conv12_resized_tensor_x16 = Lambda(resize_like, arguments={'ref_tensor':conv12, 'scale': 16})(conv12)
        conv13_resized_tensor_x8 = Lambda(resize_like, arguments={'ref_tensor':conv13, 'scale': 8})(conv13)
        conv14_resized_tensor_x4 = Lambda(resize_like, arguments={'ref_tensor':conv14, 'scale': 4})(conv14)
        conv15_resized_tensor_x2 = Lambda(resize_like, arguments={'ref_tensor':conv15, 'scale': 2})(conv15)

        average = Add()([conv12_resized_tensor_x16, 
                                conv13_resized_tensor_x8, 
                                conv14_resized_tensor_x4 ,
                                conv15_resized_tensor_x2, 
                                conv16])

        # Fuse GT with prediction
        average = Lambda(resize_like, arguments={'ref_tensor':average, 'scale': 4})(average)
        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            input_image = get_source_inputs(input_tensor)
        else:
            pass

        # Create model for debug
        model = Model(inputs=input_image, outputs=average, name='lightflow')

        return model


def main(plot = True):
    INPUT_SHAPE = (384, 512,6)
    model = LightFlow.build(input_shape=INPUT_SHAPE)
    if plot is True:
        plot_model(model, to_file='LightFLow.png', show_shapes=True)
    print(model.summary())

if __name__ == '__main__':
    main()
