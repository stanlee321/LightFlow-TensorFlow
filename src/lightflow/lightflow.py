import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.keras.layers import Average

# Import Own Lib
from ..net import Net, Mode
from ..utils import  average_endpoint_error, LeakyReLU

try:
    from ..downsample import downsample
except Exception as e:
    print('Running without GPU')
    downsample = None


def resize_like(input_tensor, ref_tensor, scale): # resizes input tensor wrt. ref_tensor
    H, W = ref_tensor.get_shape()[1], ref_tensor.get_shape()[2]
    return tf.image.resize_nearest_neighbor(input_tensor, [H.value*scale, W.value*scale])


class LightFlow(Net):
    def __init__(self, mode=Mode.TRAIN, debug=False):
        
        super(LightFlow, self).__init__(mode=mode, debug=debug)

    def model(self, inputs, training_schedule, trainable=True):
        _concat_axis = 3
        alpha = 1.0
        beta = 1.0
        # Expected INPUT_SHAPE = (384, 512,6)
        concat_inputs = tf.concat([inputs['input_a'], inputs['input_b']], axis=_concat_axis, name='inputs')
        
        is_training = trainable
        weights_regularizer = slim.l2_regularizer(training_schedule['weight_decay'])

        with tf.variable_scope('LightFlow', [concat_inputs]):
            # Fusion Network
            with tf.variable_scope('encoder'):
                # #################################################
                # ENCODER 
                # #################################################

                # Conv1_dw / Conv1
                conv1 = LightFlow._depthwise_separable_conv(concat_inputs, 32, beta, alpha, 2, sc='conv1', batchnorm_istraining=is_training, l2_reg=weights_regularizer)

                # Conv2_dw / Conv2
                conv2 = LightFlow._depthwise_separable_conv(conv1, 64, beta, alpha, 2, sc='conv2', batchnorm_istraining=is_training, l2_reg=weights_regularizer)
                
                # Conv3_dw / Conv3
                conv3 = LightFlow._depthwise_separable_conv(conv2, 128, beta, alpha, 2, sc='conv3', batchnorm_istraining=is_training, l2_reg=weights_regularizer)
        
                # Conv4a_dw / Conv4a
                conv4a = LightFlow._depthwise_separable_conv(conv3, 256, beta, alpha, 2, sc='conv4a', batchnorm_istraining=is_training, l2_reg=weights_regularizer)

                # Conv4b_dw / Conv4b1
                conv4b = LightFlow._depthwise_separable_conv(conv4a, 256, beta, alpha, 1, sc='conv4b', batchnorm_istraining=is_training, l2_reg=weights_regularizer)

                # Conv5a_dw / Conv5a
                conv5a = LightFlow._depthwise_separable_conv(conv4b, 512, beta, alpha, 2, sc='conv5a', batchnorm_istraining=is_training, l2_reg=weights_regularizer)

                # Conv5b_dw / Conv5b
                conv5b = LightFlow._depthwise_separable_conv(conv5a, 512, beta, alpha, 1, sc='conv5b', batchnorm_istraining=is_training, l2_reg=weights_regularizer)

                # Conv6a_dw / Conv6a
                conv6a = LightFlow._depthwise_separable_conv(conv5b, 1024, beta, alpha, 2, sc='conv6a', batchnorm_istraining=is_training, l2_reg=weights_regularizer)

                # Conv6b_dw / Conv6b
                conv6b = LightFlow._depthwise_separable_conv(conv6a, 1024, beta, alpha, 1, sc='conv6b', batchnorm_istraining=is_training, l2_reg=weights_regularizer)

            with tf.variable_scope('decoder'):    
                #################################################
                # DECODER 
                #################################################
                
                # Conv7_dw / Conv7
                conv7 = LightFlow._depthwise_separable_conv(conv6b, 256, beta, alpha, 1, sc='conv7', batchnorm_istraining=is_training, l2_reg=weights_regularizer)

                # Conv8_dw /Conv8
                # Operations
                conv7_x2 = resize_like(conv7, conv7, 2)
                concat_op1 = tf.concat([conv7_x2, conv5b], axis=_concat_axis, name='concat_op1')
                # Convolutions
                conv8 =  LightFlow._depthwise_separable_conv(concat_op1, 128, beta, alpha, 1, sc='conv8', batchnorm_istraining=is_training, l2_reg=weights_regularizer)

                # Conv9_dw /Conv9
                # Operations
                conv8_x2 = resize_like(conv8, conv8, 2)
                concat_op2 = tf.concat([conv8_x2, conv4b], axis=_concat_axis, name='concat_op2')
                #Convolutions   
                conv9 =  LightFlow._depthwise_separable_conv(concat_op2, 64, beta, alpha, 1, sc='conv9', batchnorm_istraining=is_training, l2_reg=weights_regularizer)

                # Conv10_dw / Conv10
                # Operations
                conv9_x2 = resize_like(conv9, conv9, 2)
                concat_op3 = tf.concat([conv9_x2, conv3], axis=_concat_axis, name='concat_op3')
                #Convolutions
                conv10 =  LightFlow._depthwise_separable_conv(concat_op3, 32, beta, alpha, 1, sc='conv10', batchnorm_istraining=is_training, l2_reg=weights_regularizer)

                # Conv11_dw / Con11
                # Operations
                conv10_x2 = resize_like(conv10, conv10, 2)
                concat_op4 = tf.concat([conv10_x2, conv2], axis=_concat_axis, name='concat_op4')
                # Convolutions
                conv11 =  LightFlow._depthwise_separable_conv(concat_op4, 16, beta, alpha, 1, sc='conv11', batchnorm_istraining=is_training, l2_reg=weights_regularizer)

            with tf.variable_scope('optical_flow_pred'):    

                ##################################################
                # Optical Flow Predictions
                ##################################################

                # Conv12_dw / conv12
                conv12 =  LightFlow._depthwise_separable_conv(conv7, 2, beta, alpha, 1, sc='conv12', batchnorm_istraining=is_training, l2_reg=weights_regularizer)

                # Conv13_dw / conv13
                conv13 =  LightFlow._depthwise_separable_conv(conv8, 2, beta, alpha, 1, sc='conv13', batchnorm_istraining=is_training, l2_reg=weights_regularizer)

                # Conv14_dw / conv14
                conv14 =  LightFlow._depthwise_separable_conv(conv9, 2, beta, alpha, 1, sc='conv14', batchnorm_istraining=is_training, l2_reg=weights_regularizer)

                # Conv15_dw / con15
                conv15 =  LightFlow._depthwise_separable_conv(conv10, 2, beta, alpha, 1, sc='conv15', batchnorm_istraining=is_training, l2_reg=weights_regularizer)

                # Conv16_dw / conv16
                conv16 =  LightFlow._depthwise_separable_conv(conv11, 2, beta, alpha, 1, sc='conv16', batchnorm_istraining=is_training, l2_reg=weights_regularizer)

                ###################################################
                # Multiple Optical Flow Predictions Fusion
                ###################################################
                conv12_x16 = resize_like(conv12, conv12, 16)
                conv13_x8  = resize_like(conv13, conv13, 8)
                conv14_x4  = resize_like(conv14, conv14, 4)
                conv15_x2  = resize_like(conv15, conv15, 2)
                
                # 96x128x2
                flow = Average(name='average_layer')([
                    conv12_x16, 
                    conv13_x8, 
                    conv14_x4,
                    conv15_x2, 
                    conv16])

                # Fuse groundtrunth resolution with prediction
                #flow = tf.image.resize_bilinear(average, tf.stack([height, width]), align_corners=True)
                #flow = Lambda(resize_like, arguments={'ref_tensor':average, 'scale': 4})(average)
            
            return {
                'inputs': concat_inputs,
                'flow': flow
                }
    
    @staticmethod
    def _depthwise_separable_conv(inputs, num_pwc_filters, depth_multiplier, width_multiplier, stride, sc, batchnorm_istraining=None, l2_reg=None):
        # Helper function to build the depth-wise separable convolution layer.
        num_pwc_filters = round(num_pwc_filters * width_multiplier)

        # skip pointwise by setting num_outputs=None
        depthwise_conv_logit = slim.separable_convolution2d(inputs, num_outputs=None,
                                                    stride=stride,
                                                    depth_multiplier=depth_multiplier,
                                                    kernel_size=[3,3],
                                                    scope=sc+'/depthwise_conv',
                                                    activation_fn=None,
                                                    trainable=batchnorm_istraining,
                                                    weights_initializer=slim.variance_scaling_initializer())
        # Set BN
        bn = LightFlow.batch_norm(depthwise_conv_logit, batchnorm_istraining, sc+'/dw_batch_norm')
        
        depthwise_conv = tf.nn.leaky_relu(bn, alpha=0.1)

        pointwise_conv_logit = slim.convolution2d(depthwise_conv, num_pwc_filters,
                                            kernel_size=[1,1],
                                            scope=sc+'/pointwise_conv',
                                            weights_regularizer=l2_reg,
                                            activation_fn=None,
                                            trainable=batchnorm_istraining,
                                            weights_initializer=slim.variance_scaling_initializer())
        # Set BN
        bn = LightFlow.batch_norm(pointwise_conv_logit, batchnorm_istraining, sc+'/pw_batch_norm')
        pointwise_conv = tf.nn.leaky_relu(bn, alpha=0.1)
        return pointwise_conv

    @staticmethod
    def batch_norm(inputs, is_training, name):
        normalized = tf.layers.batch_normalization(
            inputs=inputs,
            axis=-1,
            momentum=0.9,
            epsilon=0.001,
            center=True,
            scale=True,
            training=is_training,
            name=name
        )
        return normalized


    def loss(self, flow, predictions):
        # L2 loss between predict_flow, concat_input(img_a,img_b)
        predicted_flow = predictions['flow']
        size = [predicted_flow.shape[1], predicted_flow.shape[2]]
        downsampled_flow = downsample(flow, size)
        loss = average_endpoint_error(downsampled_flow, predicted_flow)
        tf.losses.add_loss(loss)

        # Return the 'total' loss: loss fns + regularization terms defined in the model
        return tf.losses.get_total_loss()