import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.keras.layers import Average

# Import Own Lib
from ..depthwise_conv2d import DepthwiseConvolution2D
from ..net import Net, Mode
from ..utils import  average_endpoint_error, LeakyReLU

try:
    from ..downsample import downsample
except Exception as e:
    print('Running from Computer without GPU')
    downsample = None


def resize_like(input_tensor, ref_tensor, scale): # resizes input tensor wrt. ref_tensor
    H, W = ref_tensor.get_shape()[1], ref_tensor.get_shape()[2]
    return tf.image.resize_nearest_neighbor(input_tensor, [H.value*scale, W.value*scale])


class LightFlow(Net):
    weight_reg = None
    def __init__(self, mode=Mode.TRAIN, debug=False):
        
        super(LightFlow, self).__init__(mode=mode, debug=debug)

    def model(self, inputs, training_schedule, trainable=True):
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
            # INPUT_SHAPE = (384, 512,6)
            concat_inputs = tf.concat([inputs['input_a'], inputs['input_b']], axis=_concat_axis, name='inputs')
        
        #is_training = tf.placeholder(tf.bool)
        is_training = trainable
        LightFlow.weight_reg = slim.l2_regularizer(training_schedule['weight_decay'])

        with tf.variable_scope('LightFlow', [concat_inputs]):
            
            with tf.variable_scope('encoder'):
                #################################################
                # ENCODER 
                #################################################
                # Conv1_dw / Conv1

                conv1_dw = self.depthwiseconv(concat_inputs, 6, beta, 2, (3,3), 'conv1_dw', is_training)
                conv1 =  self.conv_2d(conv1_dw, 32, alpha, 1, (1,1), 'conv1', is_training)
        
                # Conv2_dw / Conv2
                conv2_dw = self.depthwiseconv(conv1, 32, beta, 2, (3,3), 'conv2_dw', is_training)
                conv2 = self.conv_2d(conv2_dw, 64, alpha, 1, (1,1), 'conv2', is_training)
            
                # Conv3_dw / Conv3
                conv3_dw = self.depthwiseconv(conv2, 64, beta, 2, (3,3), 'conv3_dw', is_training)
                conv3 = self.conv_2d(conv3_dw, 128, alpha, 1, (1,1), 'conv3', is_training)

        
                # Conv4a_dw / Conv4a
                conv4a_dw = self.depthwiseconv(conv3, 128, beta, 2, (3,3), 'conv4a_dw', is_training)
                conv4a = self.conv_2d(conv4a_dw, 256, alpha, 1, (1,1), 'conv4a', is_training)


                # Conv4b_dw / Conv4b1
                conv4b_dw = self.depthwiseconv(conv4a, 256, beta, 1, (3,3), 'conv4b_dw', is_training)
                conv4b = self.conv_2d(conv4b_dw, 256, alpha, 1, (1,1), 'conv4b', is_training)


                # Conv5a_dw / Conv5a
                conv5a_dw = self.depthwiseconv(conv4b, 256, beta, 2, (3,3), 'conv5a_dw', is_training)
                conv5a = self.conv_2d(conv5a_dw, 512, alpha, 1, (1,1), 'conv5a', is_training)


                # Conv5b_dw / Conv5b
                conv5b_dw = self.depthwiseconv(conv5a, 512, beta, 1, (3,3), 'conv5b_dw', is_training)
                conv5b = self.conv_2d(conv5b_dw, 512, alpha, 1, (1,1), 'conv5b', is_training)

                # Conv6a_dw / Conv6a
                conv6a_dw = self.depthwiseconv(conv5b, 512, beta, 2, (3,3), 'conv6a_dw', is_training)
                conv6a = self.conv_2d(conv6a_dw, 1024, alpha, 1, (1,1), 'conv6a', is_training)

                # Conv6b_dw / Conv6b
                conv6b_dw = self.depthwiseconv(conv6a, 1024, beta, 1, (3,3), 'conv6b_dw', is_training)
                conv6b = self.conv_2d(conv6b_dw, 1024, alpha, 1, (1,1), 'conv6b', is_training)
            
            with tf.variable_scope('decoder'):    
                #################################################
                # DECODER 
                #################################################
                
                # Conv7_dw / Conv7
                conv7_dw = self.depthwiseconv(conv6b, 1024, beta, 1, (3,3), 'conv7_dw', is_training)
                conv7 = self.conv_2d(conv7_dw, 256, alpha, 1, (1,1), 'conv7', is_training)

                # Conv8_dw /Conv8
                # Operations
                conv7_resized_tensor = resize_like(conv7, conv7, 2)
                concat_op1 = tf.concat([conv7_resized_tensor, conv5b], axis=_concat_axis, name='concat_op1')
                # Convolutions
                conv8_dw = self.depthwiseconv(concat_op1, 768, beta, 1, (3,3), 'conv8_dw', is_training)
                conv8 = self.conv_2d(conv8_dw, 128, alpha, 1, (1,1), 'conv8', is_training)

                # Conv9_dw /Conv9
                # Operations
                conv8_resized_tensor = resize_like(conv8, conv8, 2)
                concat_op2 = tf.concat([conv8_resized_tensor, conv4b], axis=_concat_axis, name='concat_op2')
                #Convolutions
                conv9_dw = self.depthwiseconv(concat_op2, 384, beta, 1, (3,3), 'conv9_dw', is_training)
                conv9 = self.conv_2d(conv9_dw, 64, alpha, 1, (1,1), 'conv9', is_training)

                # Conv10_dw / Conv10
                # Operations
                coonv9_resized_tensor = resize_like(conv9, conv9, 2)
                concat_op3 = tf.concat([coonv9_resized_tensor, conv3], axis=_concat_axis, name='concat_op3')
                #Convolutions
                conv10_dw = self.depthwiseconv(concat_op3, 192, beta, 1, (3,3), 'conv10_dw', is_training)
                conv10 = self.conv_2d(conv10_dw, 32, alpha, 1, (1,1), 'conv10', is_training)

                # Conv11_dw / Con11
                # Operations
                conv10_resized_tensor = resize_like(conv10, conv10, 2)
                concat_op3 = tf.concat([conv10_resized_tensor, conv2], axis=_concat_axis, name='concat_op3')
                # Convolutions
                conv11_dw = self.depthwiseconv(concat_op3, 96, beta, 1, (3,3), 'conv11_dw', is_training)
                conv11 = self.conv_2d(conv11_dw, 16, alpha, 1, (1,1),'conv11', is_training)

            with tf.variable_scope('optical_flow_pred'):    

                ##################################################
                # Optical Flow Predictions
                ##################################################

                # Conv12_dw / conv12
                conv12_dw = self.depthwiseconv(conv7, 256, beta, 1, (3,3), 'conv12_dw', is_training)
                conv12 = self.conv_2d(conv12_dw, 2, alpha, 1, (1,1), 'conv12', is_training)
                
                # Conv13_dw / conv13
                conv13_dw = self.depthwiseconv(conv8, 128, beta, 1, (3,3), 'conv13_dw', is_training)
                conv13 = self.conv_2d(conv13_dw, 2, alpha, 1, (1,1), 'conv13', is_training)

                # Conv14_dw / conv14
                conv14_dw = self.depthwiseconv(conv9, 64, beta, 1, (3,3), 'conv14_dw', is_training)
                conv14 = self.conv_2d(conv14_dw, 2, alpha, 1, (1,1), 'conv14', is_training)

                # Conv15_dw / con15
                conv15_dw = self.depthwiseconv(conv10, 32, beta, 1, (3,3), 'conv15_dw', is_training)
                conv15 = self.conv_2d(conv15_dw, 2, alpha, 1, (1,1), 'conv15', is_training)

                # Conv16_dw / conv16
                conv16_dw = self.depthwiseconv(conv11, 16, beta, 1, (3,3), 'conv16_dw', is_training)
                conv16 = self.conv_2d(conv16_dw, 2, alpha, 1, (1,1), 'conv16', is_training)

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
                # Fuse groundtrunth resolution with prediction
                #flow = tf.image.resize_bilinear(average, tf.stack([height, width]), align_corners=True)
                #flow = Lambda(resize_like, arguments={'ref_tensor':average, 'scale': 4})(average)
            
            return {
                'inputs': concat_inputs,
                'flow': flow
                }

    @staticmethod
    def depthwiseconv(inputs, num_pwc_filters, depth_multiplier, stride, kernel, sc, batchnorm_istraining=None):
        """ Helper function to build the depth-wise separable convolution layer.
        """
        weights_regularizer = LightFlow.weight_reg
        # skip pointwise by setting num_outputs=None
        depthwise_conv = slim.separable_convolution2d(inputs,
                                                    num_outputs=None,
                                                    stride=stride,
                                                    depth_multiplier=depth_multiplier,
                                                    kernel_size=kernel,
                                                    scope=sc+'/depthwise_conv',
                                                    activation_fn=None,
                                                    weights_regularizer=weights_regularizer)
        # Set BN
        if batchnorm_istraining is not None:
            depthwise_conv = LightFlow.bn(depthwise_conv, batchnorm_istraining, sc+'/depthwise_conv' )
        #depthwise_conv = tf.nn.leaky_relu(bn, alpha=0.1, name=sc + '/leaky_relu')
        depthwise_conv = LeakyReLU(depthwise_conv, leak=0.1, name= sc + '/leaky_relu')
        return depthwise_conv

    @staticmethod
    def conv_2d(features, num_pwc_filters, width_multiplier, stride, kernel_size, sc, batchnorm_istraining=None):
        num_pwc_filters = round(num_pwc_filters * width_multiplier)

        _stride = stride #2 if downsample else 1
        weights_regularizer = LightFlow.weight_reg

        conv = slim.convolution2d(features,
                                num_pwc_filters,
                                kernel_size=kernel_size,
                                scope=sc+'/pointwise_conv',
                                activation_fn=None,
                                weights_regularizer = weights_regularizer)
        # Set BN
        if batchnorm_istraining is not False:
            conv = LightFlow.bn(conv, batchnorm_istraining, sc+'/pointwise_conv')
        conv= LeakyReLU(conv, leak=0.1, name= sc + '/leaky_relu')
        return conv
    @staticmethod
    def bn(inputs, is_training, sc):
        with tf.variable_scope(sc):
            normalized = tf.layers.batch_normalization(
                inputs=inputs,
                axis=-1,
                momentum=0.9,
                epsilon=0.001,
                center=True,
                scale=True,
                training=is_training,
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
