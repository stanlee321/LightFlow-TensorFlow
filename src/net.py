import abc
from enum import Enum
import os
import time
import tensorflow as tf
from tensorflow.keras import backend as K
from .flowlib import flow_to_image, write_flow
import numpy as np
import numpy as np
from scipy.misc import imread, imsave
import uuid
from .training_schedules import LONG_SCHEDULE

slim = tf.contrib.slim


class Mode(Enum):
    TRAIN = 1
    TEST = 2


class Net(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, mode=Mode.TRAIN, debug=False):
        #self.global_step = slim.get_or_create_global_step()
        self.global_step = tf.train.get_or_create_global_step()

        self.mode = mode
        self.debug = debug

    @abc.abstractmethod
    def model(self, inputs, training_schedule, trainable=True):
        """
        Defines the model and returns a tuple of Tensors needed for calculating the loss.
        """
        return

    @abc.abstractmethod
    def loss(self, **kwargs):
        """
        Accepts prediction Tensors from the output of `model`.
        Returns a single Tensor representing the total loss of the model.
        """
        return

    def test(self, checkpoint, input_a_path, input_b_path, out_path, save_image=True, save_flo=False):
        input_a = imread(input_a_path)
        input_b = imread(input_b_path)

        # Convert from RGB -> BGR
        #input_a = input_a[..., [2, 1, 0]]
        #input_b = input_b[..., [2, 1, 0]]

        # Scale from [0, 255] -> [0.0, 1.0] if needed
        #if input_a.max() > 1.0:
        #    input_a = input_a / 255.0
        #if input_b.max() > 1.0:
        #    input_b = input_b / 255.0
            
        # TODO: This is a hack, we should get rid of this
        training_schedule = LONG_SCHEDULE

        inputs = {
            'input_a': tf.expand_dims(tf.constant(input_a, dtype=tf.float32), 0),
            'input_b': tf.expand_dims(tf.constant(input_b, dtype=tf.float32), 0),
        }
        predictions = self.model(inputs, training_schedule, trainable=False)
        pred_flow = predictions['flow']
        
        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, checkpoint)
            pred_flow = sess.run(pred_flow)[0, :, :, :]
            
            unique_name = 'flow-' + str(uuid.uuid4())
            if save_image:
                flow_img = flow_to_image(pred_flow)
                full_out_path = os.path.join(out_path, unique_name + '.png')
                imsave(full_out_path, flow_img)

            if save_flo:
                full_out_path = os.path.join(out_path, unique_name + '.flo')
                write_flow(pred_flow, full_out_path)

    def test_ckpt(self, checkpoint, input_a_path, input_b_path, output_path):

        input_a = imread(input_a_path)
        input_b = imread(input_b_path)

        # Convert from RGB -> BGR
        input_a = input_a[..., [2, 1, 0]]
        input_b = input_b[..., [2, 1, 0]]

        # Scale from [0, 255] -> [0.0, 1.0] if needed
        if input_a.max() > 1.0:
            input_a = input_a / 255.0
        if input_b.max() > 1.0:
            input_b = input_b / 255.0

        # TODO: This is a hack, we should get rid of this
        training_schedule = LONG_SCHEDULE
        # TODO: Remove this inputs since this are not used for this task of filter the 
        # Graph 
        inputs = {
            'input_a': tf.expand_dims(tf.constant(input_a, dtype=tf.float32), 0),
            'input_b': tf.expand_dims(tf.constant(input_b, dtype=tf.float32), 0),
        }
        graph = tf.Graph()
        with graph.as_default():
            predictions = self.model(inputs, training_schedule, trainable=False, build=True)
            pred_flow = predictions['flow']
            input_tensor = predictions['inputs']
            saver = tf.train.Saver(tf.global_variables())
            sess  = tf.Session()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())


            op = sess.graph.get_operations()

            print("in=", input_tensor.name)
            print("on=", pred_flow.name)

            saver.restore(sess, checkpoint)
            #############################
            # FIX BATCH NORM?
            """
            gd = sess.graph.as_graph_def()
            for node in gd.node:            
                if node.op == 'RefSwitch':
                    node.op = 'Switch'
                    for index in range(len(node.input)):
                        if 'moving_' in node.input[index]:
                            node.input[index] = node.input[index] + '/read'
                elif node.op == 'AssignSub':
                    node.op = 'Sub'
                    if 'use_locking' in node.attr: del node.attr['use_locking']
                elif node.op == 'AssignAdd':
                    node.op = 'Add'
                    if 'use_locking' in node.attr: del node.attr['use_locking']
                elif node.op == 'ReadVariableOp':
                    node.op = 'Switch'
                    for index in range(len(node.input)):
                        if 'moving_' in node.input[index]:
                            node.input[index] = node.input[index] + '/read'
            output_node_names = 'average/truediv'
            """
            ############################
            saver.save(sess, output_path + '/deployfinal.ckpt')

            graphdef = graph.as_graph_def()
            tf.train.write_graph(graphdef, output_path, 'lightflow.pbtxt', as_text=True)

    def train(self, log_dir, training_schedule, input_a, input_b, flow, checkpoints=None):
        tf.summary.image("image_a", input_a, max_outputs=2)
        tf.summary.image("image_b", input_b, max_outputs=2)

        self.learning_rate = tf.train.piecewise_constant(
            self.global_step,
            [tf.cast(v, tf.int64) for v in training_schedule['step_values']],
            training_schedule['learning_rates'])
        

        #optimizer = tf.train.AdamOptimizer(
        weight_decay_val = training_schedule['weight_decay']
        optimizer = tf.contrib.opt.AdamWOptimizer(
            weight_decay_val,
            self.learning_rate,
            training_schedule['momentum'],
            training_schedule['momentum2'])

        inputs = {
            'input_a': input_a,
            'input_b': input_b
        }
        predictions = self.model(inputs, training_schedule)
        total_loss = self.loss(flow, predictions)
        tf.summary.scalar('loss', total_loss)

        checkpoint_dir = './logs/lightflow/model.ckpt-9468'
        checkpoint_dir = './logs/lightflow/model.ckpt-75000'


        if checkpoints == 'latest':
            print('restoring...')
            restorer = tf.train.Saver(max_to_keep=100)
            with tf.Session() as sess:
                restorer.restore(sess, checkpoint_dir)
            """
            tf.train.latest_checkpoint(
            checkpoint_dir,
            latest_filename=None)
            """


        # Show the generated flow in TensorBoard
        if 'flow' in predictions:
            pred_flow_0 = predictions['flow'][0, :, :, :]
            pred_flow_0 = tf.py_func(flow_to_image, [pred_flow_0], tf.uint8)
            pred_flow_1 = predictions['flow'][1, :, :, :]
            pred_flow_1 = tf.py_func(flow_to_image, [pred_flow_1], tf.uint8)
            pred_flow_img = tf.stack([pred_flow_0, pred_flow_1], 0)
            tf.summary.image('pred_flow', pred_flow_img, max_outputs=2)

        true_flow_0 = flow[0, :, :, :]
        true_flow_0 = tf.py_func(flow_to_image, [true_flow_0], tf.uint8)
        true_flow_1 = flow[1, :, :, :]
        true_flow_1 = tf.py_func(flow_to_image, [true_flow_1], tf.uint8)
        true_flow_img = tf.stack([true_flow_0, true_flow_1], 0)
        tf.summary.image('true_flow', true_flow_img, max_outputs=2)

        train_op = slim.learning.create_train_op(
            total_loss,
            optimizer,
            summarize_gradients=True)

        if self.debug:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                tf.train.start_queue_runners(sess)
                slim.learning.train_step(
                    sess,
                    train_op,
                    self.global_step,
                    {
                        'should_trace': tf.constant(1),
                        'should_log': tf.constant(1),
                        'logdir': log_dir + '/debug',
                    }
                )
        else:
            with tf.Session() as sess:
                saver = tf.train.Saver(max_to_keep=50)
                sess.run(tf.global_variables_initializer())
                slim.learning.train(
                    train_op,
                    log_dir,
                    # session_config=tf.ConfigProto(allow_soft_placement=True),
                    global_step=self.global_step,
                    save_summaries_secs=60,
                    number_of_steps=training_schedule['max_iter']
                )