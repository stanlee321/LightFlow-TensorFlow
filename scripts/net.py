import abc
from enum import Enum
import os
import time
import tensorflow as tf
from tensorflow.keras import backend as K
from .flowlib import flow_to_image, write_flow
import numpy as np
from scipy.misc import imread, imsave
import uuid
from .training_schedules import LONG_SCHEDULE


def test(self, checkpoint, input_a_path, input_b_path, out_path, save_image=True, save_flo=False):
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
        saver.save(sess, output_path + '/deployfinal.ckpt')

        graphdef = graph.as_graph_def()
        tf.train.write_graph(graphdef, output_path, 'lightflow.pbtxt', as_text=True)

def train(self, log_dir, training_schedule, input_a, input_b, flow, checkpoints=None):


