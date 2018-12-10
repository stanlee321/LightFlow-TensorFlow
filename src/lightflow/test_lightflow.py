from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .lightflow import LightFlow
from ..training_schedules import LONG_SCHEDULE
import tensorflow.contrib.slim as slim

def inspect_variables(variables):
    for var in variables:
        #print(var)
        print('name = {} {}shape = {}'.format(var.name, " "*(55-len(var.name)), var.get_shape()))
    print()

def inspect_layers(endpoints):
    for k, v in endpoints.iteritems():
        print('name = {} {}shape = {}'.format(v.name, " "*(55-len(v.name)), v.get_shape()))
    print()


def test_model():
    net = LightFlow()
    # ADd model to graph

    g = tf.Graph()
    with g.as_default():
        # 4D Tensor placeholder for input images
        inputs_a = tf.placeholder(tf.float32, shape=[None, 384, 512, 3], name="image_a")
        inputs_b = tf.placeholder(tf.float32, shape=[None, 384, 512, 3], name="image_b")
        inputs = {}
        inputs['input_a'] = inputs_a
        inputs['input_b'] = inputs_b
        # add model to graph
        outputs_dict = net.model(inputs, LONG_SCHEDULE )
        outputs = outputs_dict['flow']
        inputs = outputs_dict['inputs']
    # Inspect Variables
    with g.as_default():
        print("Parameters:")
        inspect_variables(slim.get_variables(scope="LightFlow"))
        print("Inputs/Outputs:")
        inspect_variables([inputs, outputs])

if __name__ == '__main__':
    test_model()