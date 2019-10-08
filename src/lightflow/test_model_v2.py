import tensorflow as tf
from tensorflow.keras import Input

from lightflow import LightFlow

def test_params():
    # add model to graph

    concat_inputs = Input((384, 512, 6))

    model = LightFlow()
    model(concat_inputs)

    model.summary()
    # print(model)
    # tf.keras.utils.plot_model(model, 'lightflow.png', show_shapes=True)

def test_model():
    lightflow = LightFlow()

    temp_input = tf.ones(shape=(1, 384, 512, 3))
    temp_output = tf.ones(shape=(1, 384, 512, 3))

    concat = tf.concat([temp_input, temp_output], axis=3)
    print(concat.shape)
    fn_out = lightflow(concat)

    print(fn_out.shape)  # (batch_size, width, height, channels)
if __name__ == '__main__':
    test_model()

    test_params()