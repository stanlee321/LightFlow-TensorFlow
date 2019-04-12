from lightflow import LightFlow
import tensorflow as tf
from tensorflow.keras import Input

def test_params():
    # add model to graph

    concat_inputs = Input((384, 512, 6))


    model = LightFlow()
    model(concat_inputs)

    model.summary()
    # print(model)
    # tf.keras.utils.plot_model(model, 'lightflow.png', show_shapes=True)

def test_model():
    sample_lightflow = LightFlow()

    temp_input = tf.zeros(shape=(1, 384, 512, 3))
    temp_output = tf.zeros(shape=(1, 384, 512, 3))

    concat = tf.concat([temp_input, temp_output], axis=3)

    fn_out = sample_lightflow(concat)

    print(fn_out.shape)  # (batch_size, width, height, channels)
if __name__ == '__main__':
    test_model()    