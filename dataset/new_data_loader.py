import tensorflow as tf


def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    keys_to_features = {
        'image_a': tf.io.FixedLenFeature((), tf.string),
        'image_b': tf.io.FixedLenFeature((), tf.string),
        'flow': tf.io.FixedLenFeature((), tf.string),
    }

    return tf.io.parse_single_example(example_proto, keys_to_features)


def load_dataset(filename):
    # image_data = tf.io.gfile.GFile(filename, 'rb').read()
    raw_image_dataset = tf.data.TFRecordDataset(filename)
    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    print(parsed_image_dataset)

    for image_features in parsed_image_dataset:
        image_raw = image_features['image_a'].shape
        print(image_raw)
        #display.display(display.Image(data=image_raw))
if __name__ == '__main__':

    load_dataset("fc_sample.tfrecords")
