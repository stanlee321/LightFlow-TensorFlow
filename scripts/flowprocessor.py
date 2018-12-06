import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
from PIL import Image
import logging
from pathlib import Path
import click



class ImageProcessor(object):
    """performs object detection on an image
    """

    def __init__(self, path_to_model=None):
        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        self._path_to_model = path_to_model
        # strings used to add correct label for each box.
        self._detection_graph = None
        self._logger = None
        self._session = None
        self.image_tensor = None
        self.flow = None
        self.output_flow = None
    def setup(self):
        self.load_model(self._path_to_model)
        # run a detection once, because first model run is always slow
        IMAGE_HEIGHT = 384
        IMAGE_WIDTH = 512
        input_a = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        input_b = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.detect(input_a, input_b)

    def load_model(self, path):
        """load saved model from protobuf file
        """
        with tf.gfile.GFile(path, 'rb') as fid:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fid.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
        self._detection_graph = graph
        self._session = tf.Session(graph=self._detection_graph)
        # Definite input and output Tensors for detection_graph


        self.image_tensor = self._detection_graph.get_tensor_by_name("input_1:0")
        # Each box represents a part of the image where a particular object was detected.
        self.output_flow = self._detection_graph.get_tensor_by_name("average/truediv:0")

    def load_image_into_numpy_array(self, path, scale=1.0):
        """load image into NxNx3 numpy array
        """
        image = Image.open(path)
        image = image.resize(tuple(int(scale * dim) for dim in image.size))
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    def detect(self, input_a, input_b):
        """detect objects in the image
        """
        concat_inputs = np.concatenate([input_a, input_b], axis=2)
        concat_inputs = np.expand_dims(concat_inputs, axis=0)
        concat_inputs = concat_inputs.astype(np.uint8)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        # image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        self.flow = self._session.run(self.output_flow,
            feed_dict={self.image_tensor: concat_inputs})[0, :, :, :]
            
        return self.flow
    @property
    def labels(self):
        return self._labels

    def close(self):
        self._session.close()
