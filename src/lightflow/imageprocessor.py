import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
from PIL import Image
from object_detection.utils import label_map_util
import logging
from pathlib import Path
import click



class ImageProcessor(object):
    """performs object detection on an image
    """

    def __init__(self, path_to_model=None, path_to_labels=None, model_name=None):
        if model_name is None:
            model_name = 'models'
        if path_to_model is None:
            path_to_model = os.path.join(os.path.dirname(__file__), '..', model_name, 'frozen_inference_graph.pb')# 'ssdlite_mobilenet_v2_face_25_11_2018.pb')
        if path_to_labels is None:
            path_to_labels = os.path.join(os.path.dirname(__file__), '..', model_name, 'face.pbtxt')
        self._model_name = model_name
        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        self._path_to_model = path_to_model
        # strings used to add correct label for each box.
        self._path_to_labels = path_to_labels
        self._download_url = 'http://download.tensorflow.org/models/object_detection/'
        self._num_classes = 90
        self._detection_graph = None
        self._labels = dict()
        self._image = None
        self._boxes = None
        self._classes = None
        self._scores = None
        self._num = None
        self._logger = None
        self._session = None
        self.image_tensor = None
        self.detection_boxes = None
        self.detection_scores = None
        self.detection_classes = None
        self.num_detections = None

    def setup(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        self.load_model(self._path_to_model)
        # run a detection once, because first model run is always slow
        #self.detect(np.ones((150, 150, 3), dtype=np.uint8))

    def load_model(self, frozen_graph_filename):
        # We load the protobuf file from the disk and parse it to retrieve the 
        # unserialized graph_def
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we import the graph_def into a new Graph and returns it 
        with tf.Graph().as_default() as graph:
            # The name var will prefix every op/nodes in your graph
            # Since we load everything in a new graph, this is not needed
            tf.import_graph_def(graph_def, name="prefix")
        return graph

    def load_labels(self, path):
        """load labels from .pb file, and map to a dict with integers, e.g. 1=aeroplane
        """
        label_map = label_map_util.load_labelmap(path)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self._num_classes,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return category_index

    def detect(self, concat_image):
        """detect objects in the image
        """
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        
        # Actual detection.
        flow = self._session.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        return self._boxes, self._scores, self._classes, self._num

    def annotate_image(self, image, boxes, classes, scores, threshold=0.5):
        """draws boxes around the detected objects and labels them

        :return: annotated image
        """
        from object_detection.utils import visualization_utils as vis_util
        annotated_image = image.copy()
        vis_util.visualize_boxes_and_labels_on_image_array(
            annotated_image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self._labels,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=threshold)
        return annotated_image

    @property
    def labels(self):
        return self._labels

    def close(self):
        self._session.close()
