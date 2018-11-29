import logging
import os
from typing import List

import numpy as np
import tensorflow as tf

from pipeline import UnitObject
from .base_detector import BaseDetector

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.WARN)

PATH_TO_CKPT = os.path.join(os.path.dirname(__file__), 'ssd_mobilenet_v1_coco_11_06_2017', 'frozen_inference_graph.pb')


class Detector(BaseDetector):
    def __init__(self):
        super().__init__()

        self.classes_to_detect = list(range(1, 15))
        self.class_names = {1: u'person',
                            2: u'bicycle',
                            3: u'car',
                            4: u'motorcycle',
                            5: u'airplane',
                            6: u'bus',
                            7: u'train',
                            8: u'truck',
                            9: u'boat',
                            10: u'traffic light',
                            11: u'fire hydrant',
                            13: u'stop sign',
                            14: u'parking meter'}

        self.detection_graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def get_localization(self, image, debug=False) -> List[UnitObject]:
        with self.detection_graph.as_default():
            image_expanded = np.expand_dims(image, axis=0)
            (boxes, scores, classes, num_detections) = self.sess.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: image_expanded})

            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes).astype(int).tolist()
            scores = np.squeeze(scores)

            tmp_car_boxes = []
            for (cls, score, box_) in zip(classes, scores, boxes):
                if cls not in self.classes_to_detect or score <= 0.3:
                    continue

                dim = image.shape[0:2]
                box = self.box_normal_to_pixel(box_, dim)
                unit_object = UnitObject(box, cls)

                box = unit_object.box
                box_h = box[2] - box[0]
                box_w = box[3] - box[1]
                ratio = box_h / (box_w + 0.01)

                LOGGER.debug(str(box) + ', confidence: ' + str(score) + 'ratio:' + str(ratio))
                if cls == 3 and (ratio < 0.8) and (box_h > 20) and (box_w > 20):
                    tmp_car_boxes.append(unit_object)
                    LOGGER.debug('valid box')
                elif (ratio > 0.6) and cls == 1:
                    tmp_car_boxes.append(unit_object)
                    LOGGER.debug('valid box')
                else:
                    LOGGER.debug('wrong ratio or wrong size')

        if len(tmp_car_boxes) == 0:
            LOGGER.debug('no detection!')

        self.car_boxes = tmp_car_boxes

        return self.car_boxes
