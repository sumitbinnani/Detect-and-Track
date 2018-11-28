import logging

import numpy as np
import tensorflow as tf

from tracking import UnitObject
from typing import List
from .base_detector import BaseDetector

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.WARN)

PATH_TO_CKPT = '.models/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'


# category_index = {1: {'id': 1, 'name': u'person'},
#                   2: {'id': 2, 'name': u'bicycle'},
#                   3: {'id': 3, 'name': u'car'},
#                   4: {'id': 4, 'name': u'motorcycle'},
#                   5: {'id': 5, 'name': u'airplane'},
#                   6: {'id': 6, 'name': u'bus'},
#                   7: {'id': 7, 'name': u'train'},
#                   8: {'id': 8, 'name': u'truck'},
#                   9: {'id': 9, 'name': u'boat'},
#                   10: {'id': 10, 'name': u'traffic light'},
#                   11: {'id': 11, 'name': u'fire hydrant'},
#                   13: {'id': 13, 'name': u'stop sign'},
#                   14: {'id': 14, 'name': u'parking meter'}}

class Detector(BaseDetector):
    def __init__(self):
        super().__init__()

        self.classes_to_detect = [3]

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
                if (ratio < 0.8) and (box_h > 20) and (box_w > 20):
                    tmp_car_boxes.append(unit_object)
                    LOGGER.debug('valid box')
                else:
                    LOGGER.debug('wrong ratio or wrong size')

        if len(tmp_car_boxes) == 0:
            LOGGER.debug('no detector!')

        self.car_boxes = tmp_car_boxes

        return self.car_boxes
