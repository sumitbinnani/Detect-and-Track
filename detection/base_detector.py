from abc import ABC, abstractmethod
from typing import List

import numpy as np

from pipeline import UnitObject


class BaseDetector(ABC):
    def __init__(self):
        self.car_boxes: List[UnitObject] = []
        self.class_names = {}

    @staticmethod
    def box_normal_to_pixel(box, dim):
        """
        Convert box to pixel co-ordinates
        :param box:
        :param dim:
        :return:
        """
        height, width = dim[0], dim[1]
        box_pixel = [int(box[0] * height), int(box[1] * width), int(box[2] * height), int(box[3] * width)]
        return np.array(box_pixel)

    @abstractmethod
    def get_detections(self, image, debug=False) -> List[UnitObject]:
        """
        Find location of car in the image
        :param image: input image
        :param debug: whether to print debug outputs
        :return: list of bounding boxes: coordinates [y_up, x_left, y_down, x_right]
        """
