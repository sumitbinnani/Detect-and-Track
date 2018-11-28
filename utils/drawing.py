import os
import pickle

import cv2

from tracking.tracker import Tracker

pallete_path = os.path.join(os.path.dirname(__file__), "pallete")
box_colors = pickle.load(open(pallete_path, "rb"))


def draw_box_label(img, tracker: Tracker, class_names, show_label=True):
    """
    Draw Bounding Box
    :param img: input image
    :param tracker: left, top, right, bottom
    :param class_names: name of the class
    :param show_label: whether to show labels
    :return: img with bounding box
    """
    unit_object = tracker.unit_object
    x = unit_object.box
    c1 = (x[1], x[0])
    c2 = (x[3], x[2])
    cls = unit_object.class_id
    label = "{0}:{1}".format(tracker.tracking_id, class_names[cls])
    color = box_colors[cls]
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img
