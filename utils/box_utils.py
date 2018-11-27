import numpy as np


def calculate_iou(box1, box2):
    """
    Calculate intersection over union
    :param box1: a[0], a[1], a[2], a[3] <-> left, top, right, bottom
    :param box2: b[0], b[1], b[2], b[3] <-> left, top, right, bottom
    """

    w_intsec = np.maximum(0, (np.minimum(box1[2], box2[2]) - np.maximum(box1[0], box2[0])))
    h_intsec = np.maximum(0, (np.minimum(box1[3], box2[3]) - np.maximum(box1[1], box2[1])))
    s_intsec = w_intsec * h_intsec

    s_a = (box1[2] - box1[0]) * (box1[3] - box1[1])
    s_b = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return float(s_intsec) / (s_a + s_b - s_intsec)
