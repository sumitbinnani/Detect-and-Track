import cv2

from tracking.tracker import Tracker


def draw_box_label(img, tracker: Tracker, show_label=True):
    """
    Draw Bounding Box
    :param img: input image
    :param bbox_cv2: left, top, right, bottom
    :param box_color: color of the box
    :param show_label: whether to show labels
    :return: img with bounding box
    """
    # box_color= (0, 255, 255)
    box_colors = {3: (0, 255, 255), 1: (0, 0, 255)}
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.7
    font_color = (0, 0, 0)
    box_color = box_colors[tracker.unit_object.class_id]
    bbox_cv2 = tracker.unit_object.box

    left, top, right, bottom = bbox_cv2[1], bbox_cv2[0], bbox_cv2[3], bbox_cv2[2]

    # Draw the bounding box
    cv2.rectangle(img, (left, top), (right, bottom), box_color, 2)

    if show_label:
        # Draw a filled box on top of the bounding box (as the background for the labels)
        cv2.rectangle(img, (left - 2, top - 45), (right + 2, top), box_color, -1, 1)

        text = str(tracker.tracking_id)
        cv2.putText(img, text, (left, top - 5), font, font_size, font_color, 1, cv2.LINE_AA)

    return img
