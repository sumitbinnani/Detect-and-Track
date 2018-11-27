import cv2


def draw_box_label(img, bbox_cv2, box_color=(0, 255, 255), show_label=True):
    """
    Draw Bounding Box
    :param img: input image
    :param bbox_cv2: left, top, right, bottom
    :param box_color: color of the box
    :param show_label: whether to show labels
    :return: img with bounding box
    """
    # box_color= (0, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.7
    font_color = (0, 0, 0)
    left, top, right, bottom = bbox_cv2[1], bbox_cv2[0], bbox_cv2[3], bbox_cv2[2]

    # Draw the bounding box
    cv2.rectangle(img, (left, top), (right, bottom), box_color, 1)

    if show_label and False:
        # Draw a filled box on top of the bounding box (as the background for the labels)
        cv2.rectangle(img, (left - 2, top - 45), (right + 2, top), box_color, -1, 1)

        # Output the labels that show the x and y coordinates of the bounding box center.
        text_x = 'x=' + str((left + right) / 2)
        cv2.putText(img, text_x, (left, top - 25), font, font_size, font_color, 1, cv2.LINE_AA)
        text_y = 'y=' + str((top + bottom) / 2)
        cv2.putText(img, text_y, (left, top - 5), font, font_size, font_color, 1, cv2.LINE_AA)

    return img
