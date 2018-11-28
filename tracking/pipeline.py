import logging
from collections import deque

import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment

import utils.box_utils
import utils.drawing
from tracking.tracker import Tracker

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.WARN)


class DetectAndTrack:
    """
    Class that connects detector and tracking
    """

    def __init__(self, detector):
        self.max_age = 4
        self.min_hits = 1
        self.frame_count = 0
        self.tracker_list = []
        self.track_id_list = deque(list(map(str, range(100))))
        self.detector = detector

    def pipeline(self, img):
        """
        Pipeline to process detections and trackers
        :param img: current frame
        :return: frame with annotation
        """

        self.frame_count += 1

        unit_detections = self.detector.get_localization(img)  # measurement

        x_box = []

        for trk in self.tracker_list:
            x_box.append(trk.box)

        matched, unmatched_dets, unmatched_trks = self.assign_detections_to_trackers(x_box, unit_detections,
                                                                                     iou_thrd=0.3)

        LOGGER.debug('Detection: ' + str(unit_detections))
        LOGGER.debug('x_box: ' + str(x_box))
        LOGGER.debug('matched:' + str(matched))
        LOGGER.debug('unmatched_det:' + str(unmatched_dets))
        LOGGER.debug('unmatched_trks:' + str(unmatched_trks))

        # Matched Detections
        for trk_idx, det_idx in matched:
            z = unit_detections[det_idx].box
            z = np.expand_dims(z, axis=0).T
            tmp_trk = self.tracker_list[trk_idx]
            tmp_trk.kalman_filter(z)
            xx = tmp_trk.x_state.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            x_box[trk_idx] = xx
            tmp_trk.box = xx
            tmp_trk.hits += 1
            tmp_trk.no_losses = 0

        # Unmatched Detections
        for idx in unmatched_dets:
            z = unit_detections[idx].box
            z = np.expand_dims(z, axis=0).T
            tmp_trk = Tracker()  # Create a new tracker
            x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
            tmp_trk.x_state = x
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box = xx
            tmp_trk.id = self.track_id_list.popleft()  # assign an ID for the tracker
            self.tracker_list.append(tmp_trk)
            x_box.append(xx)

        # Unmatched trackers
        for trk_idx in unmatched_trks:
            tmp_trk = self.tracker_list[trk_idx]
            tmp_trk.no_losses += 1
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box = xx
            x_box[trk_idx] = xx

        # The list of tracks to be annotated
        good_tracker_list = []
        for trk in self.tracker_list:
            if (trk.hits >= self.min_hits) and (trk.no_losses <= self.max_age):
                good_tracker_list.append(trk)
                x_cv2 = trk.box
                img = utils.drawing.draw_box_label(img, x_cv2)  # Draw the bounding boxes on the

        # Manage Tracks to be deleted
        deleted_tracks = filter(lambda x: x.no_losses > self.max_age, self.tracker_list)

        for trk in deleted_tracks:
            self.track_id_list.append(trk.id)

        self.tracker_list = [x for x in self.tracker_list if x.no_losses <= self.max_age]

        return img

    @staticmethod
    def assign_detections_to_trackers(trackers, unit_detections, iou_thrd=0.3):
        """
        Matches Trackers and Detections
        :param trackers: trackers
        :param unit_detections: detections
        :param iou_thrd: threshold to qualify as a match
        :return: matches, unmatched_detections, unmatched_trackers
        """
        IOU_mat = np.zeros((len(trackers), len(unit_detections)), dtype=np.float32)
        for t, trk in enumerate(trackers):
            for d, det in enumerate(unit_detections):
                IOU_mat[t, d] = utils.box_utils.calculate_iou(trk, det.box)

        # Finding Matches using Hungarian Algorithm
        matched_idx = linear_assignment(-IOU_mat)

        unmatched_trackers, unmatched_detections = [], []
        for t, trk in enumerate(trackers):
            if t not in matched_idx[:, 0]:
                unmatched_trackers.append(t)

        for d, det in enumerate(unit_detections):
            if d not in matched_idx[:, 1]:
                unmatched_detections.append(d)

        matches = []

        # Checking quality of matched by comparing with threshold
        for m in matched_idx:
            if IOU_mat[m[0], m[1]] < iou_thrd:
                unmatched_trackers.append(m[0])
                unmatched_detections.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
