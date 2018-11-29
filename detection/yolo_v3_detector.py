import logging
import os
from typing import List

import torch
from torch.autograd import Variable

from pipeline import UnitObject
from .base_detector import BaseDetector
from .yolov3.yolov3.darknet import Darknet
from .yolov3.yolov3.preprocess import prep_frame
from .yolov3.yolov3.util import load_classes, write_results


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

BASE_PATH = os.path.join(os.path.dirname(__file__), 'yolov3')


class Detector(BaseDetector):
    cfg = os.path.join(BASE_PATH, 'cfg', 'yolov3.cfg')
    weights = os.path.join(BASE_PATH, 'weights', 'yolov3.weights')

    def __init__(self):
        super().__init__()
        self.class_names = load_classes(os.path.join(BASE_PATH, 'data', 'coco.names'))
        self.num_classes = 80
        self.batch_size = 1
        self.confidence = 0.5
        self.nms_thresh = 0.4
        self.scales = "1,2,3"
        self.inp_dim = 416  # k*32 where k is int and >1
        self.cuda = torch.cuda.is_available()
        self._load_model()

    def _load_model(self):
        model = Darknet(Detector.cfg)
        model.load_weights(Detector.weights)
        model.net_info["height"] = str(self.inp_dim)
        if self.cuda:
            model.cuda()
        model.eval()
        self.model = model

    def get_localization(self, image, debug=False) -> List[UnitObject]:
        ret = []
        im_batch = prep_frame(image, self.inp_dim)
        im_dim_list = [image.shape[1], image.shape[0]]
        im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

        if self.cuda:
            im_dim_list = im_dim_list.cuda()
            batch = im_batch.cuda()

        with torch.no_grad():
            prediction = self.model(Variable(batch), self.cuda)

        output = write_results(prediction, self.confidence, self.num_classes, nms=True, nms_conf=self.nms_thresh)

        if type(output) == int:
            return ret

        im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

        scaling_factor = torch.min(self.inp_dim / im_dim_list, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (self.inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (self.inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2
        output[:, 1:5] /= scaling_factor

        output = output.int().cpu().numpy()
        torch.cuda.empty_cache()

        for x in output:
            ret.append(UnitObject([x[2], x[1], x[4], x[3]], x[-1]))

        return ret