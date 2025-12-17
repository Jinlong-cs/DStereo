# Copyright (c) Horizon Robotics. All rights reserved.

import numpy as np

from hat.core.crowd import multi_bboxes_to_polygon


class GT:
    count = 0

    def __init__(
        self, bbox, default_type, default_overlap, gt_cls_type_raw=None
    ):
        self.id = GT.count
        GT.count += 1
        self.bbox = bbox
        self.bbox_type = bbox.__class__.__name__
        self.info = {"normal": [], "remove": [], "hard": [], "ignore": []}
        self.gt_type = default_type
        self.overlap = default_overlap
        self.eval_type = "FN"
        self.gt_cls_type_raw = gt_cls_type_raw
        self.crowd_group = None

    def set_type(self):
        remove = False
        if len(self.info["ignore"]) > 0:
            self.gt_type = "ignore"
            self.overlap = min([x[1] for x in self.info["ignore"]])
        elif len(self.info["remove"]) > 0:
            remove = True
        elif len(self.info["hard"]) > 0:
            self.gt_type = "hard"
            self.overlap = min([x[1] for x in self.info["hard"]])
        elif len(self.info["normal"]) > 0:
            self.gt_type = "normal"
            self.overlap = min([x[1] for x in self.info["normal"]])
        return remove


class DET:
    count = 0

    def __init__(self, bbox, score, det_cls_type_raw=None):
        self.id = DET.count
        DET.count += 1
        self.bbox = bbox
        self.bbox_type = bbox.__class__.__name__
        self.score = score
        self.matched_gt_id = -1
        self.eval_type = "FP"
        self.error = {}
        self.det_cls_type_raw = det_cls_type_raw


def calar(fppi, rec):
    trans_fppi = np.log10(fppi) / 3.0 + 1
    mfppi = trans_fppi[(trans_fppi >= 0) & (trans_fppi <= 1)]
    mrec = rec[(trans_fppi >= 0) & (trans_fppi <= 1)]
    ar = 0
    for i in range(len(mrec) - 1):
        ar += (mfppi[i + 1] - mfppi[i]) * max(mrec[i], mrec[i + 1])
    return ar


def calap(recall, prec):
    mrec = [0] + list(recall.flatten()) + [1]
    mpre = [0] + list(prec.flatten()) + [0]
    for i in range(len(mpre) - 2, 0, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0
    for i in range(len(mpre) - 1):
        if mpre[i + 1] > 0:
            ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
    return ap, mrec[1:-1], mpre[1:-1]


def get_optimal_metrics(scores_lut):
    if len(scores_lut) <= 0:
        optimal_thres, max_det_rate, recall, precision = [np.inf] * 4
    else:
        optimal_thres = max(
            scores_lut,
            key=lambda x: (scores_lut[x]["num_tp"] - scores_lut[x]["num_fp"])
            / (scores_lut[x]["num_gt"] + 1e-10),
        )
        max_det_rate = (
            scores_lut[optimal_thres]["num_tp"]
            - scores_lut[optimal_thres]["num_fp"]
        ) / (scores_lut[optimal_thres]["num_gt"] + 1e-10)
        recall = scores_lut[optimal_thres]["recall"]
        precision = scores_lut[optimal_thres]["precision"]
    return optimal_thres, max_det_rate, recall, precision


class CrowdGroup(object):
    def __init__(self, objs=None):
        if objs is None:
            objs = []
        self._polygon = multi_bboxes_to_polygon(
            list(
                map(
                    lambda obj: [
                        obj.bbox.x1,
                        obj.bbox.y1,
                        obj.bbox.x2,
                        obj.bbox.y2,
                    ],
                    objs,
                )
            )
        )

    @property
    def area(self):
        return self._polygon.area

    def intersection(self, obj):
        crowd_group = CrowdGroup()
        if isinstance(obj, GT) or isinstance(obj, DET):
            crowd_group._polygon = self._polygon.intersection(
                CrowdGroup([obj])._polygon
            )
        elif isinstance(obj, CrowdGTGroup):
            crowd_group._polygon = self._polygon.intersection(obj._polygon)
        return crowd_group


class CrowdGTGroup(CrowdGroup):
    def __init__(self, gts):
        super(CrowdGTGroup, self).__init__(gts)
        self._gts = gts
        self.matched_dets = []
        for gt in self._gts:
            gt.crowd_group = self
