# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Optional, Tuple

import numpy as np
import torch

from hat.core.box_utils import bbox_overlaps
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import convert_numpy
from .metric import EvalMetric
from .utils import cat_tensor_to_numpy

__all__ = ["ROICLSPrecRec"]


@OBJECT_REGISTRY.register
class ROICLSPrecRec(EvalMetric):
    """Computes precision/recall in ROI classification task.

    Args:
        num_classes: Number of classes, excluding background class.
        axis: The axis that represents classes.
        score_threshs: Score thresholds for caculate
            recall/precison, default is (0.3).
        class_dict: If provided, will print out recall/precison for
            each class, default to None, means only caculate
            recall/precision for all data.
        box_iou_thresh: The threshold of the ious between the pred_boxes
            and the gt_boxes.
        remove_bbox_fp: Whether remove the fp box before computing the
            precison and recall.
    """

    def __init__(
        self,
        num_classes: int,
        axis: int = 1,
        score_threshs: Tuple[float] = (0.0,),
        class_dict: Optional[dict] = None,
        box_iou_thresh: float = 0.5,
        remove_bbox_fp: bool = True,
    ):
        self.num_classes = num_classes
        super().__init__("rcnn_cls_accuracy", warn_without_compute=False)
        if class_dict is None:
            self.class_dict = {0: "all"}
        else:
            self.class_dict = class_dict
            self.class_dict.update({num_classes: "all"})
        self.axis = axis
        self.score_threshs = score_threshs
        self.box_iou_thresh = box_iou_thresh
        self.remove_bbox_fp = remove_bbox_fp
        self.reset()

    def _init_states(self):
        self.add_state(
            "_n_pos",
            default=torch.zeros(self.num_classes),
            dist_reduce_fx="sum",
        )
        for cls in range(self.num_classes):
            self.add_state(
                "_%d_match" % (cls),
                default=[],
                dist_reduce_fx="cat",
            )

            self.add_state(
                "_%d_score" % (cls),
                default=[],
                dist_reduce_fx="cat",
            )

    def reset(self):
        """Clear the internal statistics to initial state."""
        self.num_inst = []
        self.sum_metric = []
        self.name = []
        super().reset()

    def compute(self):
        self.gather_metrics()
        names = ["%s" % (_name) for _name in self.name]
        values = [
            x / y
            if ((y is not None) and (y != 0) and (x is not None))
            else float("nan")
            for x, y in zip(self.sum_metric, self.num_inst)
        ]
        return (names, values)

    def update(self, targets, preds):

        device = self._n_pos.device
        for (pred, target) in zip(
            *[convert_numpy(x) for x in [preds, targets]]
        ):

            if len(target) == 0 or len(pred) == 0:
                continue
            gt_labels = target[:, 4] - 1  # class 0 is background
            pred_scores = pred[:, 4:].max(axis=1)
            pred_labels = pred[:, 4:].argmax(axis=1)
            box_ious = bbox_overlaps(pred[:, :4], target[:, :4], "iou")
            gt_box_indexs = box_ious.argmax(axis=1)

            for i, gt_box_idx in enumerate(gt_box_indexs):
                if np.array(box_ious)[i, gt_box_idx] < self.box_iou_thresh:
                    continue
                pci = int(pred_labels[i])
                gci = int(gt_labels[gt_box_idx])
                self._n_pos[gci] += 1
                match_state = []
                if pci == gci:
                    match_state.append(1)
                else:
                    match_state.append(0)
                match_attr_name = "_%d_match" % (pci)
                pre_match = getattr(self, match_attr_name)
                cur_match = pre_match + [
                    torch.tensor(match_state, device=device)
                ]
                setattr(self, match_attr_name, cur_match)

                score_attr_name = "_%d_score" % (pci)
                pre_score = getattr(self, score_attr_name)
                cur_score = pre_score + [
                    torch.tensor([pred_scores[i]], device=device)
                ]
                setattr(self, score_attr_name, cur_score)

    def gather_metrics(self):
        """Update num_inst and sum_metric."""
        for score in self.score_threshs:
            for class_id, class_name in self.class_dict.items():
                rec, prec = self._recall_prec(class_id, score)
                # recall
                self.name.append(
                    class_name + "_scorethresh" + str(score) + "_rec"
                )
                self.num_inst.append(1)
                self.sum_metric.append(rec)
                # precison
                self.name.append(
                    class_name + "_scorethresh" + str(score) + "_prec"
                )
                self.num_inst.append(1)
                self.sum_metric.append(prec)

    def _recall_prec(self, cls, score_thresh):
        """Get recall and precision from internal records."""
        if self.class_dict[cls] == "all":
            _score, _match = [], []
            for ci in range(self.num_classes):
                _score.extend(
                    cat_tensor_to_numpy(getattr(self, "_%d_score" % (ci)))
                )
                _match.extend(
                    cat_tensor_to_numpy(getattr(self, "_%d_match" % (ci)))
                )
                _n_pos = sum(self._n_pos.cpu().numpy())
        else:
            _score = cat_tensor_to_numpy(getattr(self, "_%d_score" % (cls)))
            _match = cat_tensor_to_numpy(getattr(self, "_%d_match" % (cls)))
            _n_pos = self._n_pos[cls].cpu().numpy()
        score_l = np.array(_score)
        match_l = np.array(_match, dtype=np.int32)

        valid = score_l >= score_thresh
        match_l = match_l[valid]

        tp = np.sum(match_l == 1)
        fp = np.sum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        with np.errstate(divide="ignore", invalid="ignore"):
            prec = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if _n_pos > 0:
            rec = tp / _n_pos
        else:
            rec = None

        return rec, prec

    def bboxMatchBeforeOks(self, dts, box_ious, box_iou_thresh):
        if len(box_ious) == 0:
            return dts
        idx = box_ious.max(axis=1) > box_iou_thresh
        dts = dts[idx]

        return dts
