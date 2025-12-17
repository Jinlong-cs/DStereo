# Copyright (c) Horizon Robotics. All rights reserved.

from typing import List

import numpy as np
import torch

from hat.core.box_utils import bbox_overlaps
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import convert_numpy
from .metric import EvalMetric
from .utils import cat_tensor_to_numpy

__all__ = ["ROIKPSMetric", "ROIKPSMetricCoupling"]


@OBJECT_REGISTRY.register
class ROIKPSMetric(EvalMetric):
    """Calculate mean AP and precison/reacall for ROI keypoints detection task.

    Args:
        name: Metric name.
        kps_num: Number of keypoints.
        gt_pts_type: GT visual type, filter pts for metric.
        kps_iou_thresh: OKS threshold for TP.
        box_iou_thresh: IOU overlap threshold for parent bbox TP.
    """

    def __init__(
        self,
        name: str = "roi_kps",
        kps_num: int = 2,
        gt_pts_type: int = 0,
        kps_iou_thresh: float = 0.5,
        box_iou_thresh: float = 0.5,
    ):
        super().__init__(name, warn_without_compute=False)

        self.kps_num = kps_num
        self.gt_pts_type = gt_pts_type
        self.kps_iou_thresh = kps_iou_thresh
        self.box_iou_thresh = box_iou_thresh
        self.sigmas = (
            np.full(shape=(kps_num,), fill_value=1.0, dtype=np.float64)
            / 10.0
            * 0.5
        )
        self.reset()

    def _init_states(self):
        self.add_state(
            "_n_pos",
            default=torch.zeros(1),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "_match",
            default=[],
            dist_reduce_fx="cat",
        )

        self.add_state(
            "_score",
            default=[],
            dist_reduce_fx="cat",
        )

    def reset(self):
        """Clear the internal statistics to initial state."""
        self.num_inst = 0
        self.ap_metric = 0.0
        self.precison_metric = 0.0
        self.recall_metric = 0.0
        self.names = ["ap", "precision", "recall"]

    def compute(self):
        self.gather_metrics()
        if self.num_inst == 0:
            return (self.names, [float("nan")] * len(self.names))
        else:
            names = ["%s" % (i) for i in self.names]
            values = [
                x / self.num_inst if x is not None else None
                for x in (
                    self.ap_metric,
                    self.precison_metric,
                    self.recall_metric,
                )
            ]

            return (names, values)

    def update(self, targets: List[torch.Tensor], preds: List[torch.Tensor]):
        """Update statistic variables.

        Args:
            targets: Length of list is batch_size, and the tensors'shape is
                n x 10. Where n means the number of boxes per image.
            preds: The same structure with the targets.
        """

        device = self._n_pos.device
        for (pred, target) in zip(
            *[convert_numpy(x) for x in [preds, targets]]
        ):
            box_ious, kps_ious = self.computeOKS(pred, target)
            if len(pred) == 0 or len(target) == 0:
                continue

            gt_box_index = box_ious.argmax(axis=1)
            for i, gt_box_idx in enumerate(gt_box_index):
                if np.array(box_ious)[i, gt_box_idx] < self.box_iou_thresh:
                    continue
                if np.array(kps_ious)[i, gt_box_idx] >= self.kps_iou_thresh:
                    match_state = [1]
                else:
                    match_state = [0]
                self._n_pos += 1
                self._match = self._match + [
                    torch.tensor(match_state, device=device)
                ]
                self._score = self._score + [
                    torch.tensor(np.mean(pred[i, 6::3]), device=device)
                ]

    def gather_metrics(self):
        """Update num_inst and sum_metric."""
        rec, prec = self._recall_prec()

        ap = self._average_precision(rec, prec)
        self.ap_metric = ap
        self.num_inst = 1
        if rec is None:
            self.recall_metric = None
        else:
            self.recall_metric = rec[-1]
        if prec is None:
            self.precison_metric = None
        else:
            self.precison_metric = prec[-1]

    def _recall_prec(self):
        """Get recall and precision from internal records."""

        if len(self._match) == 0 or len(self._score) == 0:
            return None, None
        match = cat_tensor_to_numpy(self._match).astype(np.int32)
        score = cat_tensor_to_numpy(self._score)
        n_pos = cat_tensor_to_numpy(self._n_pos)
        assert len(score) == len(
            match
        ), f"length of score:{len(score)}, length of match:{len(match)}"

        order = score.argsort()[::-1]
        match = match[order]

        tp = np.cumsum(match == 1)
        fp = np.cumsum(match == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec is nan.
        with np.errstate(divide="ignore", invalid="ignore"):
            prec = tp / (fp + tp)
        # If n_pos is 0, rec is None.
        if n_pos > 0:
            rec = tp / n_pos
        else:
            rec = None
        return rec, prec

    def _average_precision(self, rec: np.ndarray, prec: np.ndarray):
        """Calculate average precision.

        Args:
            rec: Accumulated recall.
            prec: Accumulated precision.
        Returns:
            Ap float number.
        """

        if rec is None or prec is None:
            return np.nan

        # append sentinel values at both ends
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], np.nan_to_num(prec), [0.0]))

        # compute precision integration ladder
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # look for recall value changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # sum (\delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def bboxMatchBeforeOks(self, dts, box_ious, box_iou_thresh):
        if len(box_ious) == 0:
            return dts
        idx = box_ious.max(axis=1) > box_iou_thresh
        dts = dts[idx]

        return dts

    def computeOKS(self, dts, gts):
        if len(gts) == 0 or len(dts) == 0:
            return [], []

        box_ious = bbox_overlaps(dts[:, :4], gts[:, :4], "iou")
        kps_ious = np.zeros((len(dts), len(gts)))
        _vars = (self.sigmas * 2) ** 2

        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            xg = gt[4::3]
            yg = gt[5::3]
            vg = gt[6::3]
            valid_pts_index = vg > self.gt_pts_type
            num_valid_pts = np.count_nonzero(valid_pts_index)
            x1, y1, x2, y2 = gt[:4]
            w, h = x2 - x1, y2 - y1
            _x1 = x1 - w
            _x2 = x2 + w
            _y1 = y1 - w
            _y2 = y2 + w
            # 0: normal, 1: hard, 2: ignore
            for i, dt in enumerate(dts):
                xd = dt[4::3]
                yd = dt[5::3]
                if num_valid_pts > 0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in
                    # (x0,y0) & (x1,y1)
                    z = np.zeros((self.kps_num,))
                    dx = np.max((z, _x1 - xd), axis=0) + np.max(
                        (z, xd - _x2), axis=0
                    )  # noqa
                    dy = np.max((z, _y1 - yd), axis=0) + np.max(
                        (z, yd - _y2), axis=0
                    )  # noqa
                error = (
                    (dx ** 2 + dy ** 2) / _vars / (w * h + np.spacing(1)) / 2
                )  # noqa
                kps_sim = np.exp(-error)

                if num_valid_pts > 0:
                    kps_sim = kps_sim[valid_pts_index]

                kps_ious[i, j] = np.mean(kps_sim)

        return box_ious, kps_ious


@OBJECT_REGISTRY.register
class ROIKPSMetricCoupling(ROIKPSMetric):
    """Calculate mean AP and P/R for ROI keypoints detection task.

    This metric apply metric coupling with roi detection results.

    Args:
        name: Metric name.
        kps_num: Number of keypoints.
        gt_pts_type: GT visual type, filter pts for metric.
        kps_iou_thresh: OKS threshold for TP.
        box_iou_thresh: IOU overlap threshold for parent bbox TP.
        remove_bbox_fp: Whether to remove bbox FPs.
    """

    def __init__(
        self,
        name="ROIKPSMetricCoupling",
        kps_num: int = 2,
        gt_pts_type: int = 0,
        kps_iou_thresh: float = 0.5,
        box_iou_thresh: float = 0.5,
        remove_bbox_fp: bool = True,
    ):
        super().__init__(
            name,
            kps_num,
            gt_pts_type,
            kps_iou_thresh,
            box_iou_thresh,
        )
        self.remove_bbox_fp = remove_bbox_fp

    def reset(self):
        """Clear the internal statistics to initial state."""
        super().reset()
        self.names = ["CouplingAP", "CouplingPrecision", "CouplingRecall"]

    def update(self, targets, preds):
        device = self._n_pos.device
        for (pred, target) in zip(
            *[convert_numpy(x) for x in [preds, targets]]
        ):
            # strip padding -1 for pred and gt
            if self.remove_bbox_fp:
                box_ious = bbox_overlaps(pred[:, :4], target[:, :4], "iou")
                pred = self.bboxMatchBeforeOks(
                    pred, box_ious, self.box_iou_thresh
                )
            box_ious, kps_ious = self.computeOKS(pred, target)
            self._n_pos += len(target)
            if len(pred) == 0:
                continue
            if len(target) == 0:
                self._score.extend([np.mean(dt[6::3]) for dt in pred])
                self._match.extend([0] * len(pred))
                continue

            gt_box_index = box_ious.argmax(axis=1)
            selec = [False] * len(target)
            for i, gt_box_idx in enumerate(gt_box_index):
                if selec[gt_box_idx]:
                    continue
                if np.array(kps_ious)[i, gt_box_idx] >= self.kps_iou_thresh:
                    match_state = [1]
                    selec[gt_box_idx] = True
                else:
                    match_state = [0]
                self._match = self._match + [
                    torch.tensor(match_state, device=device)
                ]
                self._score = self._score + [
                    torch.tensor(np.mean(pred[i, 6::3]), device=device)
                ]
