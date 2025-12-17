# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from typing import Dict

import numpy as np

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import convert_numpy
from .voc_detection import VOCMApMetric

logger = logging.getLogger(__name__)

__all__ = ["VOCROIDetMApMetric"]


@OBJECT_REGISTRY.register
class VOCROIDetMApMetric(VOCMApMetric):
    """Calculate mean AP for ROI object detection task."""

    def update(self, model_outs: Dict):
        """model_outs is a dict, the meaning of it's key is as following.

        pred_rcnn_bboxes (List): Each element of pred_rcnn_bboxes is the
            predict result of an image. It's shape is (N, 6), where 6 means
            (x1, y1, x2, y2, label, score).
        pred_parent_rois (List): Each element of pred_parent_rois is the first
            stage predict result of an image.
        gt_bboxes (List): Each element of gt_bboxes is the bboxes' coordinates
            of an image. It's shape is (N, 4), where 4 means (x1, y1, x2, y2).
        parent_gt_bboxes (List): Each element of parent_gt_bboxes is the parant
            bboxes' coordinates of an image.
        gt_classes (List): Each element of gt_classes is the bboxes' classes
            of an image. It's shape is (N).
        gt_difficult(List): Each element of gt_difficult is the bboxes'
            difficult flag of an image. It's shape is (N).
        """

        parent_gt_bboxes = model_outs["parent_gt_bboxes"]
        gt_bboxes = model_outs["gt_bboxes"]
        gt_labels = model_outs["gt_classes"]
        gt_real_labels = model_outs.get("gt_labels")
        gt_difficults = model_outs["gt_difficult"]
        ig_bboxes = model_outs.get("ig_bboxes", None)
        parent_pred_bboxes = model_outs["pred_parent_rois"]
        outputs = model_outs["pred_rcnn_bboxes"]
        pred_bboxes = [pred[:, :4] for pred in outputs]
        pred_scores = [pred[:, 4] for pred in outputs]
        pred_labels = [pred[:, 5] for pred in outputs]

        if gt_difficults is None:
            gt_difficults = [None for _ in gt_labels]

        for batch_id, (
            parent_pred_bbox,
            pred_bbox,
            pred_label,
            pred_score,
            parent_gt_bbox,
            gt_bbox,
            gt_label,
            gt_difficult,
        ) in enumerate(
            zip(
                *[
                    convert_numpy(x)
                    for x in [
                        parent_pred_bboxes,
                        pred_bboxes,
                        pred_labels,
                        pred_scores,
                        parent_gt_bboxes,
                        gt_bboxes,
                        gt_labels,
                        gt_difficults,
                    ]
                ]
            )
        ):
            if gt_bbox.shape[0] == 0 or parent_gt_bbox.shape[0]:
                continue
            # strip padding -1 for pred and gt
            parent_ious = self._cal_iou_fn(parent_pred_bbox, parent_gt_bbox)
            # filter unmatched parent boxes
            parent_pred_index = parent_ious.argmax(axis=0)
            parent_pred_max_ious = parent_ious.max(axis=0)
            parent_pred_index = parent_pred_index[
                parent_pred_max_ious > self.iou_thresh
            ]
            pred_bbox = pred_bbox[parent_pred_index]
            pred_score = pred_score[parent_pred_index]
            pred_label = pred_label[parent_pred_index]
            parent_gt_index = parent_ious.argmax(axis=1)
            parent_gt_max_ious = parent_ious.max(axis=1)
            parent_gt_index = parent_gt_index[
                parent_gt_max_ious > self.iou_thresh
            ]
            gt_bbox = gt_bbox[parent_gt_index]
            gt_label = gt_label[parent_gt_index]

            valid_pred = np.where(pred_label.flat >= 0)[0]
            pred_bbox = pred_bbox[valid_pred, :]
            pred_label = pred_label.flat[valid_pred].astype(int)
            pred_score = pred_score.flat[valid_pred]

            if gt_difficult is None:
                gt_difficult = np.zeros(gt_bbox.shape[0])

            possible_choices = [pred_label, gt_label]
            if gt_real_labels is not None:
                possible_choices.append(
                    convert_numpy(gt_real_labels[batch_id])
                )
            for ln in np.unique(np.concatenate(possible_choices).astype(int)):
                self._update_for_class(
                    ln,
                    gt_label,
                    gt_bbox,
                    ig_bboxes[batch_id] if ig_bboxes else None,
                    gt_real_labels[batch_id] if gt_real_labels else None,
                    gt_difficult,
                    pred_label,
                    pred_bbox,
                    pred_score,
                )

    def _cal_iou_fn(self, bbox_a, bbox_b, itype="iou"):
        bbox_a = np.maximum(bbox_a[:, :4], 0)
        bbox_b = np.maximum(bbox_b[:, :4], 0)

        return super()._cal_iou_fn(bbox_a, bbox_b, itype)
