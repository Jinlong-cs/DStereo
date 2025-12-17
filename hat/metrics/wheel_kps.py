# Copyright (c) Horizon Robotics. All rights reserved.

import json
import logging
from typing import Union

import torch

from hat.evaluation.wheel_kps.utils import (
    check_bbox_dimension,
    compute_angle_error_xwheel,
    compute_norm_distance_error_wheel,
)
from hat.metrics.metric import EvalMetric
from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)

__all__ = ["NormError", "AngleError"]


@OBJECT_REGISTRY.register
class NormError(EvalMetric):
    """Keypoints norm error.

    Args:
        name: Name of this metric instance for display.
        dimension_filter_type: filter bbox based on dimension type.
        min_pixel: min pixel of dimension.
        max_pixel: max pixel of dimension.
        save_path: result path.
    """

    def __init__(
        self,
        name: str,
        dimension_filter_type: str = "height",
        min_pixel: int = 100,
        max_pixel: Union[int, None] = None,
        save_path: str = None,
    ):
        super(NormError, self).__init__(name)
        self.dimension_filter_type = dimension_filter_type
        self.max_pixel = max_pixel
        self.min_pixel = min_pixel
        self.name = name
        self.save_path = save_path

    def update(self, gt, kps_preds):
        bboxes = gt["bbox"].cpu()
        kps_gts = gt["wheel_kps"].cpu()
        offset = gt["fail_offset"].cpu()
        need_eval_two_occ = gt["eval_kps_two_occ"].cpu()
        crop_scale = gt["crop_scale"].cpu()
        crop_offset = gt["crop_offset"].cpu()
        img_paths = gt["img_path"]

        kps_preds = kps_preds / crop_scale + crop_offset
        batch_size = bboxes.shape[0]
        normerror_kps_raw = torch.zeros((batch_size, 2))
        for i in range(batch_size):
            bbox = bboxes[i, :]
            kps_gt = kps_gts[i, :]
            kps_pred = kps_preds[i, :]
            need_eval = check_bbox_dimension(
                bbox,
                self.max_pixel,
                self.min_pixel,
                self.dimension_filter_type,
            )
            if not need_eval:
                need_eval_two_occ[i, :] = 0
                continue
            if need_eval_two_occ[i, :].sum() < 0.1:
                continue

            if self.save_path is not None:
                pred_res = {
                    "img_path": img_paths[i],
                    "kps_pred": kps_pred.numpy().tolist(),
                    "kps_gt": kps_gt.numpy().tolist(),
                }
                with open(self.save_path, "a") as f:
                    f.writelines(json.dumps(pred_res) + "\n")
            error_0, error_1 = compute_norm_distance_error_wheel(
                bbox=bbox, gt_wheel_kps=kps_gt, pred_wheel_kps=kps_pred
            )
            normerror_kps_raw[i, 0] = error_0
            normerror_kps_raw[i, 1] = error_1

        normerror_kps_offset = normerror_kps_raw - offset
        valid_error = normerror_kps_offset * need_eval_two_occ

        self.num_inst += need_eval_two_occ.sum()
        self.sum_metric += valid_error.sum()


@OBJECT_REGISTRY.register
class AngleError(EvalMetric):
    """Keypoints angle error.

    Args:
        name: Name of this metric instance for display.
        dimension_filter_type: filter bbox based on dimension type.
        min_pixel: min pixel of dimension.
        max_pixel: max pixel of dimension.
        save_path: result path.
    """

    def __init__(
        self,
        name: str,
        dimension_filter_type: str = "height",
        min_pixel: int = 100,
        max_pixel: Union[int, None] = None,
        save_path: str = None,
    ):
        super(AngleError, self).__init__(name)
        self.dimension_filter_type = dimension_filter_type
        self.max_pixel = max_pixel
        self.min_pixel = min_pixel
        self.name = name
        self.save_path = save_path

    def update(self, gt, kps_preds):
        bboxes = gt["bbox"].cpu()
        kps_gts = gt["wheel_kps"].cpu()
        need_eval_two_occ = gt["eval_kps_two_occ"].cpu()
        crop_scale = gt["crop_scale"].cpu()
        crop_offset = gt["crop_offset"].cpu()
        img_paths = gt["img_path"]

        kps_preds = kps_preds / crop_scale + crop_offset
        batch_size = bboxes.shape[0]
        for i in range(batch_size):
            bbox = bboxes[i, :]
            kps_gt = kps_gts[i, :]
            kps_pred = kps_preds[i, :]
            need_eval = check_bbox_dimension(
                bbox,
                self.max_pixel,
                self.min_pixel,
                self.dimension_filter_type,
            )
            if not need_eval:
                need_eval_two_occ[i, :] = 0
                continue
            if need_eval_two_occ[i, :].sum() > 0:
                if self.save_path is not None:
                    pred_res = {
                        "img_path": img_paths[i],
                        "kps_pred": kps_pred.numpy().tolist(),
                        "kps_gt": kps_gt.numpy().tolist(),
                    }
                    with open(self.save_path, "a") as f:
                        f.writelines(json.dumps(pred_res) + "\n")
                angle = compute_angle_error_xwheel(
                    gt_wheel_kps=kps_gt, pred_wheel_kps=kps_pred
                )
                self.num_inst += 1
                self.sum_metric += angle
