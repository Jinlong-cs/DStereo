# Copyright (c) Horizon Robotics. All rights reserved.
import copy
import logging

import torch

from hat.registry import OBJECT_REGISTRY
from .metric import EvalMetric

__all__ = ["EllipseParamError"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class EllipseParamError(EvalMetric):
    """Computes the error of the ellipse parameters.

    Make sure that the gt and predicted of the ellipse parameters
    are arranged as [tx, ty, a, b, θ].
    The value of θ ranges from 0 to 180 degrees。

    Args:
        img_shape: The size of the image passed into the model(H, W, C).
            Default to (64, 64, 3).
        do_norm: Whether to normalize the error(center, scale).
            Default to True.
        name: Name of this metric instance for display.
    """

    def __init__(self, do_norm: bool = True, name: str = "EllipseParamError"):
        self.name = name
        self.do_norm = do_norm
        self.name = [
            self.name + "_center_error",
            self.name + "_scale_error",
            self.name + "_angle_error",
        ]
        super(EllipseParamError, self).__init__(self.name)

    def _init_states(self):
        self.add_state(
            "sum_metric",
            default=torch.zeros(3),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "num_inst", default=torch.zeros(1), dist_reduce_fx="sum"
        )

    def _cal_euclidean_dist(
        self,
        gt_point: torch.Tensor,
        pred_point: torch.Tensor,
    ):
        dist = torch.sqrt(
            (gt_point[:, 0] - pred_point[:, 0]) ** 2
            + (gt_point[:, 1] - pred_point[:, 1]) ** 2
        )
        if self.do_norm:
            dist = dist / self.dist_norm
        return dist

    def update(self, gt_params: torch.Tensor, pred_params: torch.Tensor):
        if gt_params.ndim == 1:
            gt_params = gt_params.unsqueeze(0)
        if pred_params.ndim == 1:
            pred_params = pred_params.unsqueeze(0)
        bs = gt_params.shape[0]
        if self.do_norm:
            self.dist_norm = torch.sqrt(
                gt_params[:, 2] ** 2 + gt_params[:, 3] ** 2
            ).clamp(1e-5)

        # ellipse center error
        gt_center = copy.deepcopy(gt_params[:, 0:2])
        pred_center = copy.deepcopy(pred_params[:, 0:2])
        center_dist = self._cal_euclidean_dist(gt_center, pred_center)

        # ellipse scale error
        gt_scale = copy.deepcopy(gt_params[:, 2:4])
        pred_scale = copy.deepcopy(pred_params[:, 2:4])
        scale_dist = self._cal_euclidean_dist(gt_scale, pred_scale)

        # ellipse angle error
        angle_error = torch.abs(gt_params[:, 4] - pred_params[:, 4])

        self.num_inst += bs
        self.sum_metric[0] += torch.sum(center_dist)
        self.sum_metric[1] += torch.sum(scale_dist)
        self.sum_metric[2] += torch.sum(angle_error)

    def compute(self):
        values = [
            (sum / self.num_inst[0]).item()
            if self.num_inst[0] != 0
            else float("nan")
            for sum in self.sum_metric
        ]
        return values
