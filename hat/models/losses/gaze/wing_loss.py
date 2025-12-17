# Copyright (c) Horizon Robotics. All rights reserved.
import math
from typing import List

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY

__all__ = ["WingLoss"]


def _apply_weighting(
    loss, weight=None, sample_weight=None, adaptive_weight=False
):
    """Apply weighting to loss.

    Parameters
    ----------
    loss : The loss to be weighted.
    weight : Global scalar weight for loss.
    sample_weight : Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has shape (64, 10)
        and you want to weight each sample in the batch separately,
        `sample_weight` should have shape (64, 1).

    Returns
    -------
    loss : Weighted loss
    """
    if adaptive_weight:
        sum_loss = torch.sum(loss, dim=(-1, -2))
        num_loss = torch.sum(loss > 0, dim=(-1, -2))
        num_ones = torch.ones_like(num_loss)
        num_loss = torch.where(num_loss > 0, num_loss, num_ones)
        loss = torch.div(sum_loss, num_loss)
    else:
        if sample_weight is not None:
            loss = torch.mul(loss, sample_weight)
        if weight is not None:
            loss = loss * weight
    return loss


@OBJECT_REGISTRY.register
class WingLoss(nn.Module):
    """Implementation for Wing Loss.

    WingLoss from "Wing Loss for Robust Facial Landmark Localisation,
    with Convolutional Neural Networks", <https://arxiv.org/abs/1711.06753>`.
    Different from native winloss, an adaptive_weight interface is provided
    here, which can adjust the output of the final loss according to the loss
    itself.

    Args:
        w: The w limits the range of the non-linear part.
        epsilon: The epsilon controls the curvature.
        is_averaged_output: Whether output loss is averaged.
        adaptive_weight: Whether adaptive loss according to the value of loss.
    """

    def __init__(
        self,
        w=10.0,
        epsilon=2.0,
        is_averaged_output=False,
        adaptive_weight=False,
        **kwargs
    ):
        super(WingLoss, self).__init__()
        self.w = w
        self.epsilon = epsilon
        self.is_averaged_output = is_averaged_output
        self.adaptive_weight = adaptive_weight
        self.c = w * (1.0 - math.log(1.0 + w / epsilon))

        self.weight = kwargs.get("weight", None)
        self.batch_axis = kwargs.get("batch_axis", 0)

    def forward(self, pred, label, sample_weight=None):
        abs_x = torch.abs(pred - label)
        loss = torch.where(
            abs_x < self.w,
            self.w * torch.log(1.0 + abs_x / self.epsilon),
            abs_x - self.c,
        )
        loss = _apply_weighting(
            loss,
            weight=self.weight,
            sample_weight=sample_weight,
            adaptive_weight=self.adaptive_weight,
        )
        if self.is_averaged_output:
            return torch.mean(loss, dim=self.batch_axis)
        else:
            return torch.sum(loss, axis=self.batch_axis)


class GazeWingLoss(nn.Module):
    """Implementation for Gaze Wing Loss.

    Wing loss specifically for gaze,
    where w and e for pitch and yaw is different.

    Args:
        pitch_w: range of nonlinear part for pitch angle.
        yaw_w: range of nonlinear part for yaw angle.
        pitch_e: defines the curvature of the nonlinear region for pitch angle.
        yaw_e: defines the curvature of the nonlinear region for yaw angle.
    """

    def __init__(
        self,
        pitch_w: float,
        yaw_w: float,
        pitch_e: float = 0.5,
        yaw_e: float = 0.5,
        weight: List = None,
        is_averaged_output: bool = False,
        **kwargs
    ):
        super(GazeWingLoss, self).__init__()
        batch_axis = kwargs.get("batch_axis", 0)
        self.pitch_wing_loss = WingLoss(
            w=pitch_w,
            epsilon=pitch_e,
            is_averaged_output=is_averaged_output,
            weight=weight,
            batch_axis=batch_axis,
        )
        self.yaw_wing_loss = WingLoss(
            w=yaw_w,
            epsilon=yaw_e,
            is_averaged_output=is_averaged_output,
            weight=weight,
            batch_axis=batch_axis,
        )

    def forward(self, pred, target):
        pred_gaze = pred.squeeze()
        pred_lpitch = pred_gaze[:, 0]
        pred_lyaw = pred_gaze[:, 1]
        pred_rpitch = pred_gaze[:, 2]
        pred_ryaw = pred_gaze[:, 3]

        gt_gaze = target.squeeze()
        gt_lpitch = gt_gaze[:, 0]
        gt_lyaw = gt_gaze[:, 1]
        gt_rpitch = gt_gaze[:, 2]
        gt_ryaw = gt_gaze[:, 3]

        lpitch_loss = self.pitch_wing_loss(pred_lpitch, gt_lpitch)
        lyaw_loss = self.yaw_wing_loss(pred_lyaw, gt_lyaw)
        rpitch_loss = self.pitch_wing_loss(pred_rpitch, gt_rpitch)
        ryaw_loss = self.yaw_wing_loss(pred_ryaw, gt_ryaw)
        return (lpitch_loss + lyaw_loss + rpitch_loss + ryaw_loss) / 4.0
