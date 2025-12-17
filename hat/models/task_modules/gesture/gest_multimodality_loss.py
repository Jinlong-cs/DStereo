# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List

import torch.nn as nn

from hat.models.losses.utils import weight_reduce_loss
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "GestMultiModeLoss",
]


@OBJECT_REGISTRY.register
class GestMultiModeLoss(nn.Module):
    r"""Calculates multimodality loss for gesture task.

    Args:
        fusion_type: Type of multimodality feature fusion.
            Defaults to "result_fusion".
        loss_kps: Losses of kps branch module.
            Defaults to None.
        loss_frames: Losses of frames branch module.
            Defaults to None.
        loss_kps_frames: Losses of feature fusion module.
            Defaults to None.
        loss_weight:
    """

    def __init__(
        self,
        fusion_type: str = "result_fusion",
        loss_kps: nn.Module = None,
        loss_frames: nn.Module = None,
        loss_kps_frames: nn.Module = None,
        loss_weight: List = None,
    ):
        super(GestMultiModeLoss, self).__init__()
        self.fusion_type = fusion_type
        self.loss_weight = loss_weight
        if self.fusion_type == "result_fusion":
            assert (
                loss_kps and loss_frames and len(loss_weight) == 2
            ), f"{fusion_type} need loss_kps:{loss_kps} and loss_frames:{loss_frames} and \
                    loss_weight:{loss_weight}"
            self.loss_kps = loss_kps
            self.loss_frames = loss_frames
        elif self.fusion_type == "feature_fusion":
            assert (
                loss_kps_frames and len(loss_weight) == 1
            ), f"{fusion_type} need loss_kps_frames:{loss_kps_frames}"
            self.loss_kps_frames = loss_kps_frames
        elif self.fusion_type == "cross_modal_fusion":
            raise NotImplementedError(
                f"Todo: now just support feature/result fusion, \
                not support {fusion_type}"
            )

    def forward(
        self,
        logit_output,
        target,
        weight,
    ):

        if self.fusion_type == "result_fusion":
            # kps branch
            _loss_kps = self.loss_kps(logit_output[0], target)
            # frames branch
            _loss_frames = self.loss_frames(logit_output[1], target)
            # weight
            loss = weight_reduce_loss(
                loss=_loss_kps + _loss_frames, weight=weight, reduction="mean"
            )
        elif self.fusion_type == "feature_fusion":
            # feature kps and frames
            loss_kps_frames = self.loss_kps_frames(logit_output[2], target)
            loss = weight_reduce_loss(
                loss=loss_kps_frames, weight=weight, reduction="mean"
            )
        return loss
