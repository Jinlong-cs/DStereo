# Copyright (c) Horizon Robotics. All rights reserved.

from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from hat.models.losses.smooth_l1_loss import SmoothL1Loss
from hat.registry import OBJECT_REGISTRY

__all__ = ["Lmks2Loss"]


@OBJECT_REGISTRY.register
class Lmks2Loss(nn.Module):
    """KPS Loss."""

    def __init__(
        self,
    ):
        super().__init__()
        self.cls_loss = SmoothL1Loss()
        self.reg_loss = SmoothL1Loss()

    @autocast(enabled=False)
    def forward(
        self, preds: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        cls_pred = preds["kps_label_pred"]
        reg_pred = preds["kps_pos_offset_pred"]

        cls_label = labels["kps_cls_label"]
        cls_weight = labels["kps_cls_label_weight"]

        reg_label = labels["kps_pos_offset"]
        reg_weight = labels["kps_pos_offset_weight"]
        kps_class_loss = self.cls_loss(
            cls_pred,
            cls_label,
            cls_weight,
            avg_factor=(cls_weight.sum(dim=[1, 2, 3]) > 0).sum() + 1e-6,
        )
        kps_pos_offset_loss = self.reg_loss(
            reg_pred,
            reg_label,
            reg_weight,
            avg_factor=(reg_weight.sum(dim=[1, 2, 3]) > 0).sum() + 1e-6,
        )

        output_loss = OrderedDict(
            kps_class_loss=kps_class_loss,
            kps_reg_loss=kps_pos_offset_loss,
        )

        return {"wheel_kps_loss": output_loss}
