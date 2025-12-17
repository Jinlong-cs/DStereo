# Copyright (c) Horizon Robotics. All rights reserved.
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from hat.registry import OBJECT_REGISTRY

__all__ = ["LdmkLoss", "HumanPoseLoss"]


@OBJECT_REGISTRY.register
class LdmkLoss(nn.Module):
    def __init__(self, loss_type: str):
        super().__init__()
        self.loss_type = loss_type.lower()

    def forward(
        self,
        label: torch.Tensor,
        pred: torch.Tensor,
        weight: torch.Tensor,
        **kwargs
    ):
        if self.loss_type == "l1":
            loss = torch.abs(label - pred) * weight
        elif self.loss_type == "l2":
            loss = torch.square(label - pred) * weight
        elif self.loss_type == "smooth_l1":
            raise NotImplementedError()
        elif self.loss_type == "wing_loss":
            raise NotImplementedError()
        elif self.loss_type == "awing":
            raise NotImplementedError()
        elif self.loss_type == "gnll":
            assert "pred_var" in kwargs
            raise NotImplementedError()
        else:
            raise ValueError("Not supported loss type.")
        return loss.mean(0).sum()


@OBJECT_REGISTRY.register
class HumanPoseLoss(nn.Module):
    """Human landmark loss module.

    Loss structure that collect img, label and loss function.

    Args:
        ldmk_num: Number of landmark.
        feat_height: The height of the output feature.
        feat_width: The width of the output feature.
        cls_loss: Heatmap loss module.
        reg_loss: Offset loss module.
    """

    def __init__(
        self,
        ldmk_num: int,
        feat_height: int,
        feat_width: int,
        cls_loss: nn.Module,
        reg_loss: nn.Module,
    ):
        super().__init__()
        self.ldmk_num = ldmk_num
        self.cls_loss = cls_loss
        self.reg_loss = reg_loss
        self.feat_height = feat_height
        self.feat_width = feat_width

    @autocast(enabled=False)
    def forward(self, preds, labels):
        # convert to float32 while using amp
        for k, v in preds.items():
            preds[k] = v.float()

        ldmk_pred = preds["ldmk_pred"]
        cls_pred = ldmk_pred[:, 0 : self.ldmk_num, :, :]
        reg_pred = ldmk_pred[:, self.ldmk_num : self.ldmk_num * 3, :, :]

        cls_label = labels["ldmk_cls_label"].view(
            -1, self.ldmk_num, self.feat_height, self.feat_width
        )
        cls_label_weight = labels["ldmk_cls_label_weight"].view(
            -1, self.ldmk_num, self.feat_height, self.feat_width
        )

        reg_label = labels["ldmk_reg_label"].view(
            -1, self.ldmk_num * 2, self.feat_height, self.feat_width
        )
        reg_label_weight = labels["ldmk_reg_label_weight"].view(
            -1, self.ldmk_num * 2, self.feat_height, self.feat_width
        )

        ldmk_label_loss = self.cls_loss(
            pred=cls_pred,
            target=cls_label,
            weight=cls_label_weight,
        )

        ldmk_pos_offset_loss = self.reg_loss(
            pred=reg_pred,
            target=reg_label,
            weight=reg_label_weight,
        )

        output_loss = OrderedDict(
            ldmk_cls_loss=ldmk_label_loss,
            ldmk_reg_loss=ldmk_pos_offset_loss,
        )

        return output_loss
