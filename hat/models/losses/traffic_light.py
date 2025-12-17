# Copyright (c) Horizon Robotics. All rights reserved.
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from hat.core.detection_utils import rearrange_det_dense_head_out
from hat.registry import OBJECT_REGISTRY
from .cross_entropy_loss import CEWithLabelSmooth

__all__ = ["TrafficTimeCrossEntropyLoss", "TLAttrRPNSepLoss"]


def rearrange_det_attr_head_out(attr_pred: List[torch.Tensor]) -> torch.Tensor:
    """Rearrange output of detection dense head.

    Generally, output of detection dense head should be in (N, C, H, W)
    format for each stride. This function rearranges both of them (regression
    and classification output), to concatenate predictions along each stride.
    """
    attr_shape = attr_pred[0].shape
    hwc_lst = list(
        map(
            lambda pred: pred.reshape(pred.shape[0], -1).shape[-1]
            // (attr_shape[1] // 3),  # noqa
            attr_pred,
        )
    )
    reorg_pred = [
        _cls.permute(0, 2, 3, 1).reshape(attr_shape[0], _hwc, -1)
        for _cls, _hwc in zip(attr_pred, hwc_lst)
    ]

    return torch.hstack(reorg_pred)


@OBJECT_REGISTRY.register
class TrafficTimeCrossEntropyLoss(nn.Module):
    """
    The losses of cross-entropy with label smooth.

    Args:
        smooth_alpha (float): Alpha of label smooth.
        ignore_index: The label index that will be ignored in evaluation.
        loss_weights: Loss weight for each type.
        digit_length: End channel dim of digits.
        ones_length: End channel dim of single-digit.
        tens_length: End channel dim of ten-digit.
        hundreds_length: End channel dim of hundred-digit.
    """

    def __init__(
        self,
        loss_type="CEWithLabelSmooth",
        smooth_alpha=0.1,
        ignore_index: int = -100,
        loss_weights: Optional[Tuple] = (1, 1, 1, 1),
        digit_length: int = 3,
        ones_length: int = 13,
        tens_length: int = 23,
        hundreds_length: int = 33,
    ):
        super(TrafficTimeCrossEntropyLoss, self).__init__()
        if loss_type == "CEWithLabelSmooth":
            self.loss = CEWithLabelSmooth(
                smooth_alpha=smooth_alpha,
                ignore_index=ignore_index,
            )
        else:
            raise NotImplementedError
        self.loss_weights = loss_weights
        self.digit_length = digit_length
        self.ones_length = ones_length
        self.tens_length = tens_length
        self.hundreds_length = hundreds_length

    def forward(self, input, target):
        pred_nums_digit = input[:, : self.digit_length].flatten(1)
        pred_ones_place = input[
            :, self.digit_length : self.ones_length
        ].flatten(1)
        pred_tens_place = input[
            :, self.ones_length : self.tens_length
        ].flatten(1)
        pred_hundreds_place = input[
            :, self.tens_length : self.hundreds_length
        ].flatten(1)
        gt_nums_digit = target[:, 0].view(-1)
        gt_ones_place = target[:, 1].view(-1)
        gt_tens_place = target[:, 2].view(-1)
        gt_hundreds_place = target[:, 3].view(-1)
        nums_digit_loss = self.loss(pred_nums_digit, gt_nums_digit)
        ones_place_loss = self.loss(pred_ones_place, gt_ones_place)
        tens_place_loss = self.loss(pred_tens_place, gt_tens_place)
        hundreds_place_loss = self.loss(pred_hundreds_place, gt_hundreds_place)

        return [
            self.loss_weights[0] * nums_digit_loss,
            self.loss_weights[1] * ones_place_loss,
            self.loss_weights[2] * tens_place_loss,
            self.loss_weights[3] * hundreds_place_loss,
        ]


@OBJECT_REGISTRY.register
class TLAttrRPNSepLoss(nn.Module):
    """RPN Loss module with separate regression and classification loss.

    Compute classification loss and regression loss separately, keyed
    by 'rpn_cls_loss' and 'rpn_reg_loss' respectively. The type of
    each loss is specified in corresponding config.

    Args:
        cls_loss: classification loss module.
        reg_loss: regression loss module.
    """

    def __init__(
        self,
        cls_loss: nn.Module,
        reg_loss: nn.Module,
    ):
        super().__init__()
        self.cls_loss = cls_loss
        self.reg_loss = reg_loss

    @autocast(enabled=False)
    def forward(
        self,
        head_out: Dict[str, List[torch.Tensor]],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        reg_pred, cls_pred = rearrange_det_dense_head_out(
            [v.float() for v in head_out["rpn_reg_pred"]],
            [v.float() for v in head_out["rpn_cls_pred"]],
        )

        attr_pred_list = [
            rearrange_det_attr_head_out(
                [v.float() for v in attr_pred],
            )
            for _, attr_pred in head_out["rpn_attr_pred_dict"].items()
        ]
        attrs_type = head_out["rpn_attr_pred_dict"].keys()
        attr_loss = [
            self.cls_loss(
                pred=attr_pred.flatten(end_dim=-2),
                target=label.flatten(end_dim=-2),
                weight=mask.flatten(end_dim=-2),
            )
            for attr_pred, label, mask in zip(
                attr_pred_list,
                targets["attr_label_list"],
                targets["attr_label_mask_list"],
            )
        ]

        cls_loss = self.cls_loss(
            pred=cls_pred.flatten(end_dim=-2),
            target=targets["cls_label"].flatten(end_dim=-2),
            weight=targets["cls_label_mask"].flatten(end_dim=-2),
        )

        reg_label_mask = targets["reg_label_mask"].expand_as(
            targets["reg_label"]
        )
        reg_loss = self.reg_loss(
            pred=reg_pred.flatten(end_dim=-2),
            target=targets["reg_label"].flatten(end_dim=-2),
            weight=reg_label_mask.flatten(end_dim=-2),
            avg_factor=reg_label_mask.sum() + 1e-6,
        )
        attr_loss_dict = {}
        for attr, loss in zip(attrs_type, attr_loss):
            attr_loss_dict[f"{attr}_loss"] = loss
        loss_dict = OrderedDict(rpn_cls_loss=cls_loss, rpn_reg_loss=reg_loss)
        loss_dict.update(attr_loss_dict)
        return loss_dict
