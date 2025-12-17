# Copyright (c) Horizon Robotics. All rights reserved.
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from hat.registry import OBJECT_REGISTRY

__all__ = ["PersonPositionLoss"]


@OBJECT_REGISTRY.register
class PersonPositionLoss(nn.Module):
    """Person position loss module.

    Loss structure that collect img, label and loss function
    for person position.

    Args:
        oms_loss: OMS loss function.
        dms_loss: DMS loss function.
    """

    def __init__(
        self,
        oms_loss: nn.Module,
        dms_loss: nn.Module,
    ):
        super().__init__()
        self.oms_loss = oms_loss
        self.dms_loss = dms_loss

    @autocast(enabled=False)
    def forward(self, preds, labels):

        # convert to float32 while using amp
        for k, v in preds.items():
            preds[k] = v.float()

        pred_oms = preds["pred_oms"].flatten(1, -1)
        pred_dms = preds["pred_dms"].flatten(1, -1)

        dms_label = labels["dms_cls_label"].view(
            -1,
        )
        dms_label_weight = labels["dms_cls_label_weight"].view(
            -1,
        )

        oms_label = labels["oms_cls_label"].view(
            -1,
        )
        oms_label_weight = labels["oms_cls_label_weight"].view(
            -1,
        )
        p_dms = torch.argmax(pred_dms, 1)
        p_oms = torch.argmax(pred_oms, 1)
        # keep valid label which is not equal to -1
        keep_dms = torch.where(dms_label != -1)[0]
        keep_oms = torch.where(oms_label != -1)[0]
        dms_valid_label = dms_label[keep_dms]
        oms_valid_label = oms_label[keep_oms]
        # keep pred with corresponding to valid label
        p_valid_dms = p_dms[keep_dms]
        p_valid_oms = p_oms[keep_oms]
        # cal oms and dms label number that predict correctly
        correct_dms = torch.sum(p_valid_dms == dms_valid_label)
        correct_oms = torch.sum(p_valid_oms == oms_valid_label)
        # cal oms and ams accuracy of prediciton
        dms_acc = correct_dms / (dms_valid_label.size(0) + 1e-6)
        oms_acc = correct_oms / (oms_valid_label.size(0) + 1e-6)

        oms_loss = self.oms_loss(
            pred=pred_oms,
            target=oms_label,
            weight=oms_label_weight,
        )

        dms_loss = self.dms_loss(
            pred=pred_dms,
            target=dms_label,
            weight=dms_label_weight,
        )

        output_loss = OrderedDict(
            oms_acc=oms_acc,
            dms_acc=dms_acc,
            oms_loss=oms_loss,
            dms_loss=dms_loss,
        )

        return output_loss
