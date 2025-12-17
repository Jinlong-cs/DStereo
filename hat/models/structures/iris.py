# Copyright (c) Horizon Robotics. All rights reserved.

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list

__all__ = ["IrisClassifier"]


@OBJECT_REGISTRY.register
class IrisClassifier(nn.Module):
    """The classifier of iris visibility.

    Args:
        backbone: Backbone module.
        head: Head module.
        losses: Losses module.
        loss_weights: Loss weight for each branch.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        losses=None,
        loss_weights: tuple = (1, 1),
    ):
        super(IrisClassifier, self).__init__()
        self.backbone = backbone
        self.head = head
        self.losses = losses
        self.loss_weights = loss_weights

    def forward(self, data):
        image = data["img"]
        target = data.get("labels", None)

        preds = self.head(self.backbone(image)[-1])
        if target is None:
            return preds

        if not self.training or self.losses is None:
            return preds, target

        def merge_preds_label(preds, target):
            return _as_list(preds) + _as_list(target)

        merge_data = merge_preds_label(preds, target)
        l_loss, r_loss = self._cal_loss(merge_data)

        return [preds, (l_loss + r_loss), l_loss, r_loss]

    def _cal_loss(self, x):
        pred_l = torch.reshape(x[0], (-1, 2))
        pred_r = torch.reshape(x[1], (-1, 2))
        mask_l, mask_r = x[3], x[4]
        label_l, label_r = x[5], x[6]

        def mask_loss(mask, pred, label, loss_func):
            mask_pred = pred.index_select(
                dim=0, index=torch.where(mask == 1)[0]
            )
            mask_label = label.index_select(
                dim=0, index=torch.where(mask == 1)[0]
            )
            loss = loss_func(
                mask_pred,
                mask_label.long(),
            )
            return loss

        loss_l = mask_loss(mask_l, pred_l, label_l, self.losses)
        loss_r = mask_loss(mask_r, pred_r, label_r, self.losses)

        return self.loss_weights[0] * torch.sum(
            loss_l, dim=0
        ), self.loss_weights[1] * torch.sum(loss_r, dim=0)

    def fuse_model(self):
        for module in [self.backbone, self.head]:
            if module is not None and hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        self.losses.qconfig = None
        for module in [self.backbone, self.head]:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()
