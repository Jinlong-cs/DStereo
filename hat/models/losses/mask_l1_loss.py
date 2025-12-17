from typing import Optional

import torch.nn as nn
import torch.nn.functional as F

from hat.registry import OBJECT_REGISTRY
from .utils import weight_reduce_loss

__all__ = ["MaskL1Loss"]


@OBJECT_REGISTRY.register
class MaskL1Loss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        loss_name: Optional[str] = None,
    ):
        super(MaskL1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.loss_name = loss_name

    def forward(self, pred, target, avg_factor, weight=None, mask=None):
        loss = F.l1_loss(pred, target, reduction="none")
        if mask is not None:
            if pred.ndim - mask.ndim == 1:
                mask = mask[..., None]
            loss = loss * mask
        if weight is not None:
            if pred.ndim - weight.ndim == 1:
                weight = weight[..., None]
        loss = weight_reduce_loss(
            loss, weight, reduction=self.reduction, avg_factor=avg_factor
        )
        if self.loss_weight is not None:
            loss = self.loss_weight * loss
        if self.loss_name is None:
            return loss
        return {self.loss_name: loss}
