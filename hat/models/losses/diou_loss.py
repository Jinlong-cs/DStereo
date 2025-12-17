# Copyright (c) Horizon Robotics. All rights reserved.
# Source code reference to mmdetection, gluonnas

from typing import Optional, Union

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from hat.registry import OBJECT_REGISTRY
from .utils import weight_reduce_loss

__all__ = ["DIoULoss"]


@OBJECT_REGISTRY.register
class DIoULoss(nn.Module):
    """Generalized Intersection over Union Loss.

    Args:
        loss_weight: Global weight of loss. Defaults is 1.0.
        eps: A small value to avoid zero denominator.
        reduction: The method used to reduce the loss. Options are
            [`none`, `mean`, `sum`].
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        eps: float = 1e-6,
        reduction: str = "mean",
    ):
        super(DIoULoss, self).__init__()
        self.loss_weight = loss_weight
        self.eps = eps
        self.reduction = reduction

    @staticmethod
    def _cal_diou_loss(pred, target, eps=1e-6):
        # overlap
        lt = torch.max(pred[..., :2], target[..., :2])
        rb = torch.min(pred[..., 2:], target[..., 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]
        # union
        ap = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
        ag = (target[..., 2] - target[..., 0]) * (
            target[..., 3] - target[..., 1]
        )
        union = ap + ag - overlap + eps
        # IoU
        ious = overlap / union
        # distance of central points
        center_p_x = (pred[..., 0] + pred[..., 2]) / 2
        center_p_y = (pred[..., 1] + pred[..., 3]) / 2
        center_g_x = (target[..., 0] + target[..., 2]) / 2
        center_g_y = (target[..., 1] + target[..., 3]) / 2
        c_d_square = (center_p_x - center_g_x) ** 2 + (
            center_p_y - center_g_y
        ) ** 2
        # diagonal of enclose
        enclose_x1y1 = torch.min(pred[..., :2], target[..., :2])
        enclose_x2y2 = torch.max(pred[..., 2:], target[..., 2:])
        enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)
        diagonal_square = (
            enclose_wh[..., 0] ** 2 + enclose_wh[..., 1] ** 2 + eps
        )
        # DIoU
        dious = ious - c_d_square / diagonal_square
        loss = 1 - dious
        return loss

    @autocast(enabled=False)
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        avg_factor: Optional[Union[float, torch.Tensor]] = None,
    ):
        """
        Forward method.

        Args:
            pred: Predicted bboxes of format (x1, y1, x2, y2),
                represent upper-left and lower-right point, with shape(N, 4).
            target: Corresponding gt_boxes, the same shape as
                pred.
            weight: Element-wise weight loss weight, with
                shape(N,).
            avg_factor: Average factor that is used to average the
                loss.
        """
        # cast to fp32
        pred = pred.float()
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()

        if weight is not None and weight.dim() > 1:
            # reduce the weight of shape (n, 4) to (n,) to match the
            # diou_loss of shape (n,)
            assert weight.shape == pred.shape, (
                "pred and weight should have same shape,"
                + "but got {} and {}".format(pred.shape, weight.shape)
            )
            weight = weight.mean(-1)

        loss = self._cal_diou_loss(pred, target, self.eps)
        loss = weight_reduce_loss(loss, weight, self.reduction, avg_factor)

        loss = self.loss_weight * loss

        return loss
