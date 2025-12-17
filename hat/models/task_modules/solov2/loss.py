# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from hat.models.losses.utils import weight_reduce_loss
from hat.registry import OBJECT_REGISTRY

__all__ = ["SOLOV2Loss", "MaskLoss"]


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss for dense detection.

    Args:
        inputs: A float tensor of arbitrary shape.
            The predictions for each example.
        targets: A float tensor with the same shape as inputs.
            Stores the binary classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
            positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
            balance easy vs hard examples.
    Returns:
        Loss tensor
    """

    prob = inputs.sigmoid()
    loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )

    if gamma > 0:
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss


def dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-3,
    reduction: str = "mean",
    naive_dice: bool = False,
    avg_factor: Optional[int] = None,
):
    """Calculate dice loss.

    Args:
        pred: The prediction, has a shape (n, *)
        target: The learning label of the prediction, shape (n, *),
            same shape of pred.
        weight: The weight of loss for each prediction, has a shape (n,).
            Defaults to None.
        eps: Avoid dividing by zero. Default: 1e-3.
        reduction: The method used to reduce the loss into a scalar.
            Defaults to 'mean'. Options are "none", "mean" and "sum".
        naive_dice: If false, use the dice loss defined in the V-Net paper,
            otherwise, use the naive dice loss in which the power of the number
            in the denominator is the first power instead of the second power.
            Defaults to False.
        avg_factor: Average factor that is used to average the loss.
            Defaults to None.
    """

    input = pred.flatten(1)
    target = target.flatten(1).float()

    a = torch.sum(input * target, 1)
    if naive_dice:
        b = torch.sum(input, 1)
        c = torch.sum(target, 1)
        d = (2 * a + eps) / (b + c + eps)
    else:
        b = torch.sum(input * input, 1) + eps
        c = torch.sum(target * target, 1) + eps
        d = (2 * a) / (b + c)

    loss = 1 - d
    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@OBJECT_REGISTRY.register
class MaskLoss(nn.Module):
    def __init__(
        self,
        use_sigmoid: bool = True,
        activate: bool = True,
        reduction: str = "mean",
        naive_dice: bool = False,
        loss_weight: float = 1.0,
        eps: float = 1e-3,
        loss_types: Tuple[str] = ("dice", "focal", "bce"),
    ):
        """Compute dice loss, bce loss, and focal loss for mask predictions.

        Args:
            use_sigmoid: Whether to the prediction is used for sigmoid or
                softmax. Defaults to True.
            activate: Whether to activate the predictions inside,
                this will disable the inside sigmoid operation.
                Defaults to True.
            reduction: The method used to reduce the loss.
                Options are "none", "mean" and "sum". Defaults to 'mean'.
            naive_dice: If false, use the dice loss defined in the V-Net paper,
                otherwise, use the naive dice loss in which the power of the
                number in the denominator is the first power instead of the
                second power. Defaults to False.
            loss_weight: Weight of loss. Defaults to 1.0.
            eps: Avoid dividing by zero. Defaults to 1e-3.
            loss_types: Losses for computing,
                including 'dice', 'focal', and 'bce'.
        """

        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.activate = activate
        self.reduction = reduction
        self.naive_dice = naive_dice
        self.loss_weight = loss_weight
        self.eps = eps
        self.loss_types = loss_types

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        reduction_override: Optional[str] = None,
        avg_factor: Optional[int] = None,
    ):
        """Forward function.

        Args:
            pred: The prediction, has a shape (n, *).
            target: The label of the prediction,
                shape (n, *), same shape of pred.
            weight: The weight of loss for each
                prediction, has a shape (n,). Defaults to None.
            reduction_override: The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".
            avg_factor: Average factor that is used to average
                the loss. Defaults to None.
        """

        assert reduction_override in (None, "none", "mean", "sum")
        reduction = (
            reduction_override if reduction_override else self.reduction
        )

        loss = 0
        if "focal" in self.loss_types:
            assert self.activate and self.use_sigmoid
            loss_f = sigmoid_focal_loss(pred, target)

            loss = loss + loss_f.mean((-1, -2))

        if "bce" in self.loss_types:
            assert self.activate and self.use_sigmoid
            loss_b = F.binary_cross_entropy_with_logits(
                pred, target, reduction="none"
            )
            loss = loss + loss_b.mean((-1, -2))

        if "dice" in self.loss_types:
            if self.activate:
                if self.use_sigmoid:
                    pred = pred.sigmoid()
                else:
                    raise NotImplementedError
            loss_d = dice_loss(
                pred,
                target,
                weight,
                eps=self.eps,
                reduction=reduction,
                naive_dice=self.naive_dice,
                avg_factor=avg_factor,
            )
            loss = loss + loss_d

        return loss * self.loss_weight


@OBJECT_REGISTRY.register
class SOLOV2Loss(torch.nn.Module):
    """SOLOV2Loss loss wrapper."""

    def __init__(
        self,
        cls_loss: torch.nn.Module,
        mask_loss: torch.nn.Module,
        attr_loss: List[torch.nn.Module],
        cls_loss_name: str,
        mask_loss_name: str,
        attr_loss_names: List[str],
    ):
        super().__init__()
        self.cls_loss = cls_loss
        self.mask_loss = mask_loss
        self.attr_loss = attr_loss
        self.cls_loss_name = cls_loss_name
        self.mask_loss_name = mask_loss_name
        self.attr_loss_names = attr_loss_names

    @autocast(enabled=False)
    def forward(self, pred: Tuple, target: Tuple[Dict]) -> Dict:
        assert len(target) == 3  # (cls_target, mask_target, attr_targets)
        res = {}
        # `pred` is in target tuple, so we get pred from target.
        cls_res = {self.cls_loss_name: self.cls_loss(**target[0])}
        res.update(cls_res)

        mask_preds = target[1]["pred"]
        mask_targets = target[1]["target"]
        num_pos = target[1]["avg_factor"]
        mask_loss = []

        for img_mask_preds, img_mask_targets in zip(mask_preds, mask_targets):
            if img_mask_preds is None:
                continue
            mask_loss.append(
                self.mask_loss(
                    img_mask_preds, img_mask_targets, reduction_override="none"
                )
            )
        if num_pos == 0:
            mask_loss = target[1]["pred_mean"].sum() * 0.0
        else:
            mask_loss = torch.cat(mask_loss).sum() / num_pos

        mask_res = {self.mask_loss_name: mask_loss}
        res.update(mask_res)

        attr_res = {}
        assert len(self.attr_loss) == len(target[2])
        for attr_loss, t, loss_name in zip(
            self.attr_loss, target[2], self.attr_loss_names
        ):
            if t["avg_factor"] == 0:
                attr_res.update({loss_name: t["pred_mean"].sum() * 0.0})
            else:
                attr_res.update({loss_name: attr_loss(**t)})

        res.update(attr_res)
        # Three dict shouldn't contain same key.
        assert len(res) == len(cls_res) + len(mask_res) + len(
            attr_res
        ), "results have same name keys, this may cause bugs."
        return res
