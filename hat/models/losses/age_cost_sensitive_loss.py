# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY

__all__ = [
    "CostSensitiveLoss",
]


@OBJECT_REGISTRY.register
class CostSensitiveLoss(nn.Module):
    r"""Calculates the cost sensitive error between label and pred.

    L = \sum_{k}Cost_{k}(a_{gt}) * \left \| f_{k}(I) -
            \bold{1}[a_{gt} > k] \right \|_{2}^{2}.

    Args:
        num_classes: Predict age in [0, age_classes].
        error_max: The maximum permissible error. Defaults to 3.
        from_sigmoid: Whether preds are actived by sigmoid. Defaults to False.

    """

    def __init__(
        self,
        num_classes: int,
        error_max: int = 3,
        from_sigmoid: bool = False,
        name: Optional[str] = None,
    ):

        super(CostSensitiveLoss, self).__init__()
        self.num_classes = num_classes
        self.error_max = error_max
        self.from_sigmoid = from_sigmoid
        self.name = name

    def _calc_cost(self, label_age):
        assert len(label_age.shape) == 1

        age = label_age.repeat_interleave(self.num_classes).reshape(
            -1, self.num_classes
        )
        base = torch.arange(1, self.num_classes + 1).to(age.device)
        abs_diff = torch.abs(age - base)
        cost = abs_diff >= self.error_max
        return cost

    def forward(self, pred, label_age, label_ord):
        if not self.from_sigmoid:
            pred = torch.sigmoid(pred)
        cost = self._calc_cost(label_age)
        loss_square = torch.square(pred - label_ord).squeeze()
        assert loss_square.shape[1] == self.num_classes
        mask = (label_age != -1).flatten()
        loss = torch.sum(loss_square * cost, axis=1) * mask
        loss = torch.sum(loss)

        if self.name is None:
            return loss
        else:
            return {self.name: loss}
