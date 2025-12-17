# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY

__all__ = [
    "FasMultiheadFocalLoss",
]


@OBJECT_REGISTRY.register
class FasMultiheadFocalLoss(nn.Module):
    """Computes the multi-head focal loss.

    The loss of each database is calculated separately.

    Args:
        gamma: The `gamma` parameter in focal loss. Defaults to 2.
        name: The key of loss in return dict.
    """

    def __init__(self, gamma: int = 2, name: Optional[str] = None):

        super(FasMultiheadFocalLoss, self).__init__()
        self.gamma = gamma
        self.name = name

    def forward(self, preds, fas_label, car_cls):
        losses = 0
        pred_multi_head = [
            nn.functional.sigmoid(pred.squeeze()) for pred in preds
        ]
        for i in range(len(preds)):
            # Get the index corresponding to the current database label.
            pred_cur_car_head = pred_multi_head[i]
            cur_car_samples = car_cls == i
            if sum(cur_car_samples) == 0:
                continue
            # Get the scores of samples that are corresponding to
            # the current database label.
            pred_samples = pred_cur_car_head[cur_car_samples]

            label_cur_car = fas_label[cur_car_samples]
            one_hot = label_cur_car > 0
            pt = torch.where(one_hot, pred_samples, 1 - pred_samples)
            loss = (
                -1
                * torch.pow((1 - pt), self.gamma)
                * torch.log(torch.clip_(pt, min=1e-12, max=1.0))
            )

            losses += loss.sum() / sum(cur_car_samples)

        if self.name is None:
            return losses
        else:
            return {self.name: losses}
