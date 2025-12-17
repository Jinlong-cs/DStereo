# -*- coding: utf-8 -*-
# Copyright (c) Horizon Robotics. All rights reserved.

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY

__all__ = ["CosFace"]


@OBJECT_REGISTRY.register
class CosFace(nn.Module):
    """
    Implementation for CosFace.

    Args:
        margin_arc: margin angle
        margin_am: hard thresh
        scale: scale
    """

    def __init__(self, scale: float = 64.0, margin: float = 0.40):
        super(CosFace, self).__init__()
        self.scale = scale
        self.margin = margin

    def forward(self, cos_theta: torch.Tensor, labels: torch.Tensor):
        cos_theta = cos_theta.clamp(-1, 1)
        valid_label_index = torch.where(labels != -1)[0]
        batch_size = cos_theta.size()[0]
        ind = torch.arange(batch_size).to(torch.long)
        valid_ind = ind[valid_label_index]
        valid_labels = labels[valid_label_index]
        output = cos_theta * 1.0
        output[valid_ind, valid_labels] -= self.margin
        output *= self.scale

        return output
