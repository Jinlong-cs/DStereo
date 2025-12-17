# -*- coding: utf-8 -*-
# Copyright (c) Horizon Robotics. All rights reserved.

import math

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY

EPS = 1e-10

__all__ = ["ArcFace"]


@OBJECT_REGISTRY.register
class ArcFace(nn.Module):
    """
    Implementation for ArcFace.

    Args:
        margin_arc: margin angle
        margin_am: hard thresh
        scale: scale
    """

    def __init__(
        self,
        margin_arc: float = 0.5,
        margin_am: float = 0.0,
        scale: float = 64.0,
    ):
        super(ArcFace, self).__init__()
        self.margin_arc = margin_arc
        self.margin_am = margin_am
        self.scale = scale
        self.cos_margin = math.cos(margin_arc)
        self.sin_margin = math.sin(margin_arc)
        self.min_cos_theta = math.cos(math.pi - margin_arc)

    def forward(self, cos_theta: torch.Tensor, labels: torch.Tensor):
        cos_theta = cos_theta.clamp(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2) + EPS)
        cos_theta_m = cos_theta * self.cos_margin - sin_theta * self.sin_margin

        cos_theta_m = torch.where(
            cos_theta > self.min_cos_theta,
            cos_theta_m,
            cos_theta - self.margin_am,
        )

        valid_label_index = torch.where(labels != -1)[0]
        batch_size = cos_theta.size()[0]
        ind = torch.arange(batch_size).to(torch.long).to(labels.device)
        valid_label = labels[valid_label_index]
        valid_ind = ind[valid_label_index]

        output = cos_theta * 1.0
        output[valid_ind, valid_label] = cos_theta_m[valid_ind, valid_label]
        output *= self.scale
        return output
