# Copyright (c) Horizon Robotics, All rights reserved.

from typing import Optional

import torch
from torch import nn

from hat.registry import OBJECT_REGISTRY

__all__ = [
    "Conv2dSubSampling",
    "SimpleMixBlock",
]


@OBJECT_REGISTRY.register
class Conv2dSubSampling(nn.Module):
    def __init__(
        self,
        in_module: Optional[nn.Sequential] = None,
        sub_module: nn.Sequential = None,
        feat_as_channel: bool = False,
        out_module: Optional[nn.Sequential] = None,
    ):

        super(Conv2dSubSampling, self).__init__()
        self.in_module = in_module
        self.sub_module = sub_module
        self.feat_as_channel = feat_as_channel
        self.out_module = out_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_module is not None:
            x = self.in_module(x)
        x = x.unsqueeze(1)  # b, t, c - > b, 1, t, c
        if self.feat_as_channel:
            x = x.transpose(1, 3)  # b, c, t, 1
        x = self.sub_module(x)  # b, c, t, 1(f)
        if self.feat_as_channel:
            x = x.transpose(1, 3).squeeze(1)
        else:
            b, c, t, f = x.size()
            x = x.transpose(1, 2).contiguous().view(b, t, c * f)
        if self.out_module is not None:
            x = self.out_module(x)
        return x


@OBJECT_REGISTRY.register
class SimpleMixBlock(nn.Module):

    MODES = ["add", "cat"]

    def __init__(self, mode: str = "add", **kwargs):
        super().__init__()
        assert mode in self.MODES, f"Unknown Mix Mode: {mode}"
        self.mode = mode
        self.mix_op = nn.quantized.FloatFunctional()
        self.kwargs = kwargs

    def forward(self, fea1, fea2):
        if self.mode == "add":
            return self.mix_op.add(fea1, fea2, **self.kwargs)
        elif self.mode == "cat":
            return self.mix_op.cat((fea1, fea2), **self.kwargs)
        else:
            return None
