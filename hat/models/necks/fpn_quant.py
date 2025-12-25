# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List

import horizon_plugin_pytorch
import torch
import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class FPNQuant(nn.Module):
    """
    FpnQuant quant fpn/bifpn features used for two stage models mainly.

    Args:
        stride_num (int): Number of fpn/bifpn strides.
    """

    def __init__(self, stride_num: int = 4):
        super().__init__()
        self.stride_num = stride_num
        self.create_quant_ops()

    def create_quant_ops(self):
        self.quant_ops = torch.nn.ModuleList()
        for _ in range(self.stride_num):
            self.quant_ops.append(QuantStub(scale=None))

    def forward(self, feats: List[torch.Tensor]):
        assert (
            len(feats) == self.stride_num
        ), "length of feats must be equal to stride_num"

        feats_out = [self.quant_ops[i](fmap) for i, fmap in enumerate(feats)]
        return feats_out

    def set_qconfig(self):
        self.qconfig = horizon_plugin_pytorch.quantization.get_default_qat_qconfig()


@OBJECT_REGISTRY.register
class FPNDeQuant(nn.Module):
    """FPNDeQuant dequant fpn/bifpn features used for two stage models mainly."""  # noqa

    def __init__(self):
        super().__init__()
        self.dequant = DeQuantStub()

    def forward(self, feats: List[torch.Tensor]):
        feats = [self.dequant(fmap) for fmap in feats]
        return feats

    def set_qconfig(self):
        self.qconfig = horizon_plugin_pytorch.quantization.get_default_qat_out_qconfig()
