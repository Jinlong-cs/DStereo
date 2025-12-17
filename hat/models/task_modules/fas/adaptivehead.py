# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List

import torch.nn as nn
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY

__all__ = ["FasAdaptiveHead"]


@OBJECT_REGISTRY.register
class FasAdaptiveHead(nn.Module):
    """Adaptive multi Head for face-anti-spoof.

    Args:
        in_channels : Channels of each input feature map.
        database_labels : Car types of imgs in datasets.
    """

    def __init__(
        self,
        in_channels: int,
        database_labels: List,
    ):

        super(FasAdaptiveHead, self).__init__()
        assert max(database_labels) == len(set(database_labels)) - 1
        self.neck = ConvModule2d(
            in_channels,
            128,
            kernel_size=3,
            stride=2,
            padding=1,
            act_layer=nn.ReLU(inplace=True),
        )
        self.multi_head = nn.ModuleList()
        for _i in range(max(database_labels) + 1):
            cur_conv = ConvModule2d(
                128,
                1,
                kernel_size=2,
                stride=1,
                padding=0,
            )
            self.multi_head.append(cur_conv)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.neck(x)
        preds = []
        for head in self.multi_head:
            pred = head(x)
            pred = self.dequant(pred)
            preds.append(pred)
        return preds

    def fuse_model(self):
        self.neck.fuse_model()
        for module in self.multi_head:
            for m in module:
                if hasattr(m, "fuse_model"):
                    m.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        for head in self.multi_head:
            head.qconfig = qconfig_manager.get_default_qat_out_qconfig()
