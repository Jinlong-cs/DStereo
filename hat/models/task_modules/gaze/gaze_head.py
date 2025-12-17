# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Dict, List

import torch.nn as nn
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d

__all__ = ["GazeHead"]


class GazeHead(nn.Module):
    """General block for output layers.

    receive feature maps from backbone and output channelx1x1 results.

    Args:
        channels: Number of channels for each FC layers
        bn_kwargs: Batch norm parameters
        use_pool: Use global pooling at the beginning if True
        output_add_bias: Use bias in output layer if True
    """

    def __init__(
        self,
        channels: List,
        bn_kwargs: Dict,
        use_pool=True,
        output_add_bias=True,
        dropout_ratio=0.5,
        **kwargs,
    ):
        super(GazeHead, self).__init__()

        self.use_pool = use_pool
        self.kernel = 1 if use_pool else 3
        self.dropout_ratio = dropout_ratio

        # gaze head network
        self.net = nn.Sequential()
        if self.use_pool:
            self.net.add_module(
                name="pool_module",
                module=nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            )
        else:
            ...

        for i in range(len(channels[:-2])):
            self.net.add_module(
                name=f"gaze_head{i}",
                module=ConvModule2d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=self.kernel,
                    stride=1,
                    padding=0,
                    groups=1,
                    bias=False,
                    norm_layer=nn.BatchNorm2d(channels[i + 1], **bn_kwargs),
                    act_layer=nn.ReLU(),
                ),
            )

        if self.dropout_ratio > 0:
            self.net.add_module(
                name="dropout_module",
                module=nn.Dropout(p=self.dropout_ratio),
            )
        # calu out layer input channel
        outlayer_inputchannel = channels[-2]
        self.output_layer = ConvModule2d(
            in_channels=outlayer_inputchannel,
            out_channels=channels[-1],
            kernel_size=self.kernel,
            stride=1,
            padding=0,
            groups=1,
            bias=output_add_bias,
            norm_layer=None,
            act_layer=None,
        )
        self.dequant = DeQuantStub()

    def forward(self, x):
        extend_input = x
        output = self.net(extend_input["shared_feat"])
        output = self.output_layer(output)
        gaze = self.dequant(output)

        return {"gaze": gaze}

    def fuse_model(self):
        for module in self.net:
            if hasattr(module, "fuse_model"):
                module.fuse_model()
        self.output_layer.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        self.output_layer.qconfig = (
            qconfig_manager.get_default_qat_out_qconfig()
        )
