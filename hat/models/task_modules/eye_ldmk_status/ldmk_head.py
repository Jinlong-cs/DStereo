# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List

import torch.nn as nn
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "EyeLdmkHead",
    "EyeLdmkVectorHead",
    "EyeLdmkHeatmapHead",
]


@OBJECT_REGISTRY.register
class EyeLdmkHeatmapHead(nn.Module):
    """
    Convert feature map to predicted heatmap.

    Use 1x1 conv2d to transform feature to num_ldmk-channel heatmap.

    Args:
        in_channels: channel number of decoder module.
        num_ldmk: number of landmark.
    """

    def __init__(
        self,
        in_channels: int,
        num_ldmk: int,
    ):
        super(EyeLdmkHeatmapHead, self).__init__()
        self.num_ldmk = num_ldmk
        self.head = ConvModule2d(in_channels, self.num_ldmk, 1)
        self.dequant = DeQuantStub()

    def forward(self, x):
        heatmap_pred = self.head(x)
        heatmap_pred = self.dequant(heatmap_pred)
        return heatmap_pred

    def fuse_model(self):
        self.head.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        # disable output quantization for last quanti layer.
        self.head.qconfig = qconfig_manager.get_default_qat_out_qconfig()


@OBJECT_REGISTRY.register
class EyeLdmkHead(nn.Module):
    """
    Eye status heatmap and vector head.

    Args:
        heatmap_head: Module for heatmap.
        vector_head: Module for vector.
        bias: Whether to use bias in module.
    """

    def __init__(
        self,
        heatmap_head: nn.Module,
        vector_head: nn.Module,
    ):
        super(EyeLdmkHead, self).__init__()
        self.vector_head = vector_head
        self.heatmap_head = heatmap_head

    def forward(self, x):
        pred_x, pred_y = self.vector_head(x)

        heatmap = self.heatmap_head(x)
        return pred_x, pred_y, heatmap

    def fuse_model(self):
        modules = [self.vector_head, self.heatmap_head]
        for m in modules:
            if hasattr(m, "fuse_model"):
                m.fuse_model()

    def set_qconfig(self):
        modules = [self.vector_head, self.heatmap_head]
        for m in modules:
            if hasattr(m, "set_qconfig"):
                m.set_qconfig()


@OBJECT_REGISTRY.register
class EyeLdmkVectorHead(nn.Module):
    """Multi-branch for eye status cls.

    Eye status are divided into 2 groups: left eye, right eye.

    Args:
        num_ldmk: Eye ldmk num.
        in_channels: In_channel of module.
        bias: Whether to use bias in module.
        strides_x: X position strides list.
        kernel_x: X position kernel list.
        strides_y: Y position strides list.
        kernel_y: Y position kernel list.
    """

    def __init__(
        self,
        num_ldmk: int,
        in_channels: int,
        bias: bool = True,
        strides_x: List[int] = None,
        kernel_x: List[int] = None,
        strides_y: List[int] = None,
        kernel_y: List[int] = None,
    ):
        super(EyeLdmkVectorHead, self).__init__()
        self.num_ldmk = num_ldmk
        self.in_channels = in_channels
        self.bias = bias

        strides_x_default = [2, 2, 2, 2, 2, 1]
        strides_y_default = [2, 2, 2, 2, 1]

        kernel_x_default = [2, 2, 2, 3, 1, 1]
        kernel_y_default = [2, 2, 2, 5, 1]

        strides_x_list = strides_x_default if strides_x is None else strides_x
        strides_x_list = [(s, 1) for s in strides_x_list]

        kernel_x_list = kernel_x_default if kernel_x is None else kernel_x
        kernel_x_list = [(k, 1) for k in kernel_x_list]

        self.stage_x = self._make_stage(
            kernel_list=kernel_x_list,
            stride_list=strides_x_list,
        )

        strides_y_list = strides_y_default if strides_y is None else strides_y
        strides_y_list = [(1, s) for s in strides_y_list]

        kernel_y_list = kernel_y_default if kernel_y is None else kernel_y
        kernel_y_list = [(1, k) for k in kernel_y_list]

        self.stage_y = self._make_stage(
            kernel_list=kernel_y_list,
            stride_list=strides_y_list,
        )
        self.q_config_x_pos = len(strides_x_list) - 1
        self.q_config_y_pos = len(strides_y_list) - 1

        self.dequant_x = DeQuantStub()
        self.dequant_y = DeQuantStub()

    def _make_stage(self, kernel_list, stride_list):
        layers = []
        in_channels = self.in_channels
        for kernel, stride in zip(kernel_list, stride_list):
            layers.append(
                ConvModule2d(
                    in_channels,
                    self.num_ldmk,
                    stride=stride,
                    kernel_size=kernel,
                    padding=(0, 0),
                    bias=self.bias,
                    norm_layer=None,
                    act_layer=None,
                )
            )
            in_channels = self.num_ldmk

        return nn.Sequential(*layers)

    def forward(self, x):
        x_pred = self.stage_x(x)
        y_pred = self.stage_y(x)

        vector_x_pred = self.dequant_x(x_pred)
        vector_y_pred = self.dequant_y(y_pred)

        return vector_x_pred, vector_y_pred

    def fuse_model(self):
        modules = [self.stage_x, self.stage_y]
        for m in modules:
            if hasattr(m, "fuse_model"):
                m.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        # disable output quantization for last quanti layer.
        getattr(
            self.stage_x, f"{self.q_config_x_pos}"
        ).qconfig = qconfig_manager.get_default_qat_out_qconfig()

        getattr(
            self.stage_y, f"{self.q_config_y_pos}"
        ).qconfig = qconfig_manager.get_default_qat_out_qconfig()
