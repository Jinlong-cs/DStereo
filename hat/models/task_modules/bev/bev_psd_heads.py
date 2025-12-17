# Copyright (c) Horizon Robotics. All rights reserved.

from typing import List, Mapping

import horizon_plugin_pytorch as horizon
import horizon_plugin_pytorch.nn as hnn
import torch.nn as nn
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.base_modules.separable_conv_module import (
    SeparableGroupConvModule2d,
)
from hat.models.weight_init import bias_init_with_prob, normal_init
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "ANCBEVPSDGlobalHead",
    "ANCBEVPSDLocalHead",
    "ANCBEVPSDHead",
]


@OBJECT_REGISTRY.register
class ANCBEVPSDHead(nn.Module):
    """Head module for ipm super psd task.

    reference to
    https://horizonrobotics.feishu.cn/docs/doccn1XThiPa1cWduH3SwqcOe3b
    Args:
        feature_name: name from stage2 neck output
        global_head: Global head module of super psd task.
        local_head: Local head module of super psd task.
        local_head_near: Local head module which focuses on near range.
    """

    def __init__(
        self,
        feature_name: str = None,
        global_head: nn.Module = None,
        local_head: nn.Module = None,
        local_head_near: nn.Module = None,
    ):
        super().__init__()
        self.global_head = global_head
        self.local_head = local_head
        self.feature_name = feature_name
        self.local_head_near = local_head_near

    def forward(self, data):
        features = (
            data[self.feature_name][0]
            if isinstance(data, Mapping)
            else data[0]
        )
        total_pred = []
        if self.global_head:
            global_pred = self.global_head(features)
            total_pred.extend(global_pred)
        if self.local_head:
            local_pred = self.local_head(features)
            total_pred.extend(local_pred)
        if self.local_head_near:
            local_pred_near = self.local_head_near(features)
            total_pred.extend(local_pred_near)
        return tuple(total_pred)

    def fuse_model(self):
        if hasattr(self.global_head, "fuse_model"):
            self.global_head.fuse_model()
        if hasattr(self.local_head, "fuse_model"):
            self.local_head.fuse_model()
        if self.local_head_near and hasattr(
            self.local_head_near, "fuse_model"
        ):
            self.local_head_near.fuse_model()

    def set_qconfig(self):
        self.qconfig = horizon.quantization.get_default_qat_qconfig()


@OBJECT_REGISTRY.register
class ANCBEVPSDGlobalHead(nn.Module):
    """Global head module for bev psd.

    Args:
        num_slot_type: The number of parking-space category(exclude background)
        in_channels: channels number of input tensor.
        out_channels: channels number of output tensor.
        in_strides: strides for input.
        out_stride: the stride of selected downsampling.
        group_base: A param for head block.
    """

    def __init__(
        self,
        num_slot_type: int,
        in_channels: int,
        out_channels: int,
        in_strides: list,
        out_stride: int,
        stack: int = 1,
        group_base: int = 8,
    ):
        super().__init__()
        self.in_strides = in_strides
        self.out_stride = out_stride
        group_num = int(in_channels / group_base)
        self.global_head_block = SeparableGroupConvModule2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            groups=group_num,
            padding=1,
            stride=1,
            pw_norm_layer=nn.BatchNorm2d(out_channels),
            pw_act_layer=nn.ReLU(inplace=True),
        )
        encode_global_block = []
        for _ in range(stack):
            block = ConvModule2d(
                out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_layer=nn.BatchNorm2d(out_channels),
                act_layer=nn.ReLU(inplace=True),
            )
            encode_global_block.append(block)
        self.encode_global_block = nn.Sequential(*encode_global_block)
        self.classification_conv = nn.Conv2d(out_channels, 1, 1, padding=0)
        self.offset_conv = nn.Conv2d(out_channels, 8, 1, padding=0)
        self.occupancy_conv = nn.Conv2d(out_channels, 1, 1, padding=0)
        self.slot_type_conv = nn.Conv2d(
            out_channels, num_slot_type, 1, padding=0
        )
        self.direction_conv = nn.Conv2d(out_channels, 2, 1, padding=0)
        self.dequant = DeQuantStub()
        self.init_weight()

    def forward(self, features):
        global_features = self.global_head_block(
            features[self.in_strides.index(self.out_stride)]
        )
        global_features = self.encode_global_block(global_features)
        classification = self.classification_conv(global_features)
        offset = self.offset_conv(global_features)
        occupancy = self.occupancy_conv(global_features)
        slot_type = self.slot_type_conv(global_features)
        direction = self.direction_conv(global_features)
        if not self.training:
            slot_type = horizon.argmax(slot_type, dim=1, keepdim=True)
        else:
            slot_type = self.dequant(slot_type)
        classification = self.dequant(classification)
        offset = self.dequant(offset)
        occupancy = self.dequant(occupancy)
        direction = self.dequant(direction)

        return [classification, offset, occupancy, slot_type, direction]

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.classification_conv, std=0.01, bias=bias_cls)

    def set_qconfig(self):
        self.qconfig = horizon.quantization.get_default_qat_qconfig()

    def fuse_model(self):
        self.global_head_block.fuse_model()
        assert isinstance(self.encode_global_block, nn.Sequential)
        for block in self.encode_global_block:
            if hasattr(block, "fuse_model"):
                block.fuse_model()


@OBJECT_REGISTRY.register
class ANCBEVPSDLocalHead(nn.Module):
    """Local head module for bev psd.

    Args:
        in_channels: channels number of input tensor.
        out_channels: channels number of output tensor.
        in_strides: strides for input.
        out_stride: the stride of selected downsampling.
        stack: The number of head block stacked.
        group_base: A param for head block.
        crop_roi: RoI of input feature, [x0, y0, x1, y1].
        crop_roi_output: RoI of output feature, [x0, y0, x1, y1].
        resize: Resize shape for the input feature, [H, W].
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_strides: list,
        out_stride: int,
        stack: int = 1,
        group_base: int = 8,
        crop_roi: List[int] = None,
        crop_roi_output: List[int] = None,
        resize: List[int] = None,
    ):
        super().__init__()
        self.in_strides = in_strides
        self.out_stride = out_stride
        group_num = int(in_channels / group_base)
        self.local_head_block = SeparableGroupConvModule2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            groups=group_num,
            padding=1,
            stride=1,
            pw_norm_layer=nn.BatchNorm2d(out_channels),
            pw_act_layer=nn.ReLU(inplace=True),
        )
        encode_local_block = []
        for _ in range(stack):
            block = ConvModule2d(
                out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_layer=nn.BatchNorm2d(out_channels),
                act_layer=nn.ReLU(inplace=True),
            )
            encode_local_block.append(block)
        self.encode_local_block = nn.Sequential(*encode_local_block)
        self.classification_conv = nn.Conv2d(out_channels, 4, 1, padding=0)
        self.offset_conv = nn.Conv2d(out_channels, 8, 1, padding=0)
        self.sline_angle_conv = nn.Conv2d(out_channels, 8, 1, padding=0)
        self.point_type_conv = nn.Conv2d(out_channels, 4, 1, padding=0)
        self.dequant = DeQuantStub()
        assert crop_roi is None or len(crop_roi) == 4
        self.crop_roi = crop_roi
        assert crop_roi_output is None or len(crop_roi_output) == 4
        self.crop_roi_output = crop_roi_output
        if resize is not None:
            assert len(resize) == 2
            self.resize_module = hnn.Interpolate(size=resize, mode="bilinear")
        else:
            self.resize_module = None
        self.init_weight()

    def forward(self, features):
        in_feat = features[self.in_strides.index(self.out_stride)]

        if self.resize_module is not None:
            in_feat = self.resize_module(in_feat)
        if self.crop_roi is not None:
            in_feat = in_feat[
                :,
                :,
                self.crop_roi[0] : self.crop_roi[2],
                self.crop_roi[1] : self.crop_roi[3],
            ]

        local_features = self.local_head_block(in_feat)
        local_features = self.encode_local_block(local_features)

        if self.crop_roi_output is not None:
            local_features = local_features[
                :,
                :,
                self.crop_roi_output[0] : self.crop_roi_output[2],
                self.crop_roi_output[1] : self.crop_roi_output[3],
            ]

        classification = self.classification_conv(local_features)
        offset = self.offset_conv(local_features)
        sline_angle = self.sline_angle_conv(local_features)
        point_type = self.point_type_conv(local_features)
        classification = self.dequant(classification)
        offset = self.dequant(offset)
        sline_angle = self.dequant(sline_angle)
        point_type = self.dequant(point_type)
        return [classification, offset, sline_angle, point_type]

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.classification_conv, std=0.01, bias=bias_cls)

    def set_qconfig(self):
        self.qconfig = horizon.quantization.get_default_qat_qconfig()

    def fuse_model(self):
        self.local_head_block.fuse_model()
        assert isinstance(self.encode_local_block, nn.Sequential)
        for block in self.encode_local_block:
            if hasattr(block, "fuse_model"):
                block.fuse_model()
