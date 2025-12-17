# Copyright (c) Horizon Robotics. All rights reserved.

import horizon_plugin_pytorch as horizon
import torch.nn as nn
from horizon_plugin_pytorch.nn import Interpolate
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "SuperPSDGlobalHead",
    "SuperPSDLocalHead",
    "SuperPSDHead",
    "SuperPSDLocalCPNHead",
    "SuperPSDGlobalCPNHead",
]


@OBJECT_REGISTRY.register
class SuperPSDHead(nn.Module):
    """Head module for ipm super psd task.

    Args:
        global_head: Global head module of super psd task.
        local_head: Local head module of super psd task.
    """

    def __init__(
        self, global_head: nn.Module = None, local_head: nn.Module = None
    ):
        super().__init__()
        self.global_head = global_head
        self.local_head = local_head

    def forward(self, features):

        total_pred = []
        if self.global_head:
            global_pred = self.global_head(features)
            total_pred.extend(global_pred)
        if self.local_head:
            local_pred = self.local_head(features)
            total_pred.extend(local_pred)
        return tuple(total_pred)

    def fuse_model(self):
        if hasattr(self.global_head, "fuse_model"):
            self.global_head.fuse_model()
        if hasattr(self.local_head, "fuse_model"):
            self.local_head.fuse_model()

    def set_qconfig(self):
        self.qconfig = horizon.quantization.get_default_qat_qconfig()


class Bottleneck(nn.Module):
    """Bottleneck layers for CPN module.

    Args:
        inplanes: Channel number for input tensor.
        planes: Half channel number for output tensor.
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = ConvModule2d(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=1,
            bias=False,
            norm_layer=nn.BatchNorm2d(num_features=planes),
            act_layer=nn.ReLU(inplace=True),
        )
        self.conv2 = ConvModule2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            padding=1,
            bias=False,
            stride=stride,
            norm_layer=nn.BatchNorm2d(num_features=planes),
            act_layer=nn.ReLU(inplace=True),
        )
        self.conv3 = ConvModule2d(
            in_channels=planes,
            out_channels=planes * 2,
            kernel_size=1,
            bias=False,
            norm_layer=nn.BatchNorm2d(num_features=planes * 2),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = ConvModule2d(
            in_channels=inplanes,
            out_channels=planes * 2,
            kernel_size=1,
            stride=stride,
            bias=False,
            norm_layer=nn.BatchNorm2d(num_features=planes * 2),
        )
        self.add = nn.quantized.FloatFunctional()
        self.stride = stride

    def fuse_model(self):
        for m in self.modules():
            if isinstance(m, ConvModule2d):
                m.fuse_model()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.add.add(out, residual)
        out = self.relu(out)
        return out


class RefineNet(nn.Module):
    """RefineNet module for super psd local head module.

    Args:
        in_channels: Channel number for input tensor.
        out_shape: Shape for output tensor.
        out_channels: Channel number for output tensor.
    """

    def __init__(self, num_cascade, in_channels, out_shape, out_channels):
        super(RefineNet, self).__init__()
        cascade = []
        self.num_cascade = num_cascade
        for i in range(num_cascade):
            cascade.append(
                self.make_layer(in_channels, num_cascade - i - 1, out_shape)
            )
        self.cascade = nn.ModuleList(cascade)
        self.final_predict = self.predict(
            num_cascade * in_channels, out_channels
        )
        self.cat = nn.quantized.FloatFunctional()

    def make_layer(self, input_channel, num, output_shape):
        layers = []
        for _ in range(num):
            layers.append(Bottleneck(input_channel, 16))
        layers.append(Interpolate(size=output_shape, mode="bilinear"))
        return nn.Sequential(*layers)

    def predict(self, input_channel, out_channels):
        layers = []
        layers.append(Bottleneck(input_channel, 16))
        layers.append(
            ConvModule2d(
                in_channels=32,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm_layer=nn.BatchNorm2d(num_features=out_channels),
            )
        )
        return nn.Sequential(*layers)

    def fuse_model(self):
        for m in self.modules():
            if isinstance(m, ConvModule2d):
                m.fuse_model()

    def forward(self, x):
        refine_fms = []
        for i in range(self.num_cascade):
            refine_fms.append(
                self.cascade[self.num_cascade - 1 - i](
                    x[self.num_cascade - 1 - i]
                )
            )
        out = self.cat.cat(refine_fms, dim=1)
        out = self.final_predict(out)
        return out


@OBJECT_REGISTRY.register
class SuperPSDGlobalCPNHead(nn.Module):
    """CPN global head module for super psd.

    Args:
        in_channels: channels number of input tensor.
        out_shape: shape of output tensor.
        out_channels: channels number of output tensor.
    """

    def __init__(
        self,
        num_cascade: int,
        num_slot_type: int,
        in_channels: int,
        out_shape: int,
        out_channels: int,
    ):
        super().__init__()
        self.cpn = RefineNet(num_cascade, in_channels, out_shape, out_channels)
        self.classification_conv = nn.Conv2d(in_channels, 1, 1, padding=0)
        self.offset_conv = nn.Conv2d(in_channels, 8, 1, padding=0)
        self.occupancy_conv = nn.Conv2d(in_channels, 1, 1, padding=0)
        self.slot_type_conv = nn.Conv2d(
            in_channels, num_slot_type, 1, padding=0
        )
        self.direction_conv = nn.Conv2d(in_channels, 2, 1, padding=0)
        self.dequan = DeQuantStub()

    def forward(self, features):
        global_features = self.cpn(features)
        classification = self.classification_conv(global_features)
        offset = self.offset_conv(global_features)
        occupancy = self.occupancy_conv(global_features)
        slot_type = self.slot_type_conv(global_features)
        direction = self.direction_conv(global_features)
        if not self.training:
            slot_type = horizon.argmax(slot_type, dim=1, keepdim=True)
        else:
            slot_type = self.dequan(slot_type)
        classification = self.dequan(classification)
        offset = self.dequan(offset)
        occupancy = self.dequan(occupancy)
        direction = self.dequan(direction)

        return [classification, offset, occupancy, slot_type, direction]

    def set_qconfig(self):
        self.qconfig = horizon.quantization.get_default_qat_qconfig()

    def fuse_model(self):
        self.cpn.fuse_model()


@OBJECT_REGISTRY.register
class SuperPSDLocalCPNHead(nn.Module):
    """CPN local head module for super psd.

    Args:
        in_channels: channels number of input tensor.
        out_shape: shape of output tensor.
        out_channels: channels number of output tensor.
    """

    def __init__(self, num_cascade, in_channels, out_shape, out_channels):
        super().__init__()
        self.cpn = RefineNet(num_cascade, in_channels, out_shape, out_channels)
        self.classification_conv = nn.Conv2d(out_channels, 4, 1, padding=0)
        self.offset_conv = nn.Conv2d(out_channels, 8, 1, padding=0)
        self.sline_angle_conv = nn.Conv2d(out_channels, 8, 1, padding=0)
        self.point_type_conv = nn.Conv2d(out_channels, 4, 1, padding=0)
        self.dequan = DeQuantStub()
        # self.init_weight()

    def forward(self, features):

        local_features = self.cpn(features)
        classification = self.classification_conv(local_features)
        offset = self.offset_conv(local_features)
        sline_angle = self.sline_angle_conv(local_features)
        point_type = self.point_type_conv(local_features)
        classification = self.dequan(classification)
        offset = self.dequan(offset)
        sline_angle = self.dequan(sline_angle)
        point_type = self.dequan(point_type)
        return [classification, offset, sline_angle, point_type]

    def init_weight(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, 0)

    def set_qconfig(self):
        self.qconfig = horizon.quantization.get_default_qat_qconfig()

    def fuse_model(self):
        self.cpn.fuse_model()


@OBJECT_REGISTRY.register
class SuperPSDLocalHead(nn.Module):
    """Local head module for super psd task.

    Args:
        in_channels: Channel number of input tensor.
        stride_idx: Stride index of input tensor.
    """

    def __init__(self, in_channels: int, stride_idx: int):
        super(SuperPSDLocalHead, self).__init__()
        self.conv = nn.Sequential(
            ConvModule2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_layer=nn.BatchNorm2d(num_features=in_channels),
                act_layer=nn.ReLU(),
            ),
            ConvModule2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_layer=nn.BatchNorm2d(num_features=in_channels),
                act_layer=nn.ReLU(),
            ),
        )
        self.classification_conv = nn.Conv2d(in_channels, 4, 1, padding=0)
        self.offset_conv = nn.Conv2d(in_channels, 8, 1, padding=0)
        self.sline_angle_conv = nn.Conv2d(in_channels, 8, 1, padding=0)
        self.point_type_conv = nn.Conv2d(in_channels, 4, 1, padding=0)
        self.stride_idx = stride_idx
        self.dequan = DeQuantStub()
        self.init_weight()

    def forward(self, features):
        local_features = features[self.stride_idx]
        local_features = self.conv(local_features)
        classification = self.classification_conv(local_features)
        offset = self.offset_conv(local_features)
        sline_angle = self.sline_angle_conv(local_features)
        point_type = self.point_type_conv(local_features)
        classification = self.dequan(classification)
        offset = self.dequan(offset)
        sline_angle = self.dequan(sline_angle)
        point_type = self.dequan(point_type)
        return [classification, offset, sline_angle, point_type]

    def init_weight(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, 0)

    def set_qconfig(self):
        self.qconfig = horizon.quantization.get_default_qat_qconfig()

    def fuse_model(self):
        for m in self.modules():
            if isinstance(m, ConvModule2d):
                m.fuse_model()


@OBJECT_REGISTRY.register
class SuperPSDGlobalHead(nn.Module):
    """Global head module of super psd.

    Args:
        num_slot_type: Total number of different slot types.
        in_channels: Channel number of input tensor.
        stride_idx: Stride index of input tensor.

    """

    def __init__(self, num_slot_type: int, in_channels: int, stride_idx: int):
        super(SuperPSDGlobalHead, self).__init__()
        self.stride_idx = stride_idx
        self.conv = nn.Sequential(
            ConvModule2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_layer=nn.BatchNorm2d(num_features=in_channels),
                act_layer=nn.ReLU(),
            ),
            ConvModule2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_layer=nn.BatchNorm2d(num_features=in_channels),
                act_layer=nn.ReLU(),
            ),
        )
        self.classification_conv = nn.Conv2d(in_channels, 1, 1, padding=0)
        self.offset_conv = nn.Conv2d(in_channels, 8, 1, padding=0)
        self.occupancy_conv = nn.Conv2d(in_channels, 1, 1, padding=0)
        self.slot_type_conv = nn.Conv2d(
            in_channels, num_slot_type, 1, padding=0
        )
        self.direction_conv = nn.Conv2d(in_channels, 2, 1, padding=0)
        self.dequan = DeQuantStub()
        self.init_weight()

    def forward(self, features):
        global_features = features[self.stride_idx]
        global_features = self.conv(global_features)
        classification = self.classification_conv(global_features)
        offset = self.offset_conv(global_features)
        occupancy = self.occupancy_conv(global_features)
        slot_type = self.slot_type_conv(global_features)
        direction = self.direction_conv(global_features)
        if not self.training:
            slot_type = horizon.argmax(slot_type, dim=1, keepdim=True)
        else:
            slot_type = self.dequan(slot_type)
        classification = self.dequan(classification)
        offset = self.dequan(offset)
        occupancy = self.dequan(occupancy)
        direction = self.dequan(direction)

        return [classification, offset, occupancy, slot_type, direction]

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, 0)

    def set_qconfig(self):
        self.qconfig = horizon.quantization.get_default_qat_qconfig()

    def fuse_model(self):
        for m in self.modules():
            if isinstance(m, ConvModule2d):
                m.fuse_model()
