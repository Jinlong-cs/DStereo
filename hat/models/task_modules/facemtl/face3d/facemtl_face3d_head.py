# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Dict, Optional

import torch.nn as nn
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY

__all__ = ["FaceMtlFace3dHead"]


@OBJECT_REGISTRY.register
class FaceMtlFace3dHead(nn.Module):
    """Face3d head for facemtl.

    Args:
        input_channels: Channels of each input feature map.
        out_channels: Channels for the module.
        only_global_pose: Whether to return only global_pose. Default: False.
        group_base: Group base for FaceMtlFace3dHead.
        use_group: Whether use group conv.
        bn_kwargs: Extra keyword arguments for bn layers. Default: None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        only_global_pose: bool = False,
        group_base: int = 16,
        use_group: bool = True,
        bn_kwargs: Optional[Dict] = None,
    ):
        super(FaceMtlFace3dHead, self).__init__()
        self.only_global_pose = only_global_pose
        if bn_kwargs is None:
            bn_kwargs = {}

        # multi convs head
        self.features = nn.Sequential(
            ConvModule2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=int(in_channels / group_base) if use_group else 1,
                norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
                act_layer=nn.ReLU(inplace=True),
            ),
            ConvModule2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,  # input: 160*160
                # kernel_size=2,   # input: 128*128
                stride=1,
                padding=0,
                groups=int(out_channels / group_base) if use_group else 1,
                norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
                act_layer=nn.ReLU(inplace=True),
            ),
        )

        self.global_pose = ConvModule2d(out_channels, 3, kernel_size=1)
        self.dequant = DeQuantStub()
        if not self.only_global_pose:
            # multi convs head
            self.jaw_pose = ConvModule2d(out_channels, 3, kernel_size=1)
            self.camera = ConvModule2d(out_channels, 3, kernel_size=1)
            self.shape = ConvModule2d(out_channels, 100, kernel_size=1)
            self.expression = ConvModule2d(out_channels, 50, kernel_size=1)
            self.texture = ConvModule2d(out_channels, 50, kernel_size=1)
            self.light = ConvModule2d(out_channels, 27, kernel_size=1)

    def forward(self, x):
        # multi convs head
        x = self.features(x)

        global_pose = self.global_pose(x)
        global_pose = self.dequant(global_pose)
        if self.only_global_pose:
            return global_pose, None, None, None, None, None, None
        jaw_pose = self.jaw_pose(x)
        camera = self.camera(x)
        shape = self.shape(x)
        expression = self.expression(x)
        texture = self.texture(x)
        light = self.light(x)
        # dequant
        jaw_pose = self.dequant(jaw_pose)
        camera = self.dequant(camera)
        shape = self.dequant(shape)
        expression = self.dequant(expression)
        texture = self.dequant(texture)
        light = self.dequant(light)
        return global_pose, jaw_pose, camera, shape, expression, texture, light

    def fuse_model(self):
        modules = [self.features, self.global_pose]
        if not self.only_global_pose:
            modules += [
                self.jaw_pose,
                self.camera,
                self.shape,
                self.expression,
                self.texture,
                self.light,
            ]
        for module in modules:
            for m in module:
                if hasattr(m, "fuse_model"):
                    m.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        self.global_pose.qconfig = (
            qconfig_manager.get_default_qat_out_qconfig()
        )
        if not self.only_global_pose:
            self.camera.qconfig = qconfig_manager.get_default_qat_out_qconfig()
            self.camera.qconfig = qconfig_manager.get_default_qat_out_qconfig()
            self.jaw_pose.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
            self.shape.qconfig = qconfig_manager.get_default_qat_out_qconfig()
            self.expression.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
            self.texture.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
            self.light.qconfig = qconfig_manager.get_default_qat_out_qconfig()
