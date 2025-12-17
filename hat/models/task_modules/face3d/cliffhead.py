# Copyright (c) Horizon Robotics. All rights reserved.
import torch
import torch.nn as nn
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY

__all__ = ["CLIFFHead"]


@OBJECT_REGISTRY.register
class CLIFFHead(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        in_channels: int,
        cat_info_lens: int = 4,
    ):
        """Head based on CLIFF for face3d.

        Args:
            kernel_size : kernel size of average pooling.
            in_channels : channels of each input feature map.
            cat_info_lens: length of cliff infos.
        """

        super(CLIFFHead, self).__init__()
        self.pool = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size))
        cliff_in_channels = in_channels + cat_info_lens
        self.global_pose = ConvModule2d(cliff_in_channels, 3, kernel_size=1)
        self.dequant = DeQuantStub()
        self.jaw_pose = ConvModule2d(cliff_in_channels, 3, kernel_size=1)
        self.camera = ConvModule2d(cliff_in_channels, 3, kernel_size=1)
        self.shape = ConvModule2d(cliff_in_channels, 100, kernel_size=1)
        self.expression = ConvModule2d(cliff_in_channels, 50, kernel_size=1)
        self.texture = ConvModule2d(cliff_in_channels, 50, kernel_size=1)
        self.light = ConvModule2d(cliff_in_channels, 27, kernel_size=1)

    def forward(self, feat, bbox_info):
        feat = self.pool(feat)
        x = torch.cat([feat, bbox_info], 1)
        global_pose = self.global_pose(x)
        jaw_pose = self.jaw_pose(x)
        camera = self.camera(x)
        shape = self.shape(x)
        expression = self.expression(x)
        texture = self.texture(x)
        light = self.light(x)
        # dequant
        global_pose = self.dequant(global_pose)
        jaw_pose = self.dequant(jaw_pose)
        camera = self.dequant(camera)
        shape = self.dequant(shape)
        expression = self.dequant(expression)
        texture = self.dequant(texture)
        light = self.dequant(light)
        return global_pose, jaw_pose, camera, shape, expression, texture, light

    def fuse_model(self):
        modules = [self.pool, self.global_pose]
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
