# Copyright (c) Horizon Robotics. All rights reserved.
import torch.nn as nn
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY

__all__ = ["Face3dHead"]


@OBJECT_REGISTRY.register
class Face3dHead(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        in_channels: int,
        only_global_pose: bool = False,
    ):
        """Multi Head for face3d.

        Only head pose branch is active during inference.

        Args:
            kernel_size : kernel size of average pooling.
            in_channels : channels of each input feature map.
        """

        super(Face3dHead, self).__init__()
        self.only_global_pose = only_global_pose
        self.pool = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size))
        self.global_pose = ConvModule2d(in_channels, 3, kernel_size=1)
        self.dequant = DeQuantStub()
        if not self.only_global_pose:
            self.jaw_pose = ConvModule2d(in_channels, 3, kernel_size=1)
            self.camera = ConvModule2d(in_channels, 3, kernel_size=1)
            self.shape = ConvModule2d(in_channels, 100, kernel_size=1)
            self.expression = ConvModule2d(in_channels, 50, kernel_size=1)
            self.texture = ConvModule2d(in_channels, 50, kernel_size=1)
            self.light = ConvModule2d(in_channels, 27, kernel_size=1)

    def forward(self, x):
        x = self.pool(x)
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
