# Copyright (c) Horizon Robotics. All rights reserved.

import logging
from typing import Optional

import torch.nn as nn
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY

__all__ = ["SmokeClsModel"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class SmokeClsModel(nn.Module):
    """
    The classifier of smoke cls task.

    Args:
        backbone: Backbone module.
            Defaults to None.
        losses: Losses module.
            Defaults to None.
        num_classes: number of classes
            Defaults to 3.
    """

    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        losses: Optional[nn.Module] = None,
        num_classes: int = 3,
    ):

        super(SmokeClsModel, self).__init__()
        self.backbone = backbone
        self.losses = losses
        self.num_classes = num_classes
        self.head = nn.Sequential(
            ConvModule2d(
                in_channels=256,
                out_channels=64,
                kernel_size=2,
                stride=1,
                padding=0,
                act_layer=nn.ReLU(),
                bias=True,
            ),
            ConvModule2d(
                in_channels=64,
                out_channels=self.num_classes,
                kernel_size=3,
                stride=1,
                padding=0,
                norm_layer=None,
                act_layer=None,
            ),
        )
        self.dequant = DeQuantStub()

    def forward(self, data):
        image = data["img"]
        target = data.get("labels", None)

        feat = self.backbone(image)
        if target is None:
            return feat

        preds = self.head(feat[-1]).squeeze()
        preds = self.dequant(preds)

        if not self.training or self.losses is None:
            return preds, target

        losses = self.losses(preds, target)
        return preds, losses

    def fuse_model(self):
        for module in [self.backbone, self.losses]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        if self.backbone is not None and hasattr(self.backbone, "set_qconfig"):
            self.backbone.set_qconfig()

        if self.losses is not None:
            self.losses.qconfig = None

    def set_calibration_qconfig(self):
        from hat.utils import qconfig_manager

        self.calibration_qconfig = (
            qconfig_manager.get_default_calibration_qconfig()
        )
        if self.losses is not None:
            self.losses.qconfig = None
