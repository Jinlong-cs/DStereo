# Copyright (c) Horizon Robotics. All rights reserved.

import logging
from typing import Optional

import torch.nn as nn

from hat.registry import OBJECT_REGISTRY

__all__ = ["FasClassifier"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class FasClassifier(nn.Module):
    """The basic structure of face-anti-spoof classifier.

    Args:
        backbone: Backbone module.
        head: Head module.
        loss: Loss module. Defaults to None.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        loss: Optional[nn.Module] = None,
    ):

        super(FasClassifier, self).__init__()
        self.backbone = backbone
        self.head = head
        self.loss = loss

    def forward(self, data):
        image = data["img"]
        feats = self.backbone(image)
        preds = self.head(feats[-1])

        if not self.training or self.loss is None:
            return preds

        else:
            fas_label = data["fas_label"]
            car_cls = data["database_labels"]
            loss = self.loss(preds, fas_label, car_cls)
            return preds, loss

    def fuse_model(self):
        for module in [self.backbone, self.head, self.loss]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        if self.backbone is not None:
            if hasattr(self.backbone, "set_qconfig"):
                self.backbone.set_qconfig()

        if self.head is not None:
            if hasattr(self.head, "set_qconfig"):
                self.head.set_qconfig()

        if self.loss is not None:
            self.loss.qconfig = None

    def set_calibration_qconfig(self):
        from hat.utils import qconfig_manager

        self.calibration_qconfig = (
            qconfig_manager.get_default_calibration_qconfig()
        )

        if self.loss is not None:
            self.loss.qconfig = None
