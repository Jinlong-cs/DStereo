# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from collections import OrderedDict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY

__all__ = ["ImageClassifier"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class ImageClassifier(nn.Module):
    """The structure of image classifier.

    note: it returns a dict when call forward function.

    Args:
        backbone: Backbone module.
        head: Head module.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
    ):
        super(ImageClassifier, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(
        self, data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, torch.Tensor]:
        # assemble model outputs
        out_dict = OrderedDict()
        image = data["img"]
        feats = self.backbone(image)
        out_dict.update(self.head(feats[-1], data))
        return out_dict

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

    def set_calibration_qconfig(self):
        from hat.utils import qconfig_manager

        self.calibration_qconfig = (
            qconfig_manager.get_default_calibration_qconfig()
        )
