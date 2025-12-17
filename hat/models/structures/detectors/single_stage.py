# Copyright (c) Horizon Robotics. All rights reserved.

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class SingleStageDetector(nn.Module):
    """Base class for single-stage detectors.

        Single-stage detectors directly and densely predict bounding boxes
        on the output features of the backbone+neck.

    Args:
        backbone: backbone network.
        neck: neck network.
        box_module: construct box module in a network with head, target, loss.

    """

    def __init__(
        self,
        backbone: nn.Module,
        box_module: Optional[nn.Module],
        neck: Optional[nn.Module] = None,
    ):
        super(SingleStageDetector, self).__init__()
        assert box_module is not None, "`box_module` can not be None"

        self.backbone = backbone
        self.neck = neck
        self.box_module = box_module

    @property
    def with_neck(self):
        return self.neck is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(
        self, data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        imgs = data.pop("img")
        feats = self.extract_feat(imgs)
        outputs = self.box_module(feats, data)
        return outputs

    def fuse_model(self):
        for module in self.children():
            module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        for module in self.children():
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()
