# Copyright (c) Horizon Robotics. All rights reserved.

import logging
from typing import Dict, Tuple

from torch.nn import Module

from hat.models.losses.utils import weight_reduce_loss
from hat.registry import OBJECT_REGISTRY

__all__ = ["DynGestureClassifier"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class DynGestureClassifier(Module):
    """
    The basic structure of dyn gesture classifier.

    Args:
        backbone: Backbone module.
        head: Head module.
        loss: Losses module.
    """

    def __init__(self, backbone: Module, head: Module, loss: Module = None):
        super(DynGestureClassifier, self).__init__()
        self.backbone = backbone
        self.head = head
        self.loss = loss

    def forward(self, data: Dict) -> Tuple:
        keypoints = data["clip_keypoints"]
        target = data.get("act_label", None)
        weight = data.get("label_weight", None)

        feature = self.backbone(keypoints)[-1]
        preds = self.head(feature)
        if target is None:
            return preds

        if not self.training or self.loss is None:
            return preds, target

        losses = self.loss(preds, target.squeeze())
        # weight
        losses = weight_reduce_loss(
            loss=losses, weight=weight, reduction="mean"
        )
        return preds, losses

    def fuse_model(self):
        for module in [self.backbone, self.head]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        for module in [self.backbone, self.head]:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()

        if self.loss is not None:
            self.loss.qconfig = None
