# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from typing import Dict

from torch import nn

from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class BMFCOS(nn.Module):
    """The basic structure of fcos used for the cloudmodel project.

    This detector structure can work with MultitaskGraphModel when
    lazy_forward=True is set.

    Args:
        backbone: Backbone module.
        neck: Neck module.
        head: Head module.
        target: Target module.
        loss: Loss module.
        postprocess: Postprocess module.

    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module = None,
        head: nn.Module = None,
        target: nn.Module = None,
        loss: nn.Module = None,
        postprocess: nn.Module = None,
    ):
        super(BMFCOS, self).__init__()

        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.target = target
        self.loss = loss
        self.postprocess = postprocess
        self.qconfig = None

    @property
    def has_target(self) -> bool:
        return self.target is not None

    @property
    def has_loss(self) -> bool:
        return self.loss is not None

    @property
    def has_postprocess(self) -> bool:
        return self.postprocess is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone + neck."""
        x = self.backbone(img)
        if self.neck is not None:
            x = self.neck(x)
        return x

    def forward(self, data: Dict, **kwargs):
        imgs = data["img"]
        feats = self.extract_feat(imgs)
        preds = self.head(feats)

        if self.has_target:
            targets = self.target(data, preds)
        else:
            targets = None

        if self.has_loss:
            losses = self.loss(preds, targets)
            return losses

        if self.has_postprocess:
            preds = self.postprocess(preds, data)
            return preds

    def fuse_model(self):
        for module in [self.backbone, self.neck, self.head]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        for module in [self.backbone, self.neck, self.head]:
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()
        if self.has_target:
            self.target.qconfig = None
        if self.has_loss:
            self.loss.qconfig = None
        if self.has_postprocess:
            self.postprocess.qconfig = None
