# Copyright (c) Horizon Robotics. All rights reserved.

import logging

import torch.nn as nn
from horizon_plugin_pytorch import quantization

from hat.registry import OBJECT_REGISTRY

__all__ = ["SuperPSDDetector"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class SuperPSDDetector(nn.Module):
    """
    The basic structure of SuperPSDDetector.

    Args:
        backbone (torch.nn.Module): Backbone module.
        neck (torch.nn.Module): Neck module.
        global_head (torch.nn.Module): Global Head module.
        local_head (torch.nn.Module): Local Head module.
        global_decoder (torch.nn.Module): Global Decoder module.
        local_decoder (torch.nn.Module): Local Decoder module.
        losses (torch.nn.Module): Losses module.
    """

    def __init__(self, backbone, neck, head=None, target=None, losses=None):
        super(SuperPSDDetector, self).__init__()
        self.backbone = None
        self.neck = None
        self.head = None
        self.target = None
        self.losses = None

        self.backbone = backbone
        self.neck = neck
        if head is not None:
            self.head = head
        if target is not None:
            self.target = target
        if losses is not None:
            self.losses = losses

    def forward(self, data: dict):
        preds, targets, grads = [], [], []
        img = data.get("img", None)
        label = data.get("label", None)
        features = self.backbone(img)
        features = self.neck(features)
        if self.head is not None:
            total_preds = self.head(features)
            preds.extend(total_preds)
        if label is None:
            return preds
        if not self.training or self.losses is None:
            data["preds"] = preds
            return data
        result_dict = self.target(label, preds)
        targets, grads = (
            result_dict[0]["targets"],
            result_dict[0]["grads"],
        )
        losses = self.losses(preds, targets, grads)
        data["losses"] = losses
        return data

    def fuse_model(self):
        for module in [self.backbone, self.neck, self.head, self.losses]:
            if module is not None and hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        self.qconfig = quantization.get_default_qat_qconfig()

        for module in [self.backbone, self.neck, self.head, self.losses]:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()
                else:
                    module.qconfig = None
