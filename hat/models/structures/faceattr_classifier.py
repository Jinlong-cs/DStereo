# Copyright (c) Horizon Robotics. All rights reserved.

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.quantization import DeQuantStub

from hat.registry import OBJECT_REGISTRY

__all__ = ["FaceAttrClassifier"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class FaceAttrClassifier(nn.Module):
    """The faceattr structure of classifier.

    Only attr-heads update when training.

    Args:
        backbone: Faceid backbone module.
        head: Face attr age-gender heads module.
        loss_age: Losses module for age. Defaults to None.
        loss_gender: Losses module for gender. Defaults to None.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        loss_age: Optional[nn.Module] = None,
        loss_gender: Optional[nn.Module] = None,
        deploy: bool = False,
    ):

        super(FaceAttrClassifier, self).__init__()
        self.backbone = backbone
        self.head = head
        self.loss_age = loss_age
        self.loss_gender = loss_gender
        self.deploy = deploy

        self.dequant = DeQuantStub()

    def forward(self, data):
        image = data["img"]

        with torch.no_grad():
            feat_outputs, pred_id = self.backbone(image)
            feat = feat_outputs[-1]
            outputs = {
                "pred_faceid": pred_id,
            }

        pred_age, pred_gender = self.head(feat)
        outputs["pred_age"] = pred_age
        outputs["pred_gender"] = pred_gender

        if self.loss_gender is None or self.loss_age is None or self.deploy:
            return outputs

        gt_age = data["age"]
        gt_age_ord = data["ord_age"]
        gt_gender = data["gender"]
        outputs["age"] = gt_age
        outputs["gender"] = gt_gender

        loss_age = self.loss_age(pred_age, gt_age, gt_age_ord)
        loss_gender = self.loss_gender(pred_gender, gt_gender, gt_gender != -1)
        losses = loss_age + loss_gender
        outputs["losses"] = losses
        outputs["loss_age"] = loss_age
        outputs["loss_gender"] = loss_gender
        return outputs

    def fuse_model(self):
        for module in [
            self.backbone,
            self.head,
            self.loss_age,
            self.loss_gender,
        ]:
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

        if self.loss_age is not None:
            self.loss_age.qconfig = None

        if self.loss_gender is not None:
            self.loss_gender.qconfig = None

    def set_calibration_qconfig(self):
        from hat.utils import qconfig_manager

        self.calibration_qconfig = (
            qconfig_manager.get_default_calibration_qconfig()
        )
        if self.loss_age is not None:
            self.loss_age.qconfig = None
        if self.loss_gender is not None:
            self.loss_gender.qconfig = None
