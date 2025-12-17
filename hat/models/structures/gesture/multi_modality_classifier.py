# Copyright (c) Horizon Robotics. All rights reserved.

import logging
from typing import Dict, Optional, Tuple

import torch.nn as nn

from hat.registry import OBJECT_REGISTRY

__all__ = ["GestMultiModalityClassifier", "StaMultiModalityClassifier"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class GestMultiModalityClassifier(nn.Module):
    """
    The basic structure of multi modality classifier.

    Support build network for static, static-dyn gesture.

    Args:
        backbone: Backbone module.
        head: Head module.
        losses: Losses module.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        loss: nn.Module,
    ):
        super(GestMultiModalityClassifier, self).__init__()
        self.backbone = backbone
        self.head = head
        self.loss = loss

    def forward(self, data: Dict) -> Tuple:
        feature_kps, feature_frames = self.backbone(data)
        logit_output = self.head(feature_kps, feature_frames)

        target = data.get("act_label", None)
        weight = data.get("label_weight", None)

        if target is None:
            return logit_output

        if self.loss is None:
            return logit_output, target

        losses = self.loss(logit_output, target, weight)
        return logit_output, losses

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


@OBJECT_REGISTRY.register
class StaMultiModalityClassifier(nn.Module):
    """
    The basic structure of multi modality classifier.

    Support build network for static, static-dyn gesture.

    Args:
        kps_encoder: kps encoder module.
        rgb_encoder: rgb encoder module.
        head: Head module.
        loss: loss module.
        add_motion_in_rgb_branch: Whether to add motion information
            to the rgb branch.
        mode: training mode: 'train'; split model: 'rgb' 'kps' 'head';
    """

    def __init__(
        self,
        kps_encoder: nn.Module,
        rgb_encoder: nn.Module,
        head: nn.Module,
        loss: nn.Module,
        add_motion_in_rgb_branch: Optional[bool] = False,
        mode: str = "train",
    ):
        super(StaMultiModalityClassifier, self).__init__()
        self.kps_encoder = kps_encoder
        self.rgb_encoder = rgb_encoder
        self.head = head
        self.loss = loss
        self.add_motion_in_rgb_branch = add_motion_in_rgb_branch
        self.mode = mode

    def forward(self, data: Dict) -> Tuple:
        if self.mode == "train":
            b, t, c, w, h = data["frames"].shape
            data["frames"] = data["frames"].reshape(-1, c, w, h)

        # split model: kps branch
        if self.mode == "kps":
            keypoints = data["clip_keypoints"]
            feature_kps = self.kps_encoder(keypoints)
            logit_output = self.head([feature_kps])

        # split model: rgb branch
        elif self.mode == "rgb":
            frames = data["frames"]
            if self.add_motion_in_rgb_branch:
                motion_input = data["motion_input"]
                feature_rgb = self.rgb_encoder([frames, motion_input])
            else:
                feature_rgb = self.rgb_encoder([frames])
            return feature_rgb

        # split model: head
        elif self.mode == "head":
            head_feature = data["head_feature"]
            logit_output = self.head([head_feature])
            return logit_output

        # for train
        elif self.mode == "train":
            frames = data["frames"]
            keypoints = data["clip_keypoints"]
            feature_kps = self.kps_encoder(keypoints)
            if self.add_motion_in_rgb_branch:
                motion_input = data["motion_input"]
                feature_rgb = self.rgb_encoder([frames, motion_input])
            else:
                feature_rgb = self.rgb_encoder([frames])
            logit_output = self.head([feature_kps, feature_rgb])

        else:
            raise NotImplementedError(self.mode)

        target = data.get("act_label", None)
        weight = data.get("label_weight", None)

        if target is None:
            return logit_output

        if self.loss is None and target is not None:
            return (logit_output, target)

        target = target.squeeze()
        losses = self.loss(logit_output, target, weight)
        return (logit_output, losses)

    def fuse_model(self):
        for module in [self.head, self.kps_encoder, self.rgb_encoder]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        for module in [self.head, self.kps_encoder, self.rgb_encoder]:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()

        if self.loss is not None:
            self.loss.qconfig = None
