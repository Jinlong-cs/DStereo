# Copyright (c) Horizon Robotics. All rights reserved.

from collections import OrderedDict
from inspect import signature
from typing import Dict, Optional

import torch.nn as nn

from hat.registry import OBJECT_REGISTRY

__all__ = ["Camera3D", "Camera3DPE"]


@OBJECT_REGISTRY.register
class Camera3D(nn.Module):
    """
    The basic structure of Camera3D task.

    Args:
        backbone: Backbone module.
        neck: Neck module.
        head: Head module.
        desc: Description module.
        losses: Losses module.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        head: nn.Module,
        target: Optional[nn.Module] = None,
        desc: Optional[nn.Module] = None,
        loss: Optional[nn.Module] = None,
        postprocess: Optional[nn.Module] = None,
    ):
        super(Camera3D, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.target = target
        self.desc = desc
        self.loss = loss
        self.postprocess = postprocess

    def forward(self, data: Dict):

        img = data["img"]
        if "uv_map" in signature(self.backbone.forward).parameters:
            features = self.backbone(img, uv_map=data.get("uv_map", None))
        else:
            features = self.backbone(img)
        feat_maps = self.neck(features)
        preds = self.head(feat_maps)

        out_dict = OrderedDict()
        out_dict.update(preds)

        target = self.target(data) if self.target else data

        if self.loss is not None:
            out_dict.update(self.loss(preds, target))
        if self.postprocess is not None:
            predictions = self.postprocess(
                preds, target["calib"], target["distCoeffs"]
            )
            return predictions

        return out_dict

    def fuse_model(self):
        for module in self.children():
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        if self.loss is not None:
            self.loss.qconfig = None

        for module in self.children():
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()


@OBJECT_REGISTRY.register
class Camera3DPE(Camera3D):
    """
    The basic structure of camera3d with position encoding.

    Args:
        backbone (torch.nn.Module): Backbone module.
        neck (torch.nn.Module): FPN neck module.
        neck_ufpn (torch.nn.Module): UFPN neck module.
        head (torch.nn.Module): Head module.
        desc (torch.nn.Module): descriptor module.
        losses (torch.nn.Module): Losses module.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        neck_ufpn: nn.Module,
        head: nn.Module,
        target: nn.Module = None,
        desc: Optional[nn.Module] = None,
        loss: Optional[nn.Module] = None,
        postprocess: Optional[nn.Module] = None,
    ):
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            target=target,
            desc=desc,
            loss=loss,
            postprocess=postprocess,
        )

        self.neck_ufpn = neck_ufpn

    def forward(self, data: dict):
        img = data.pop("img")
        coordinate_map = data.pop("coordinate_map")
        if "uv_map" in signature(self.backbone.forward).parameters:
            features = self.backbone(img, uv_map=data.get("uv_map", None))
        else:
            features = self.backbone(img)
        features = self.neck(features)
        feat_maps = self.neck_ufpn(features, coordinate_map)

        preds = self.head(feat_maps)

        out_dict = OrderedDict()
        out_dict.update(preds)

        target = self.target(data) if self.target else data

        if self.loss is not None:
            out_dict.update(self.loss(preds, target))
        if self.postprocess is not None:
            predictions = self.postprocess(
                preds, target["calib"], target["distCoeffs"]
            )
            return predictions

        return out_dict
