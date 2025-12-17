from inspect import signature
from typing import Dict

import torch.nn as nn

from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

__all__ = ["IspMeDetector", "AutoAssignFCOS", "LutMeDetector"]


@OBJECT_REGISTRY.register
class IspMeDetector(nn.Module):
    """
    The basic structure of isp-me-detector.

    Args:
        me (nn.Module): The model of ISP ME.
        detector (nn.Module): The model of detector.
    """

    def __init__(
        self,
        me: nn.Module,
        detector: nn.Module,
    ):
        super(IspMeDetector, self).__init__()
        self.me = me
        self.detector = detector

    def forward(self, data: Dict):
        me_out = self.me(data)
        data["img"] = me_out
        return self.detector(data)


@OBJECT_REGISTRY.register
class LutMeDetector(nn.Module):
    """
    The basic structure of lut-me-detector.

    Args:
        lutnet (nn.Module): The model of LutNet.
        me (nn.Module): The model of ISP ME.
        detector (nn.Module): The model of detector.
        deploy (bool): Whether the model is used in deploy mode.
    """

    def __init__(
        self,
        lutnet: nn.Module,
        me: nn.Module,
        detector: nn.Module,
        deploy: bool = False,
    ) -> None:
        super(LutMeDetector, self).__init__()
        self.lut_net = lutnet
        self.me = me
        self.detector = detector
        self.deploy = deploy

    def forward(self, data: Dict):
        if not self.deploy and self.lut_net is not None:
            imgs = data["img"]
            imgs = self.lut_net(imgs)
            data["img"] = imgs
        if self.me is not None:
            imgs = self.me(data)
            data["img"] = imgs
        return self.detector(data)


@OBJECT_REGISTRY.register
class AutoAssignFCOS(nn.Module):
    """
    The basic structure of auto-assign-fcos.

    Args:
        backbone (nn.Module): The model of backbone.
        neck (nn.Module): The model of neck.
        head (nn.Module): The model of head.
        targets (nn.Module): The model of targets.
        desc (nn.Module): The model of desc.
        post_process (nn.Module): The model of post_process.
        loss (nn.Module): The model of loss.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module = None,
        head: nn.Module = None,
        targets: nn.Module = None,
        desc: nn.Module = None,
        post_process: nn.Module = None,
        loss: nn.Module = None,
    ):
        super(AutoAssignFCOS, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

        self.loss = loss
        self.desc = desc

        self.targets = targets
        self.post_process = post_process

    def extract_feat(self, img, uv_map=None):
        """Directly extract features from the backbone + neck."""
        if "uv_map" in signature(self.backbone.forward).parameters:
            x = self.backbone(img, uv_map)
        else:
            x = self.backbone(img)
        if self.neck is not None:
            x = self.neck(x)
        return x

    @fx_wrap()
    def _post_process(self, data, preds):
        if self.training and self.targets is not None:
            targets = self.targets(data, preds)
            losses = self.loss(preds, targets)
            return dict(**losses)
        else:
            if self.post_process is None:
                return preds
            results = self.post_process(preds, data)
            return results

    def forward(self, data: Dict):
        imgs = data["img"]
        feats = self.extract_feat(imgs, data.get("uv_map", None))
        preds = self.head(feats)
        if self.desc is not None:
            preds = self.desc(preds)
        return self._post_process(data, preds)

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
