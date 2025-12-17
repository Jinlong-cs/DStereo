import logging
from collections import OrderedDict
from typing import Optional

import torch.nn as nn

from hat.registry import OBJECT_REGISTRY

__all__ = ["TollGageModel"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class TollGageModel(nn.Module):
    """The basic structure of toll gage model.

    Args:
        backbone: backbone module.
        neck: Neck module.
        head: Head module.
        desc: Directory description.
        losses: Loss module for TollGageModel.
        postprocess: Postprocess module.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module = None,
        head: nn.Module = None,
        loss: Optional[nn.Module] = None,
        desc: Optional[nn.Module] = None,
        postprocess: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.loss = loss
        self.desc = desc
        self.postprocess = postprocess

    def forward(self, data: dict):
        img = data["img"]
        item = {
            "gt_heatmap": data.get("gt_heatmap", None),
            "gt_heatmap_weight": data.get("gt_heatmap_weight", None),
            "gt_offset": data.get("gt_offset", None),
            "gt_offset_weight": data.get("gt_offset_weight", None),
        }
        feat = self.backbone(img)
        if self.neck is not None:
            feat = self.neck(feat)
        if self.head is not None:
            pred = self.head(feat)
        if self.desc is not None:
            pred = self.desc(pred)
        output = OrderedDict(pred=pred)

        if self.loss is not None:
            feat = self.loss(pred, item)
            output.update(feat)

        if self.postprocess is not None:
            if self.desc is not None:
                return self.desc(self.postprocess(pred))
            else:
                return self.postprocess(pred)
        return output

    def fuse_model(self):
        for module in [self.backbone, self.neck, self.head]:
            if module is not None and hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        for module in [self.backbone, self.neck, self.head]:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()

        if self.loss is not None:
            self.loss.qconfig = None
        if self.postprocess is not None:
            self.postprocess.qconfig = None
