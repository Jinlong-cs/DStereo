import logging
from collections import OrderedDict
from typing import Dict, Optional

import torch.nn as nn

from hat.registry import OBJECT_REGISTRY

__all__ = ["LdmkModel"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class LdmkModel(nn.Module):
    """The basic structure of landmark model.

    Args:
        backbone: backbone module.
        mode: "train", "val" or "deploy" mode. In train mode, return
            loss(OrderedDict), and only "total_loss" should be backward.
            In val mode, return prediction and gt(OrderedDict) for evaluation.
            In deploy mode, return prediction. Defaults to train.
        flatten_output: Which is available only when the mode is "deloy".
            If true, converts the output to a list. Defaults to True.
        decoder: upsampling decoder. Defaults to None.
        vector_head: vector-based landmark detection head, Conv2D +
            BandPoolingModule/BandConvModule + Loss. Defaults to None.
        heatmap_head: heatmap-based landmark detection head, Conv2D + Loss.
            Detaults to None.
        feat_stride: input/output stride, which is only available in
            heatmap/vector based algorithm. Defaults to 4.
        coords_head: coordinates regression head. Defaults to None.
        cls_head: clsssification head. Defaults to None.
        loss_weights: loss weights. Defaults to None.
    """

    def __init__(
        self,
        backbone: nn.Module,
        mode: str = "train",
        flatten_output: bool = True,
        decoder: nn.Module = None,
        vector_head: nn.Module = None,
        heatmap_head: nn.Module = None,
        feat_stride: float = 4,
        coords_head: nn.Module = None,
        cls_head: nn.Module = None,
        loss_weights: Optional[Dict] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.mode = mode.lower()
        self.flatten_output = flatten_output
        assert self.mode in ["train", "val", "deploy"]
        self.decoder = decoder
        self.vector_head = vector_head
        self.heatmap_head = heatmap_head
        self.feat_stride = feat_stride
        self.coords_head = coords_head
        self.cls_head = cls_head
        self.loss_weights = {} if loss_weights is None else loss_weights
        self.nets = [
            self.backbone,
            self.decoder,
            self.vector_head,
            self.heatmap_head,
            self.coords_head,
            self.cls_head,
        ]

    def forward(self, data: dict):
        img = data["img"]
        gt_ldmk = data.get("gt_ldmk", None)
        outputs = OrderedDict()

        feat = self.backbone(img)

        if self.coords_head is not None:
            s32 = feat[-1] if isinstance(feat, list) else feat
            item = {
                "feat": s32,
                "gt_ldmk": gt_ldmk,
                "gt_ldmk_weight": data.get("gt_ldmk_weight", None),
                "loss_weight": self.loss_weights.get("ldmk", 1.0),
            }
            coords_pred = self.coords_head(item)
            outputs.update(coords_pred)

        if self.decoder is not None:
            feat = self.decoder(feat)[0]

        if self.heatmap_head is not None:
            item = {
                "feat": feat,
                "gt_heatmap": data.get("gt_heatmap", None),
                "gt_heatmap_weight": data.get("gt_heatmap_weight", None),
                "loss_weight": self.loss_weights.get("heatmap", 1.0),
            }
            heatmap_pred = self.heatmap_head(item)
            outputs.update(heatmap_pred)
        elif self.vector_head is not None:
            item = {
                "feat": feat,
                "gt_vector_x": data.get("gt_vector_x", None),
                "gt_vector_y": data.get("gt_vector_y", None),
                "gt_vector_weight_x": data.get("gt_vector_weight_x", None),
                "gt_vector_weight_y": data.get("gt_vector_weight_y", None),
                "loss_weight": self.loss_weights.get("vector", 1.0),
            }
            vector_pred = self.vector_head(item)
            outputs.update(vector_pred)

        if self.cls_head is not None:
            item = {}
            cls_pred = self.cls_head(item)
            outputs.update(cls_pred)

        if self.mode == "train":
            total_loss = 0
            for v in outputs.values():
                total_loss += v
            outputs["total_loss"] = total_loss
        elif self.mode == "val":
            outputs["gt_ldmk"] = gt_ldmk
        elif self.mode == "deploy":
            if self.flatten_output:
                outputs = list(outputs.values())
        return outputs

    def fuse_model(self):
        for module in self.nets:
            if module is not None and hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        for module in self.nets:
            if module is not None and hasattr(module, "set_qconfig"):
                module.set_qconfig()
