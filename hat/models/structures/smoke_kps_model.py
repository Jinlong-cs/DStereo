# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY

__all__ = ["SmokeKpsModel"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class SmokeKpsModel(nn.Module):
    """The basic structure of smoke kps model.

    Args:
        backbone: backbone module.
        mode: "train", "val" or "deploy" mode. In train mode, return
            loss(OrderedDict), and only "total_loss" should be backward.
            In val mode, return prediction and gt(OrderedDict) for evaluation.
            In deploy mode, return prediction(List).
        decoder: upsampling decoder.
        vector_head: vector-based smoke keypoints detection head, Conv2D +
            BandPoolingModule/BandConvModule + Loss.
        heatmap_head: heatmap-based smoke keypoints detection head,
            Conv2D + Loss.
        feat_stride: input/output stride, which is only available in
            heatmap/vector based algorithm.
        cls_head: clsssification head for cigaret visable and smoke classes.
        loss_weights: loss weights.
    """

    def __init__(
        self,
        backbone: nn.Module,
        mode: str = "train",
        decoder: nn.Module = None,
        vector_head: nn.Module = None,
        heatmap_head: nn.Module = None,
        feat_stride: float = 4,
        cls_head: nn.Module = None,
        loss_weights: Optional[Dict] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.mode = mode.lower()
        assert self.mode in {"train", "val", "deploy"}
        self.decoder = decoder
        self.vector_head = vector_head
        self.heatmap_head = heatmap_head
        self.feat_stride = feat_stride
        self.cls_head = cls_head
        self.loss_weights = {} if loss_weights is None else loss_weights
        self.nets = [
            self.backbone,
            self.decoder,
            self.vector_head,
            self.heatmap_head,
            self.cls_head,
        ]

    def forward(self, data: dict):
        img = data["img"]
        gt_ldmk = data.get("gt_ldmk")
        gt_visable = data.get("gt_visable")
        gt_classes = data.get("gt_classes")
        outputs = OrderedDict()

        # backbone and decoder, feat: [B, 128, 32, 32]
        s2, s4, s8, s16, s32 = self.backbone(img)
        feat = (
            self.decoder([s2, s4, s8, s16, s32])[0]
            if self.decoder is not None
            else s4
        )

        # head: heatmap_head, vector_head or cls_head
        if self.heatmap_head is not None:
            item = {
                "feat": feat,
                "gt_heatmap": data.get("gt_heatmap", None),
                "gt_heatmap_weight": data.get("gt_heatmap_weight", None),
                "loss_weight": self.loss_weights.get("heatmap", 1.0),
            }
            heatmap_pred = self.heatmap_head(item)
            outputs.update(heatmap_pred)

        if self.vector_head is not None:
            item = {
                "feat": feat,
                "gt_vector_x": data.get("gt_vector_x", None),
                "gt_vector_y": data.get("gt_vector_y", None),
                "gt_vector_weight_x": data.get("gt_vector_weight_x", None),
                "gt_vector_weight_y": data.get("gt_vector_weight_y", None),
                "loss_weight": self.loss_weights.get("vecotr", 1.0),
            }
            vector_pred = self.vector_head(item)
            outputs.update(vector_pred)

        if self.cls_head is not None:
            item = {
                "feat": s32,
                "gt_visable": gt_visable,
                "gt_vis_weight": data.get(
                    "gt_vis_weight", torch.ones_like(gt_visable)
                ),
                "gt_classes": gt_classes,
                "gt_cls_weight": data.get(
                    "gt_cls_weight", torch.ones_like(gt_classes)
                ),
                "vis_loss_weight": self.loss_weights.get("vis", 1.0),
                "cls_loss_weight": self.loss_weights.get("cls", 1.0),
            }
            cls_pred = self.cls_head(item)  # [4, 256, 4, 4]
            outputs.update(cls_pred)

        # return outputs
        if self.mode == "train":
            total_loss = sum(outputs.values())
            outputs["total_loss"] = total_loss
        elif self.mode == "val":
            outputs["gt_ldmk"] = gt_ldmk
            outputs["gt_ldmk_attr"] = data.get("gt_ldmk_attr")
            outputs["gt_classes"] = gt_classes
            outputs["gt_visable"] = gt_visable
        elif self.mode == "deploy":
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
