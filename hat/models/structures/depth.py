# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List, Optional

import torch.nn as nn
from horizon_plugin_pytorch import quantization

from hat.registry import OBJECT_REGISTRY

__all__ = ["DepthModel"]


@OBJECT_REGISTRY.register
class DepthModel(nn.Module):
    """The basic structure of depth task.

    Args:
        backbone: Backbone module.
        head: Head module.
        head_out_name: The key to get output from previous node outputs.
        neck: Neck module.
        losses: loss module.
        head_preproc_module: Module used to get auxiliary output.
        head_parser: parsing results visualization.
        target_generator: Module to get learning targets.
        decoder: Module to decode model outputs into depth output.
        add_desc: Module to add descs for int_infer.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        head_out_name: str,
        neck: Optional[nn.Module] = None,
        losses: Optional[nn.Module] = None,
        head_preproc_module: Optional[nn.Module] = None,
        head_parser: Optional[nn.Module] = None,
        target_generator: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        add_desc: Optional[nn.Module] = None,
    ):
        super(DepthModel, self).__init__()

        self.backbone = backbone
        self.neck = neck

        self.head_preproc_module = head_preproc_module

        self.head = head
        self.loss = losses
        self.head_parser = head_parser
        self.target_generator = target_generator
        self.decoder = decoder
        self.head_out_name = head_out_name
        self.add_desc = add_desc

    def forward(self, data: dict):
        image = data["img"]
        if "gt_depth" in data.keys():
            target = data["gt_depth"]
            if target.dim() == 4:
                target = target.squeeze(dim=-1)
            target = target.unsqueeze(dim=1)
            data["gt_depth"] = target
        else:
            target = None

        features = self.backbone(image)
        if self.neck:
            features = self.neck(features)

        # TODO(yunfeng.zhang): Improve implementation for head auxiliary input
        if self.head_preproc_module:
            head_auxi_input = self.head_preproc_module(features, data)
            preds = self.head(features, head_auxi_input)
        else:
            preds = self.head(features)

        # add desc in int_infer
        if self.add_desc:
            preds = self.add_desc(preds)

        if target is None:
            pred = preds[self.head_out_name]
            if isinstance(pred, tuple) or isinstance(pred, List):
                pred = pred[0]
            return pred

        if self.head_parser:
            preds = self.head_parser(preds)
        if self.target_generator:
            target = self.target_generator(data, preds)

        if self.loss:
            loss = self.loss(preds, target)

        if self.decoder:
            decode = self.decoder(preds, data)
            return decode

        # only the largest feature map
        pred = preds[self.head_out_name]
        if isinstance(pred, tuple) or isinstance(pred, List):
            pred = pred[0]

        res_model = {}
        res_model["depth_preds"] = pred
        res_model["depth_loss"] = loss
        res_model["gt_depth"] = data["gt_depth"]
        if "origin_img" in data:
            ori_img = data["origin_img"]
            res_model["ori_img"] = ori_img.permute((0, 2, 3, 1))
        return res_model

    def fuse_model(
        self,
    ):
        for module in [
            self.backbone,
            self.neck,
            self.head,
        ]:
            if module is not None and hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        self.qconfig = quantization.get_default_qat_qconfig()

        for module in [
            self.backbone,
            self.neck,
            self.head,
        ]:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()
                else:
                    module.qconfig = None
