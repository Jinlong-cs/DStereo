# -*- coding:utf-8 -*-
# Copyright (c) Horizon Robotics, All rights reserved.

from collections import OrderedDict
from typing import Dict, Optional

import horizon_plugin_pytorch as horizon
import torch
import torch.nn.functional as F
from horizon_plugin_pytorch.quantization import QuantStub
from torch import nn

from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class H3DStructure(nn.Module):
    """
    The structure for hand pose estimation.

    Args:
        backbone: The backbone module.
        neck: The neck module.
        encoder: The encoder module.
        head: The head module.
        decoder: The decoder module. Defaults to None.
        loss: The loss module. Defaults to None.
        enable_grid_sample: Whether to enable grid sample.
            Defaults to False.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        encoder: nn.Module,
        head: nn.Module,
        decoder: Optional[nn.Module] = None,
        loss: Optional[nn.Module] = None,
        enable_grid_sample: Optional[bool] = False,
    ):

        super(H3DStructure, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.encoder = encoder
        self.head = head
        self.decoder = decoder
        self.loss = loss
        self.enable_grid_sample = enable_grid_sample
        if enable_grid_sample:
            self.quan_img = QuantStub(scale=1 / 128)
            self.quan_grid = QuantStub()
            self.quan_posm = QuantStub()
            self.quantifunc = horizon.nn.quantized.FloatFunctional()

    def forward(self, data: Dict) -> Dict:
        if self.enable_grid_sample:
            img = data.get("img")
            grid = data.get("grid_map")
            position_map = data.get("position_map")
            img = self.quan_img(img)
            grid = self.quan_grid(grid)
            grid = torch.permute(grid, (0, 2, 3, 1))
            position_map = self.quan_posm(position_map)
            grid_img = F.grid_sample(
                img,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            img = self.quantifunc.cat((grid_img, position_map), dim=1)
            data["grid_img"] = img
        else:
            img = data["img"]

        features = self.backbone(img)
        featmaps = self.neck(features)
        encodings = self.encoder(featmaps)
        outputs = self.head(encodings)

        # for deploy
        if self.decoder is None:
            return outputs

        # for val
        out_dict = OrderedDict()
        outputs = self.decoder(outputs, data)
        if self.loss is None:
            out_dict.update(outputs)
            return out_dict

        # for train
        losses = self.loss(outputs, data)
        out_dict.update(losses)
        return out_dict

    def fuse_model(self) -> None:
        for module in [
            self.backbone,
            self.neck,
            self.encoder,
            self.head,
        ]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self) -> None:
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        for module in [
            self.backbone,
            self.neck,
            self.encoder,
            self.head,
        ]:
            if module is not None and hasattr(module, "set_qconfig"):
                module.set_qconfig()

        if self.decoder is not None:
            self.decoder.qconfig = None
        if self.loss is not None:
            self.loss.qconfig = None
