# Copyright (c) Horizon Robotics. All rights reserved.

import logging
from collections import OrderedDict, namedtuple

import torch.nn as nn
from horizon_plugin_pytorch import quantization

from hat.registry import OBJECT_REGISTRY
from hat.utils import qconfig_manager

__all__ = ["HppModel"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class HppModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        decode_head: nn.Module,
        post_process: nn.Module = None,
        losses: nn.Module = None,
        decode: nn.Module = None,
        out_indices: int = 2,
    ):
        super(HppModel, self).__init__()
        self.backbone = backbone
        self.decode_head = decode_head
        self.out_indices = out_indices
        self.losses = losses
        self.post_process = post_process
        self.decode = decode

    def forward(self, data: dict):
        image = data["img"]
        features = self.backbone(image)[self.out_indices]
        preds = self.decode_head(features)
        if self.training:
            return self.losses(preds, data)
        else:
            if self.post_process is not None:
                model_result = OrderedDict()
                model_result["probabilities"] = preds["confidences"][-1]
                model_result["offset"] = preds["offsets"][-1]
                model_result.update(self.post_process(model_result))
                HppOutput = namedtuple("HppOutput", model_result.keys())
                model_result = HppOutput(**model_result)
                return model_result
            elif self.decode is not None:
                return self.decode(preds, data)
            else:
                return {"img_name": data["img_name"], "pred": preds}

    def fuse_model(self):
        for module in [self.backbone, self.decode_head, self.losses]:
            if module is not None and hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        self.qconfig = quantization.get_default_qat_qconfig()
        for module in [self.backbone, self.decode_head, self.losses]:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()
                else:
                    module.qconfig = None

    def set_calibration_qconfig(self):
        # from hat.utils import qconfig_manager
        self.calibration_qconfig = (
            qconfig_manager.get_default_calibration_qconfig()
        )
        if self.losses is not None:
            self.losses.qconfig = None
