# Copyright (c) Horizon Robotics. All rights reserved.
import horizon_plugin_pytorch.nn.quantized as quantized
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.quantization import DeQuantStub

from hat.registry import OBJECT_REGISTRY

__all__ = ["TollPostProcess"]


@OBJECT_REGISTRY.register
class TollPostProcess(nn.Module):
    """The postprocess of Transformer for Toll Gate task.

    Args:
        None
    """

    def __init__(
        self,
        local_max_kernel,
    ):
        super(TollPostProcess, self).__init__()
        self.local_max_kernel = local_max_kernel
        self.qmax_pool = torch.nn.MaxPool2d(
            self.local_max_kernel,
            stride=1,
            padding=(self.local_max_kernel - 1) // 2,
        )
        self.dequant = DeQuantStub()
        self.mul = quantized.FloatFunctional()

    @autocast(enabled=False)
    def forward(self, preds_dicts):
        hm = preds_dicts["hm_temp"]
        maxp = self.qmax_pool(hm)
        maxp = torch.eq(hm, maxp)
        feature = self.mul.mul(hm, maxp)
        feature = self.dequant(feature)
        results = {}
        results["lmks_heatmap"] = feature
        results["lmks_offset"] = preds_dicts["offset"]
        return results
