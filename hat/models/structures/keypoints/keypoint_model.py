from typing import Dict

import torch
import torch.nn as nn
from torch.quantization import DeQuantStub

from hat.models.base_modules.basic_hbir_module import BaseHbirModule
from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

__all__ = ["HeatmapKeypointModel", "HeatmapKeypointHbirInfer"]


@OBJECT_REGISTRY.register
class HeatmapKeypointModel(nn.Module):
    """
    HeatmapKeypointModel is a model for keypoint detection using heatmaps.

    Args:
        backbone: Backbone network used for feature extraction.
        decode_head: Decode head that upsample the feature to generate heatmap.
        loss: Loss function that compute the loss
        post_processes: Module that decode keypoints prediction from heatmap.
        deploy: Flag indicating whether the model is used for deployment
                or training.
    """

    def __init__(
        self,
        backbone: nn.Module,
        decode_head: nn.Module,
        loss: nn.Module = None,
        post_process: nn.Module = None,
        deploy: bool = False,
    ):
        super(HeatmapKeypointModel, self).__init__()
        self.deploy = deploy

        self.loss = loss
        self.post_process = post_process

        self.backbone = backbone
        self.decode_head = decode_head
        self.dequant = DeQuantStub()

    @fx_wrap()
    def _post_process(self, preds: torch.Tensor, data: Dict):
        target = data.get("gt_heatmap")
        if self.deploy or self.loss is None or target is None:  # Infer
            if self.post_process is None:
                return (preds,)
            else:
                results = self.post_process(preds)
                return (preds, results)

        B, C, H, W = preds.shape
        weights = data.get("gt_heatmap_weight")
        valid_mask = (
            (data.get("gt_ldmk_attr") > 0).float().reshape([B, C, 1, 1])
        )
        loss = self.loss(
            preds,
            target,
            weight=weights,
            avg_factor=preds.shape[0],
            valid_mask=valid_mask,
        )
        return preds, loss

    def forward(self, data: Dict):
        x = data["img"]
        x = self.backbone(x)
        x = self.decode_head(x)
        out = self.dequant(x)
        out = self._post_process(out, data)
        return out

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        for module in self.children():
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()


@OBJECT_REGISTRY.register
class HeatmapKeypointHbirInfer(BaseHbirModule):
    """
    The basic structure of HeatmapKeypointHbirInfer.

    Args:
        model_path: The path of hbir model.
        post_process: Postprocess module.
    """

    def __init__(
        self,
        model_path: str,
        post_process: nn.Module,
    ):
        super().__init__(model_path=model_path)
        self.post_process = post_process

    def _post_process(self, preds: torch.Tensor, data: Dict):
        results = self.post_process(preds)
        return (preds, results)

    def forward(self, data):

        image = data["img"]
        outputs = super().forward(image)
        results = self._post_process(outputs[0], data)
        return results
