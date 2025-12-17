# Copyright (c) Horizon Robotics. All rights reserved.

import logging
from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from horizon_plugin_pytorch import quantization

from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class OneStageDetector(nn.Module):
    """The one stage detector structure.

    Args:
        backbone: backbone network.
        neck: neck network.
        rpn_module: region proposal network module.
        rpn_out_keys: keys of rpn output that should be included in final
            outputs. By default, all rpn output are included.
        output_feat: Whether to include extracted feature maps in final outpus.
    """

    def __init__(
        self,
        input_preprocess: nn.Module,
        backbone: nn.Module,
        rpn_module: nn.Module,
        neck: Optional[nn.Module] = None,
        rpn_out_keys: Optional[List[str]] = None,
        output_feat: bool = False,
        mask_in_bpu: bool = False,
        track_feature: bool = False,
        track_desc: nn.Module = None,
    ):
        super().__init__()

        assert rpn_module is not None

        self.input_preprocess = input_preprocess
        self.backbone = backbone
        self.neck = neck
        self.rpn_module = rpn_module
        self._rpn_out_keys = rpn_out_keys
        self._output_feat = output_feat
        self.mask_in_bpu = mask_in_bpu
        self.track_feature = track_feature  # 对应backbone第5个stage的feature
        self.track_desc = track_desc

    @property
    def with_neck(self):
        return self.neck is not None

    def _extract_feature(self, x: torch.Tensor) -> List[torch.Tensor]:
        feat_maps = self.backbone(x)
        track_feature = feat_maps[-1] if self.track_feature else None
        if hasattr(self, "neck"):
            feat_maps = self.neck(feat_maps)

        return feat_maps, track_feature

    def forward(
        self, data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        # feature extraction
        x = data["img"]
        if self.mask_in_bpu:
            assert (
                "mask_width" in data.keys()
            ), "when do mask op in bpu, crop_mask must in input"
            height_crop_mask = data["mask_height"]
            width_crop_mask = data["mask_width"]
            # crop_mask = [height_crop_mask, width_crop_mask]
            assert self.input_preprocess is not None
            x = self.input_preprocess(x, height_crop_mask, width_crop_mask)

        feat_maps, track_feature = self._extract_feature(x)

        # assemble model outputs
        out_dict = OrderedDict()
        if self.track_feature:
            out_dict.update(
                dict(  # noqa[C408]
                    track_feature=self.track_desc(track_feature)
                )
            )

        if self._output_feat:
            out_dict["feature_maps"] = feat_maps

        rpn_out = self.rpn_module(feat_maps, y=data)
        assert not set(rpn_out.keys()).intersection(set(out_dict.keys()))
        if self._rpn_out_keys is not None:
            out_dict.update({k: rpn_out[k] for k in self._rpn_out_keys})
        else:
            out_dict.update(rpn_out)

        return out_dict

    def fuse_model(self):
        for module in self.children():
            module.fuse_model()

    def set_qconfig(self):
        self.qconfig = quantization.get_default_qat_qconfig()

        for module in self.children():
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()
                else:
                    module.qconfig = None
