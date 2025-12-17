# Copyright (c) Horizon Robotics. All rights reserved.

import logging
from collections import OrderedDict
from inspect import signature
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class TwoStageDetector(nn.Module):
    """The two stage detector structure.

    Args:
        backbone: backbone network.
        neck: neck network.
        rpn_module: region proposal network module.
        roi_module: roi module.
        rpn_out_keys: keys of rpn output that should be included in final
            outputs. By default, all rpn output are included.
        output_feat: Whether to include extracted feature maps in final outpus.
        remap_label_key: change rpn module gt label key to "label" if set True.
    """

    def __init__(
        self,
        backbone: nn.Module,
        rpn_module: nn.Module,
        roi_module: nn.Module,
        neck: Optional[nn.Module] = None,
        rpn_out_keys: Optional[List[str]] = None,
        output_feat: bool = False,
        remap_label_key: bool = False,
    ):
        super().__init__()

        assert rpn_module is not None or roi_module is not None

        self.backbone = backbone
        self.neck = neck
        self.rpn_module = rpn_module
        self.roi_module = roi_module
        self._rpn_out_keys = rpn_out_keys
        self._output_feat = output_feat
        self.remap_label_key = remap_label_key

    @property
    def with_neck(self):
        return self.neck is not None

    @property
    def with_roi(self):
        return self.roi_module is not None

    def _extract_feature(
        self, x: torch.Tensor, uv_map: torch.Tensor = None
    ) -> List[torch.Tensor]:
        if "uv_map" in signature(self.backbone.forward).parameters:
            feat_maps = self.backbone(x, uv_map=uv_map)
        else:
            feat_maps = self.backbone(x)
        if hasattr(self, "neck"):
            feat_maps = self.neck(feat_maps)

        return feat_maps

    def forward(
        self, data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        # feature extraction
        feat_maps = self._extract_feature(
            data["img"], data.get("uv_map", None)
        )

        # assemble model outputs
        out_dict = OrderedDict()

        if self._output_feat:
            out_dict["feature_maps"] = feat_maps

        if self.remap_label_key:
            rpn_out = self.rpn_module(feat_maps, label=data)
        else:
            rpn_out = self.rpn_module(feat_maps, y=data)
        assert not set(rpn_out.keys()).intersection(set(out_dict.keys()))
        if self._rpn_out_keys is not None:
            out_dict.update({k: rpn_out[k] for k in self._rpn_out_keys})
        else:
            out_dict.update(rpn_out)

        roi_out = self.roi_module(feat_maps, rpn_out, y=data)
        assert not set(roi_out.keys()).intersection(set(out_dict.keys()))
        out_dict.update(roi_out)

        return out_dict

    def fuse_model(self):
        for module in self.children():
            module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        for module in self.children():
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()


@OBJECT_REGISTRY.register
class TwoStageDetectorPE(TwoStageDetector):
    """The two stage detector with position encoding structure.

    Args:
        backbone: backbone network.
        neck: neck network.
        rpn_module: region proposal network module.
        roi_module: roi module.
        rpn_out_keys: keys of rpn output that should be included in final
            outputs. By default, all rpn output are included.
        output_feat: Whether to include extracted feature maps in final outpus.
        pe_desc:  the __desc__ descriptor for position encoding.
    """

    def __init__(
        self,
        backbone: nn.Module,
        rpn_module: nn.Module,
        roi_module: nn.Module,
        neck: Optional[nn.Module] = None,
        rpn_out_keys: Optional[List[str]] = None,
        output_feat: bool = False,
        pe_desc: Optional[nn.Module] = None,
    ):
        super().__init__(
            backbone=backbone,
            rpn_module=rpn_module,
            roi_module=roi_module,
            neck=neck,
            rpn_out_keys=rpn_out_keys,
            output_feat=output_feat,
        )
        self.pe_desc = pe_desc

    def forward(
        self, data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        # feature extraction
        feat_maps = self._extract_feature(
            data["img"], data.get("uv_map", None)
        )

        # assemble model outputs
        out_dict = OrderedDict()

        if self._output_feat:
            out_dict["feature_maps"] = feat_maps

        rpn_out = self.rpn_module(feat_maps, y=data)
        assert not set(rpn_out.keys()).intersection(set(out_dict.keys()))
        if self._rpn_out_keys is not None:
            out_dict.update({k: rpn_out[k] for k in self._rpn_out_keys})
        else:
            out_dict.update(rpn_out)

        coordinate_map = data["coordinate_map"]
        if self.pe_desc is not None:
            coordinate_map = self.pe_desc(coordinate_map)
        roi_out = self.roi_module(feat_maps, coordinate_map, rpn_out, y=data)

        assert not set(roi_out.keys()).intersection(set(out_dict.keys()))
        out_dict.update(roi_out)

        return out_dict
