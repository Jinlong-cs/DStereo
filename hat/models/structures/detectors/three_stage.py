# Copyright (c) Horizon Robotics. All rights reserved.

import logging
from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY
from .two_stage import TwoStageDetector

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class ThreeStageDetector(TwoStageDetector):
    """The three stage detector structure.

    Args:
        backbone: backbone network.
        neck: neck network.
        rpn_module: region proposal network module.
        roi_module: roi module.
        cvt_module: convert roi_module output to three_stage_module input.
        three_stage_module: three_stage roi module.
        rpn_out_keys: keys of rpn output that should be included in final
            outputs. By default, all rpn output are included.
        output_feat: Whether to include extracted feature maps in final outpus.

    """

    def __init__(
        self,
        backbone: nn.Module,
        rpn_module: nn.Module,
        roi_module: nn.Module,
        three_stage_module: nn.Module,
        cvt_module: Optional[nn.Module] = None,
        neck: Optional[nn.Module] = None,
        rpn_out_keys: Optional[List[str]] = None,
        output_feat: bool = False,
    ):
        super(ThreeStageDetector, self).__init__(
            backbone,
            rpn_module,
            roi_module,
            neck,
            rpn_out_keys,
            output_feat=output_feat,
        )
        self.cvt_module = cvt_module
        self.three_stage_module = three_stage_module
        self._output_feat = output_feat

    @property
    def with_roi(self):
        return self.three_stage_module is not None

    def forward(
        self, data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        #
        # feature extraction
        feat_maps = self._extract_feature(data["img"])

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

        roi_out = self.roi_module(feat_maps, rpn_out, y=data)
        assert not set(roi_out.keys()).intersection(set(out_dict.keys()))
        if self.roi_module._output_head_out:
            out_dict.update(roi_out)

        if self.cvt_module is not None:
            three_stage_input = self.cvt_module(roi_out)
            roi_task_out = self.three_stage_module(
                feat_maps, three_stage_input, y=data
            )
        else:
            roi_task_out = self.three_stage_module(feat_maps, roi_out, y=data)

        assert not set(roi_task_out.keys()).intersection(set(out_dict.keys()))
        out_dict.update(roi_task_out)

        return out_dict
