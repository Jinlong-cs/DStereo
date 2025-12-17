from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class TraceRcnnModule(nn.Module):
    """The second Rcnn tasks trace architecture for two-stage model deployment.

    Args:
        obj_type: task type be traced.
        quant_module: the module to quantize fpn feats input.
        fpn_desc: desc module for fpn input.
        roi_desc: desc module for roi input.
        roi_module: roi module for roi tasks.
    """

    def __init__(
        self,
        obj_type: Optional[str],
        quant_module: Optional[nn.Module],
        fpn_desc: Optional[nn.Module],
        roi_desc: Optional[nn.Module],
        roi_module: Optional[nn.Module],
    ) -> None:
        super().__init__()

        self.obj_type = obj_type
        self.quant_module = quant_module
        self.fpn_desc = fpn_desc
        self.roi_desc = roi_desc
        self.roi_module = roi_module
        self.roi_module.eval()

    def forward(
        self, input: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        out_dict = OrderedDict()
        rpn_out = {}
        feat_maps = input["feats_input"]
        if self.fpn_desc:
            feat_maps = self.fpn_desc(feat_maps)
        feat_maps = self.quant_module(feat_maps)
        for k, v in input.items():
            if self.obj_type in k:
                if self.roi_desc:
                    v = self.roi_desc(v)
                rpn_out["pred_bboxes"] = v
                break

        roi_out = self.roi_module(feat_maps, rpn_out)
        out_dict.update(roi_out)
        return out_dict
