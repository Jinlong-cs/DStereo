from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from hat.models.base_modules.postprocess import FilterModule
from hat.registry import OBJECT_REGISTRY

__all__ = ["RPNTrafficSignFilter"]


@OBJECT_REGISTRY.register
class RPNTrafficSignFilter(nn.Module):
    """Filter Module for traffic sign assist.

    Args:
        threshold: Threshold, the lower bound of output.
        strides: input_size / feature_size in (h, w).
        idx_range: The index range of
            values counted in compare of the first input.
            Defaults to None which means use all the values.
        num_classes: Class number. Should be the number of foreground
            classes.
    """

    def __init__(
        self,
        threshold: float,
        strides: Optional[Sequence[int]] = None,
        idx_range: Optional[Tuple[int, int]] = None,
        anchor_args: Optional[dict] = None,
        rpn_head_out_key: Optional[str] = "rpn_head_out",
    ):
        super(RPNTrafficSignFilter, self).__init__()
        assert strides == [1] or strides is None
        self.filter_module = FilterModule(
            threshold=threshold,
            idx_range=idx_range,
        )
        self.anchor_args = anchor_args
        self.rpn_head_out_key = rpn_head_out_key

    def forward(
        self,
        head_out: Dict[str, List[torch.Tensor]],
        im_hw: Optional[Tuple[int, int]] = None,
    ) -> Sequence[torch.Tensor]:

        rpn_feature = head_out[self.rpn_head_out_key][0]
        num_anchor = len(self.anchor_args["anchor_wh_groups"])
        dim = rpn_feature.shape[1] // num_anchor
        rpn_feature = rpn_feature.view(
            -1, num_anchor, dim, *rpn_feature.shape[2:]
        )
        cls_pred = rpn_feature[:, :, 4:].flatten(1, 2)
        reg_pred = rpn_feature[:, :, :4].flatten(1, 2)

        filter_output = self.filter_module(*[cls_pred, reg_pred])

        return {
            "filter_coord": filter_output[0][-3],
            "filter_scores": filter_output[0][-2],
            "filter_bboxes": filter_output[0][-1],
        }
