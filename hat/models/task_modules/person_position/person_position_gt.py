from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor

from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class GetPersonPositionGT(nn.Module):
    """Get person position ground truth in postprocess.

    It is used in postprocess to get person position ground truth
    boxes and labels as prediction box. Usually used when inference
    with ground truth.

    Args:
        roi_nums: Maximum number of roi.
    """

    def __init__(
        self,
        roi_nums: int,
    ) -> None:
        super().__init__()
        self.roi_nums = roi_nums

    def forward(
        self,
        batch_rois: List[Tensor] = None,
        head_out: Dict[str, List[torch.Tensor]] = None,
        gt_boxes: Tensor = None,
        gt_boxes_num: Tensor = None,
    ):
        if gt_boxes is not None and gt_boxes_num is not None:
            for i in range(len(gt_boxes)):
                box = (
                    torch.zeros((self.roi_nums, 6), device=gt_boxes.device) - 1
                )
                box_num = int(gt_boxes_num[i].item())
                box[:box_num, :4] = gt_boxes[i, :box_num, :4]
                box[:box_num, 4] = 1.0
                box[:box_num, 5] = gt_boxes[i, :box_num, 4]
                box = box.unsqueeze(0)

                if i == 0:
                    output = box
                else:
                    output = torch.cat((output, box))
            return OrderedDict(
                gt_pred=output,
            )
