from typing import List, Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from hat.registry import OBJECT_REGISTRY

__all__ = ["DisparityNetPostProcess"]


@OBJECT_REGISTRY.register
class DisparityNetPostProcess(nn.Module):
    """
    A basic post process for StereoNet.

    Args:
        maxdisp: The max value of disparity.
    """

    def __init__(self, maxdisp: int = 320):
        super(DisparityNetPostProcess, self).__init__()
        self.maxdisp = maxdisp

    def forward(
        self,
        pred_disps: List[Tensor],
        gt_disps: List[Tensor] = None,
    ) -> Union[Tensor, List[Tensor]]:
        """Perform the forward pass of the model.

        Args:
            pred_disps: The model outputs.
            gt_disps: The gt disparitys.

        Returns:
            pred_disps: The prediction disparitys.
        """

        if self.training:
            assert gt_disps is not None
            disp_tmp = F.interpolate(
                pred_disps[-2],
                size=pred_disps[-1].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            pred_disps[-1] = F.relu(disp_tmp + pred_disps[-1])
            for i in range(len(pred_disps)):
                pred_disp_w = pred_disps[i].size()[-1]
                gt_disp_w = gt_disps.size()[-1]
                pred_disps[i] = pred_disps[i] * self.maxdisp
                if pred_disp_w != gt_disp_w:
                    pred_disps[i] = F.interpolate(
                        pred_disps[i], size=gt_disps.shape[1:], mode="bilinear"
                    )
                pred_disps[i] = pred_disps[i].squeeze(1)

            return pred_disps

        else:
            disp_tmp = F.interpolate(
                pred_disps[-2],
                size=pred_disps[-1].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            pred_disps[-1] = F.relu(disp_tmp + pred_disps[-1])
            pred_disps[-1] = pred_disps[-1].squeeze(1) * self.maxdisp

            return pred_disps[-1]
