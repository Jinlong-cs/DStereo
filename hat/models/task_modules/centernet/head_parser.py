from typing import Dict

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY

__all__ = ["CenterNetHeadParser"]


@OBJECT_REGISTRY.register
class CenterNetHeadParser(nn.Module):
    def forward(
        self, preds: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Process raw head out to bboxes for IOU Losses.

        Args:
            preds: Should at least contain keys below.
                heatmap_pred: Heatmap tensor.
                wh_pred: Width and height prediction.
                offset_pred: Center offset prediction.

        Returns:
            A dict containing keys below:
                heatmap_pred: The same as input.
                wh_pred: The same as input.
                offset_pred: The same as input.
                bbox_pred: Bbox prediction, shape is (bs, 4, feath, featw).
        """
        wh_pred = preds["wh_pred"]
        offset_pred = preds["offset_pred"]

        feat_h, feat_w = wh_pred.shape[-2:]
        center_coord = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(feat_w),
                    torch.arange(feat_h),
                ),
                dim=0,
            )
            .unsqueeze(0)
            .to(wh_pred.device)
            .to(wh_pred.dtype)
            .transpose(3, 2)
            .contiguous()
        )
        center_coord = center_coord + offset_pred
        wh_pred_sq = wh_pred.pow(2) + 1e-3
        bbox_feat = torch.cat(
            [
                center_coord - wh_pred_sq / 2,
                center_coord + wh_pred_sq / 2,
            ],
            dim=1,
        )
        output = preds
        output.update(bbox_pred=bbox_feat)
        return output
