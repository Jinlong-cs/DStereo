from typing import Dict, List

import torch
import torch.nn as nn

from hat.core.data_struct.base_struct import Points2D, Points2D_2
from hat.registry import OBJECT_REGISTRY

__all__ = ["KpsDecoder"]


@OBJECT_REGISTRY.register
class KpsDecoder(nn.Module):
    """Decoder for Key Points detection.

    Args:
        num_kps: Number of key points.
        pos_distance: float,
        roi_expand_param: Param to expand roi.
        input_padding: Input padding,
            in (w_left, w_right, h_top, h_bottom) format.
    """

    def __init__(
        self,
        num_kps: int,
        pos_distance: float,
        roi_expand_param: float = 1.0,
    ):

        super().__init__()
        self.num_kps = num_kps
        self.pos_distance = pos_distance
        self.roi_expand_param = roi_expand_param

    @torch.no_grad()
    def forward(
        self, batch_rois: List[torch.Tensor], head_out: Dict[str, torch.Tensor]
    ):
        label_maps = head_out["kps_rcnn_cls_pred"]
        offsets = head_out["kps_rcnn_reg_pred"]

        try:
            batch_rois = torch.stack(
                [rois.dequantize() for rois in batch_rois]
            )
        except NotImplementedError:
            batch_rois = torch.stack(
                [rois.as_subclass(torch.Tensor) for rois in batch_rois]
            )

        result = []
        bs = batch_rois.shape[0]
        label_maps = label_maps.reshape(
            bs,
            -1,
            label_maps.shape[1],
            label_maps.shape[2],
            label_maps.shape[3],
        )
        offsets = offsets.reshape(
            bs, -1, offsets.shape[1], offsets.shape[2], offsets.shape[3]
        )
        for rois, label_map, offset in zip(batch_rois, label_maps, offsets):
            # rois = rois[0][..., :4]
            rois = rois[..., :4]
            rois = rois.reshape(-1, 4)
            roi_wh = (rois[:, 2:] - rois[:, :2]) * self.roi_expand_param
            center = (rois[:, 2:] + rois[:, :2]) / 2

            new_rois = rois.clone()
            new_rois[:, :2] = center - roi_wh / 2
            new_rois[:, 2:] = center + roi_wh / 2
            rois = new_rois.clone()

            _, _, feat_width, feat_height = label_map.shape

            id_x = []
            id_y = []
            for i in range(offset.shape[1]):
                if i % 2 == 0:
                    id_x.append(i)
                else:
                    id_y.append(i)

            kps_deltas_x = offset[:, id_x]
            kps_deltas_y = offset[:, id_y]

            kps_scores = label_map.reshape((-1, feat_width * feat_height))
            kps_deltas_x = kps_deltas_x.reshape((-1, feat_width * feat_height))
            kps_deltas_y = kps_deltas_y.reshape((-1, feat_width * feat_height))
            max_inds = torch.argmax(kps_scores, dim=1, keepdim=True)

            max_inds_x = max_inds.squeeze() % feat_width
            max_inds_y = (max_inds.squeeze() / feat_width).floor()

            max_scores = torch.gather(
                kps_scores, index=max_inds, dim=1
            ).squeeze()
            max_deltas_x = torch.gather(
                kps_deltas_x, index=max_inds, dim=1
            ).squeeze()
            max_deltas_y = torch.gather(
                kps_deltas_y, index=max_inds, dim=1
            ).squeeze()

            max_deltas_x = max_deltas_x * self.pos_distance
            max_deltas_y = max_deltas_y * self.pos_distance

            scales_x = feat_width / ((rois[:, 2] - rois[:, 0]) + 1)  # noqa
            scales_y = feat_width / ((rois[:, 3] - rois[:, 1]) + 1)  # noqa
            scales_x = (
                scales_x.unsqueeze(1)
                .repeat(1, self.num_kps)
                .reshape(
                    -1,
                )
            )
            scales_y = (
                scales_y.unsqueeze(1)
                .repeat(1, self.num_kps)
                .reshape(
                    -1,
                )
            )

            offsets_x = rois[:, 0]
            offsets_y = rois[:, 1]
            offsets_x = (
                offsets_x.unsqueeze(1)
                .repeat(1, self.num_kps)
                .reshape(
                    -1,
                )
            )
            offsets_y = (
                offsets_y.unsqueeze(1)
                .repeat(1, self.num_kps)
                .reshape(
                    -1,
                )
            )

            pred_kps_x = ((max_inds_x + max_deltas_x) / scales_x) + offsets_x
            pred_kps_y = ((max_inds_y + max_deltas_y) / scales_y) + offsets_y

            pred_kps = torch.stack([pred_kps_x, pred_kps_y, max_scores], dim=1)
            pred_kps = pred_kps.reshape((-1, self.num_kps, 3))

            kps = {
                f"points{i}": Points2D(
                    points=pred_kps[:, i, :2],
                    scores=pred_kps[:, i, 2],
                    cls_idxs=pred_kps.new_zeros(pred_kps.shape[0]),
                )
                for i in range(self.num_kps)
            }
            result.append(Points2D_2(**kps))

        return {"pred_kps": result}
