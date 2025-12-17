import torch
import torch.nn as nn
import torch.nn.functional as F

from hat.core.box_utils import zoom_boxes
from hat.registry import OBJECT_REGISTRY

__all__ = ["HumanPoseDecoder"]


@OBJECT_REGISTRY.register
class HumanPoseDecoder(nn.Module):
    """Human pose decoder structure.

    Decode gt landmark from the prediction heatmap and offset.
    This class is mainly used in detecor like faster-rcnn.

    Args:
        num_ldmk: Number of landmark.
        pos_distance: Distance about landmark's coordinates. Defaults to 1.
        pos_distance_scale: Distance scale about
            landmark's coordinates. Defaults to 1.
        roi_expand_param: Params about expand roi. Defaults to 1.0.
        loss_type: Loss function to get ldmk score. Defaults to sigmoid.
        ldmk_transform_scores: Wheter transform ldmk score. Defaults to True.
    """

    def __init__(
        self,
        num_ldmk: int,
        pos_distance: float = 1.0,
        pos_distance_scale: float = 1.0,
        roi_expand_param: float = 1.0,
        loss_type: str = "sigmoid",
        ldmk_transform_scores: bool = True,
    ):
        super().__init__()
        self.num_ldmk = num_ldmk
        self.pos_distance = pos_distance
        self.pos_distance_scale = pos_distance_scale
        self.roi_expand_param = roi_expand_param
        self.loss_type = loss_type
        self.ldmk_transform_scores = ldmk_transform_scores

    @torch.no_grad()
    def forward(self, batch_rois, head_out):
        ldmk_pred = head_out["ldmk_pred"]
        label_maps = ldmk_pred[:, 0 : self.num_ldmk, :, :]
        offsets = ldmk_pred[:, self.num_ldmk : self.num_ldmk * 3, :, :]

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
            rois = rois[..., :4]
            rois = rois.reshape(-1, 4)
            rois = zoom_boxes(
                rois, (self.roi_expand_param, self.roi_expand_param)
            )

            _, _, feat_width, feat_height = label_map.shape

            ldmk_deltas_x = offset[:, ::2]
            ldmk_deltas_y = offset[:, 1::2]

            ldmk_scores = label_map.reshape((-1, feat_width * feat_height))
            if self.loss_type == "sigmoid":
                ldmk_scores = torch.sigmoid(ldmk_scores)
            elif self.loss_type == "softmax":
                ldmk_scores = F.softmax(ldmk_scores)
            else:
                raise Exception("loss type is not suppported!")
            pos_distance = self.pos_distance * feat_width

            if self.ldmk_transform_scores and pos_distance > 1.0:
                ldmk_scores = self._transform_ldmk_scores(
                    ldmk_scores, feat_height, feat_width
                )

            ldmk_deltas_x = ldmk_deltas_x.reshape(
                (-1, feat_width * feat_height)
            )
            ldmk_deltas_y = ldmk_deltas_y.reshape(
                (-1, feat_width * feat_height)
            )
            max_inds = torch.argmax(ldmk_scores, dim=1, keepdim=True)

            max_inds_x = max_inds.squeeze() % feat_width
            max_inds_y = (max_inds.squeeze() / feat_width).floor()

            max_scores = torch.gather(
                ldmk_scores, index=max_inds, dim=1
            ).squeeze()
            max_deltas_x = torch.gather(
                ldmk_deltas_x, index=max_inds, dim=1
            ).squeeze()
            max_deltas_y = torch.gather(
                ldmk_deltas_y, index=max_inds, dim=1
            ).squeeze()

            max_deltas_x = max_deltas_x * pos_distance
            max_deltas_y = max_deltas_y * pos_distance
            max_deltas_x = max_deltas_x / self.pos_distance_scale
            max_deltas_y = max_deltas_y / self.pos_distance_scale
            scales_x = feat_width / ((rois[:, 2] - rois[:, 0]) + 1)
            scales_y = feat_height / ((rois[:, 3] - rois[:, 1]) + 1)
            scales_x = (
                scales_x.unsqueeze(1)
                .repeat(1, self.num_ldmk)
                .reshape(
                    -1,
                )
            )
            scales_y = (
                scales_y.unsqueeze(1)
                .repeat(1, self.num_ldmk)
                .reshape(
                    -1,
                )
            )

            offsets_x = rois[:, 0]
            offsets_y = rois[:, 1]
            offsets_x = (
                offsets_x.unsqueeze(1)
                .repeat(1, self.num_ldmk)
                .reshape(
                    -1,
                )
            )
            offsets_y = (
                offsets_y.unsqueeze(1)
                .repeat(1, self.num_ldmk)
                .reshape(
                    -1,
                )
            )

            pred_ldmk_x = ((max_inds_x + max_deltas_x) / scales_x) + offsets_x
            pred_ldmk_y = ((max_inds_y + max_deltas_y) / scales_y) + offsets_y
            pred_ldmk = torch.stack(
                [pred_ldmk_x, pred_ldmk_y, max_scores], dim=1
            )
            pred_ldmk = pred_ldmk.reshape((-1, self.num_ldmk * 3))
            result.append(pred_ldmk)

        return {"pred_ldmk": result}

    def _transform_ldmk_scores(self, ldmk_scores, feat_height, feat_width):
        ldmk_scores = ldmk_scores.reshape((-1, feat_height, feat_width))
        new_ldmk_scores = torch.zeros_like(ldmk_scores)
        for h in range(feat_height):
            for w in range(feat_width):
                count = 0
                for j in [-1, 0, 1]:
                    h_j = h + j
                    if 0 <= h_j < feat_height:
                        for i in [-1, 0, 1]:
                            w_i = w + i
                            if 0 <= w_i < feat_width:
                                count += 1
                                new_ldmk_scores[:, h, w] += ldmk_scores[
                                    :, h_j, w_i
                                ]
                new_ldmk_scores[:, h, w] /= count
        new_ldmk_scores = new_ldmk_scores.reshape(
            (-1, feat_height * feat_width)
        )
        return new_ldmk_scores
