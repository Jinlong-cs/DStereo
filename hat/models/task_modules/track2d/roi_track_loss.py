# Copyright (c) Horizon Robotics. All rights reserved.

from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from hat.registry import OBJECT_REGISTRY

__all__ = [
    "TrackLoss",
]


@OBJECT_REGISTRY.register
class TrackLoss(nn.Module):
    """Track Loss module.

    Compute track loss by  'match_loss' module. The 'match_loss'
    module is specified in corresponding config.

    Args:
        match_loss: track match loss module.
        out_debug_info: Whether to output debug info (match acc /
            match fg_acc / fg_fraction) to the result dict.
            Default to False.
    """

    def __init__(
        self,
        match_loss: nn.Module,
        out_debug_info: Optional[bool] = False,
    ):
        super().__init__()
        self.match_loss = match_loss
        self.out_debug_info = out_debug_info

    def cal_metric(self, dist, target, mask):
        cls_ret_valid = torch.where(dist[mask > 0] > 0.5, 1, 0)

        cls_gt_valid = target[mask > 0]

        cls_acc = torch.mean((cls_ret_valid == cls_gt_valid).type(torch.float))

        fg_mask = cls_gt_valid > 0
        cls_fg_acc = torch.mean(
            (cls_ret_valid[fg_mask] == cls_gt_valid[fg_mask]).type(torch.float)
        )

        fg_fraction = ((cls_gt_valid > 0).sum()) / (
            (cls_gt_valid >= 0).sum() + 1e-6
        )

        return OrderedDict(
            match_acc=cls_acc,
            match_fg_acc=cls_fg_acc,
            match_fg_fraction=fg_fraction,
        )

    @autocast(enabled=False)
    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        out_dict = OrderedDict()

        track_feat = preds["track_feat"]
        # convert to float32 while using amp
        track_feat = track_feat.float()
        track_id = labels["track_id"]
        num_seq = labels["num_seq"][0]
        seq_len = labels["seq_len"][0]
        emb_len = track_feat.shape[1]
        num_cls = track_id.shape[-1]
        # num_seq x num_frame_per_seq x num_rois_per_frame x feat_len_per_roi
        track_feat_reorg = track_feat.reshape((num_seq, seq_len, -1, emb_len))
        track_feat_norm = F.normalize(track_feat_reorg, p=2, dim=-1)
        # num_seq x num_frame_per_seq x rois_track_id_per_frame x num_class
        track_id = labels["track_id"].reshape((num_seq, seq_len, -1, num_cls))
        num_rois_per_frame = track_id.shape[2]

        loss = None
        for frame_index in range(1, seq_len):
            dist = torch.bmm(
                track_feat_norm[:, frame_index - 1, :, :],
                track_feat_norm[:, frame_index, :, :].transpose(-1, -2),
            )
            # (num_seq x num_rois_per_frame x feat_len_per_roi) x
            #   (num_seq x feat_len_per_roi x num_rois_per_frame)
            # -->  num_seq x num_rois_per_frame x num_rois_per_frame
            #
            # dist[i]: row is the frame_t rois, col is the frame_{t-1} rois
            # each element is dist of frame_{t-1} roi_i and frame_t roi_j

            #    \frame_t |  roi1  |    roi2   |  ...  |   roin
            # frame_{t-1} |
            #     roi1    | dist11 |   dist12  |  ...  |   dist1n
            #     roi2    | dist21 |   dist22  |  ...  |   dist2n
            #      ...    |   ...  |    ...    |  ...  |    ...
            #     roin    | distn1 |   distn2  |  ...  |   distnn

            for cls in range(num_cls):
                key_id = track_id[:, frame_index - 1, :, cls]
                ref_id = track_id[:, frame_index, :, cls]
                key_id_mat = key_id.unsqueeze(-1).expand(
                    *key_id.shape, num_rois_per_frame
                )
                # frame_{t-1} track_id_maps:
                # roi1  | track_id1 | track_id1 |  ...  | track_id1
                # roi2  | track_id2 | track_id2 |  ...  | track_id2
                # ...   |    ...    |    ...    |  ...  |    ...
                # roin  | track_idn | track_idn |  ...  | track_idn

                ref_id_mat = ref_id.unsqueeze(-2).expand(
                    -1, num_rois_per_frame, num_rois_per_frame
                )
                # frame_{t} track_id_maps:
                #          roi1     |  roi2     |  ...  |   roin
                #        track_id1  | track_id2 |  ...  | track_idn
                #        track_id1  | track_id2 |  ...  | track_idn
                #           ...     |   ...     |  ...  |    ...
                #        track_id1  | track_id2 |  ...  | track_idn

                target = -1 * torch.ones_like(dist)
                # key and ref rois need all are positive (id > 0)
                valid_index = (key_id_mat > 0) & (ref_id_mat > 0)
                matched_index = (key_id_mat == ref_id_mat) & valid_index
                unmatch_index = (key_id_mat != ref_id_mat) & valid_index
                target[matched_index] = 1
                target[unmatch_index] = 0
                weight = target != -1
                loss_i = self.match_loss(dist, target, weight)
                if loss is None:
                    loss = [loss_i]
                else:
                    loss.append(loss_i)

                # only cal last dist metric for simple
                if self.out_debug_info:
                    acc_dict = self.cal_metric(
                        dist.detach(), target.detach(), weight.detach()
                    )
        loss_mean = loss if loss is None else sum(loss) / len(loss)
        out_dict["rcnn_track_loss"] = loss_mean
        if self.out_debug_info:
            out_dict.update(acc_dict)
        return out_dict
