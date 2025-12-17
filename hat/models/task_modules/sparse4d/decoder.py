# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Optional

import torch

from hat.core.box3d_utils import bev3d_let_nms
from hat.registry import OBJECT_REGISTRY
from .target import COS_YAW, SIN_YAW, VX, H, L, W, X, Y, Z


@OBJECT_REGISTRY.register
class SparseBox3DDecoder(object):
    def __init__(
        self,
        num_output: int = 300,
        score_threshold: Optional[float] = None,
        let_nms_kwargs: Optional[dict] = None,
    ):
        super(SparseBox3DDecoder, self).__init__()
        self.num_output = num_output
        self.score_threshold = score_threshold
        self.let_nms_kwargs = let_nms_kwargs

    def __call__(
        self,
        cls_scores: torch.Tensor,
        box_preds: torch.Tensor,
        centerness=None,
    ) -> dict:
        cls_scores = cls_scores[-1].sigmoid()
        box_preds = box_preds[-1]
        bs, num_pred, num_cls = cls_scores.shape
        cls_scores, indices = cls_scores.flatten(start_dim=1).topk(
            self.num_output, dim=1
        )
        cls_ids = indices % num_cls
        if self.score_threshold is not None:
            mask = cls_scores >= self.score_threshold

        if centerness is not None and centerness[-1] is not None:
            centerness = torch.gather(centerness[-1], 1, indices // num_cls)
            cls_scores *= centerness.sigmoid()
            cls_scores, idx = torch.sort(cls_scores, dim=1, descending=True)
            cls_ids = torch.gather(cls_ids, 1, idx)
            mask = torch.gather(mask, 1, idx)
            indices = torch.gather(indices, 1, idx)

        output = []
        for i in range(bs):
            category_ids = cls_ids[i]
            scores = cls_scores[i]
            box = box_preds[i, indices[i] // num_cls]
            if self.score_threshold is not None:
                category_ids = category_ids[mask[i]]
                scores = scores[mask[i]]
                box = box[mask[i]]

            yaw = torch.atan2(box[:, SIN_YAW], box[:, COS_YAW])
            box = torch.cat(
                [
                    box[:, [X, Y, Z]],
                    box[:, [W, L, H]].exp(),
                    yaw[:, None],
                    box[:, VX:],
                ],
                dim=-1,
            )
            if self.let_nms_kwargs is not None:
                nms_bbox = torch.cat(
                    [
                        box[:, [X, Y, Z]],
                        box[:, [H, L, W]],
                        yaw[:, None],
                        box[:, VX:],
                    ],
                    dim=-1,
                )
                keep = bev3d_let_nms(
                    nms_bbox[:, :7],
                    scores,
                    cls_ids=category_ids,
                    **self.let_nms_kwargs,  # noqa
                )
                box = box[keep]
                scores = scores[keep]
                category_ids = category_ids[keep]

            output.append(
                {
                    "boxes_3d": box.cpu(),
                    "scores_3d": scores.cpu(),
                    "labels_3d": category_ids.cpu(),
                }
            )
        return output
