# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

# from .post_processing import box3d_multiclass_nms
from hat.core.box3d_utils import bev3d_multiclass_nms
from hat.registry import OBJECT_REGISTRY

__all__ = ["MVTPostProcess"]


@OBJECT_REGISTRY.register
class MVTPostProcess(nn.Module):
    """The postprocess of MultiView Transformer for 3D Detection task.

    Args:
        num_classes: Number of classes.
        bbox_coder: Bbox coder for NMS-free detector.
        test_cfg: Configs for testing, including nms_cfg.
    """

    def __init__(
        self,
        num_classes: int,
        bbox_coder: nn.Module,
        test_cfg: Dict[str, Dict[str, Any]],
    ):
        super(MVTPostProcess, self).__init__()
        self.num_classes = num_classes
        self.bbox_coder = bbox_coder
        self.test_cfg = test_cfg

    @autocast(enabled=False)
    def forward(
        self, preds_dicts: Tuple[List[Dict[str, torch.Tensor]]]
    ) -> List[List[torch.Tensor]]:
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts: Prediction results.
        Returns:
            Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(  # type: ignore
            preds_dicts, self.num_classes
        )
        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds["bboxes"]
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5  # bottom center
            scores = preds["scores"]
            labels = preds["labels"]

            # nms post processing
            nms_cfg = self.test_cfg["nms_cfg"]
            scores_for_nms = torch.nn.functional.one_hot(
                labels, self.num_classes
            ).to(dtype=scores.dtype)
            scores_for_nms[torch.arange(labels.shape[0]), labels] = scores
            bev_xylwr = bboxes[:, [0, 1, 3, 4, 6]].clone()
            bboxes, scores, labels = bev3d_multiclass_nms(
                bboxes,
                bev_xylwr,
                scores_for_nms,
                nms_cfg["score_thr"],
                nms_cfg["nms_thr"],
                nms_cfg["max_num"],
            )

            ret_list.append([bboxes, scores, labels])
        return ret_list
