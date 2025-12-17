from typing import Dict, List, Optional

import torch

from hat.core.nus_box3d_utils import inverse_bbox3d_nus_transform
from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class NMSFreeBBoxCoder(object):
    """Bbox coder for NMS-free detector.

    Args:
        pc_range: Range of point cloud.
        post_center_range: Limit of the center.
            Default: None.
        max_num: Max number to be kept. Default: 100.
        score_threshold: Threshold to filter boxes based on score.
            Default: None.
    """

    def __init__(
        self,
        pc_range: List[float],
        post_center_range: List[float] = None,
        max_num: int = 100,
        score_threshold: Optional[float] = None,
    ):
        self.pc_range = pc_range
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold

    def encode(self) -> None:

        pass

    def decode_single(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        num_classes: int,
    ) -> Dict[str, torch.Tensor]:
        """Decode bboxes.

        Args:
            cls_scores: Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds: Outputs from the regression \
                head with normalized coordinate format \
                (cx, cy, w, l, cz, h, yaw_sin, yaw_cos, vx, vy). \
                Shape [num_query, 10].
            num_classes: numbers of classes.
        Returns:
            Decoded 3D boxes with format \
                (cx, cy, cz, w, l,  h, yaw, vx, vy).
        """
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % num_classes
        bbox_index = indexs // num_classes
        bbox_preds = bbox_preds[bbox_index]

        final_box_preds = inverse_bbox3d_nus_transform(bbox_preds)
        final_scores = scores
        final_preds = labels

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
            tmp_score = self.score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = final_scores > -1
                    break
                thresh_mask = final_scores >= tmp_score

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(  # type: ignore
                self.post_center_range, device=scores.device
            )
            mask = (
                final_box_preds[..., :3] >= self.post_center_range[:3]
            ).all(1)
            mask &= (
                final_box_preds[..., :3] <= self.post_center_range[3:]
            ).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            predictions_dict = {
                "bboxes": boxes3d,
                "scores": scores,
                "labels": labels,
            }

        else:
            raise NotImplementedError(
                "Need to reorganize output as a batch, only "
                "support post_center_range is not None for now!"
            )
        return predictions_dict

    def decode(
        self, preds_dicts: Dict[str, List[torch.Tensor]], num_classes: int
    ) -> List[dict]:
        """Decode bboxes.

        Args:
            preds_dicts: \
            all_cls_scores: Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds: Sigmoid outputs from the regression \
                head with normalized coordinate format \
                (cx, cy, w, l, cz, h, yaw_sin, yaw_cos, vx, vy). \
                Shape [nb_dec, bs, num_query, 10].
        Returns:
            Decoded 3D boxes with format \
                (cx, cy, cz, w, l,  h, yaw, vx, vy).
        """
        all_cls_scores = preds_dicts["all_cls_scores"][-1]
        if preds_dicts.get("refined_bbox_preds", None) is not None:
            all_bbox_preds = preds_dicts["refined_bbox_preds"][-1]
        else:
            all_bbox_preds = preds_dicts["all_bbox_preds"][-1]

        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(
                self.decode_single(
                    all_cls_scores[i], all_bbox_preds[i], num_classes
                )
            )
        return predictions_list
