from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from hat.core.nus_box3d_utils import bbox3d_nus_transform
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import multi_apply
from hat.utils.distributed import reduce_mean

__all__ = ["HungarianBBox3DLoss"]


@OBJECT_REGISTRY.register
class HungarianBBox3DLoss(nn.Module):
    """Loss computation with hungarian assigner for 3d object task.

    There are 2 losses included: focal loss for classification, l1 loss for
    bbox regression.

    Args:
        num_classes: number of classes.
        code weights: the code weights for bbox regression.
        loss_cls: loss for classification.
        loss_bbox: loss for bbox regression.
        assigner: hungarian bbox3d assigner.
        sync_cls_avg_factor: if True, reduce mean to get cls_avg_factor.
        bg_cls_weight: the weight of background class. default 0.
    """

    def __init__(
        self,
        loss_cls: nn.Module,
        loss_bbox: nn.Module,
        assigner: nn.Module,
        num_classes: int = 0,
        code_weights: List[float] = [1.0] * 8 + [0.0] * 2,
        sync_cls_avg_factor: bool = True,
        bg_cls_weight: float = 0.0,
    ):
        super(HungarianBBox3DLoss, self).__init__()

        self.loss_cls = loss_cls
        self.loss_bbox = loss_bbox
        self.assigner = assigner
        self.num_classes = num_classes

        self.register_buffer(
            "code_weights", torch.tensor(code_weights), persistent=False
        )

        self.bg_cls_weight = bg_cls_weight
        self.sync_cls_avg_factor = sync_cls_avg_factor

    def _sampling_bbox(self, assign_result, gt_bboxes):
        pos_inds = (
            torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        neg_inds = (
            torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )

        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]

        return (pos_inds, neg_inds, pos_assigned_gt_inds, pos_gt_bboxes)

    def _get_target_single(
        self,
        cls_score: torch.Tensor,
        bbox_pred: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
    ) -> Any:
        """Compute regression and classification targets for one image. \
            Outputs from a single decoder layer of \
            a single feature level are used.

        Args:
            cls_score: Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred: Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes: Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels: Ground truth class indices for one image
                with shape (num_gts, ).
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels: Labels of each image.
                - label_weights: Label weights of each image.
                - bbox_targets: BBox targets of each image.
                - bbox_weights: BBox weights of each image.
                - pos_inds: Sampled positive indices for each image.
                - neg_inds: Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(  # type: ignore
            bbox_pred,
            cls_score,
            gt_bboxes,
            gt_labels,
        )

        (
            pos_inds,
            neg_inds,
            pos_assigned_gt_inds,
            pos_gt_bboxes,
        ) = self._sampling_bbox(assign_result, gt_bboxes)

        # label targets
        labels = gt_bboxes.new_full(
            (num_bboxes,), self.num_classes, dtype=torch.long
        )
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[
            ..., : bbox_pred.shape[-1] - 1
        ]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        if gt_bboxes.numel() == 0:
            bbox_targets[pos_inds] = torch.empty_like(gt_bboxes)
        else:
            bbox_targets[pos_inds] = pos_gt_bboxes
        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            pos_inds,
            neg_inds,
        )

    def get_targets(
        self,
        cls_scores_list: List[torch.Tensor],
        bbox_preds_list: List[torch.Tensor],
        gt_bboxes_list: List[torch.Tensor],
        gt_labels_list: List[torch.Tensor],
    ) -> Any:
        """Compute regression and classification targets for a batch image. \
                Outputs from a single decoder layer of a \
                single feature level are used.

        Args:
            cls_scores_list: Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list: Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list: Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list: Ground truth class indices for each
                image with shape (num_gts, ).
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list: Labels for all images.
                - label_weights_list: Label weights for all \
                    images.
                - bbox_targets_list: BBox targets for all \
                    images.
                - bbox_weights_list: BBox weights for all \
                    images.
                - num_total_pos: Number of positive samples in all \
                    images.
                - num_total_neg: Number of negative samples in all \
                    images.
        """

        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_target_single,
            cls_scores_list,
            bbox_preds_list,
            gt_labels_list,
            gt_bboxes_list,
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        )

    def loss_single(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        gt_bboxes_list: List[torch.Tensor],
        gt_labels_list: List[torch.Tensor],
        refined_bbox_preds: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loss function for outputs from a single decoder \
            layer of a single feature level.

        Args:
            cls_scores: Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds: Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list: Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list: Ground truth class indices for each
                image with shape (num_gts, ).
            refined_bbox_preds: Sigmoid refined outputs from a single decoder
                layer for all images, with normalized coordinate (cx, cy, w, h)
                and shape [bs, num_query, 4].
        Returns:
            A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]

        gt_labels_mask_list = [
            gt_labels_ >= 0 for gt_labels_ in gt_labels_list
        ]
        gt_bboxes_list = [
            gt_bboxes_[gt_labels_mask_list[img_i]]
            for img_i, gt_bboxes_ in enumerate(gt_bboxes_list)
        ]
        gt_labels_list = [
            gt_labels_[gt_labels_mask_list[img_i]]
            for img_i, gt_labels_ in enumerate(gt_labels_list)
        ]
        cls_reg_targets = self.get_targets(
            cls_scores_list,
            bbox_preds_list,
            gt_bboxes_list,
            gt_labels_list,
        )
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.num_classes)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = (
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        )
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor])
            )

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor
        )
        loss_cls = loss_cls.get(self.loss_cls.loss_name, 0)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = bbox3d_nus_transform(bbox_targets)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        bbox_pred_channels = bbox_preds.size(-1)
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :bbox_pred_channels],
            normalized_bbox_targets[isnotnan, :bbox_pred_channels],
            bbox_weights[isnotnan, :bbox_pred_channels],
            avg_factor=num_total_pos,
        )

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)

        if refined_bbox_preds is not None:
            refined_bbox_preds = refined_bbox_preds.reshape(
                -1, bbox_preds.size(-1)
            )
            loss_refined_bbox = self.loss_bbox(
                refined_bbox_preds[isnotnan, :bbox_pred_channels],
                normalized_bbox_targets[isnotnan, :bbox_pred_channels],
                bbox_weights[isnotnan, :bbox_pred_channels],
                avg_factor=num_total_pos,
            )
            loss_refined_bbox = torch.nan_to_num(loss_refined_bbox)
            loss_bbox = loss_bbox + loss_refined_bbox

        return loss_cls, loss_bbox

    @autocast(enabled=False)
    def forward(
        self,
        gt_bboxes_list: List[torch.Tensor],
        gt_labels_list: List[torch.Tensor],
        preds_dicts: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        all_cls_scores = preds_dicts["all_cls_scores"]
        all_bbox_preds = preds_dicts["all_bbox_preds"]
        enc_cls_scores = preds_dicts.get("enc_cls_scores", None)
        enc_bbox_preds = preds_dicts.get("enc_bbox_preds", None)
        refined_bbox_preds = preds_dicts.get("refined_bbox_preds", None)

        num_dec_layers = len(all_cls_scores)
        if refined_bbox_preds is None:
            refined_bbox_preds = [None] * num_dec_layers

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_bbox_preds,
            all_gt_bboxes_list,
            all_gt_labels_list,
            refined_bbox_preds,
        )

        loss_dict = OrderedDict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = self.loss_single(
                enc_cls_scores,
                enc_bbox_preds,
                gt_bboxes_list,
                binary_labels_list,
                None,
            )
            loss_dict["enc_loss_cls"] = enc_loss_cls
            loss_dict["enc_loss_bbox"] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict["loss_cls"] = losses_cls[-1]
        loss_dict["loss_bbox"] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f"d{num_dec_layer}.loss_cls"] = loss_cls_i
            loss_dict[f"d{num_dec_layer}.loss_bbox"] = loss_bbox_i
            num_dec_layer += 1

        return loss_dict
