from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from hat.registry import OBJECT_REGISTRY

__all__ = ["CenterNetIOULoss", "CenterNetFocalLoss"]


@OBJECT_REGISTRY.register
class CenterNetIOULoss(nn.Module):
    """Centernet loss wrapper with IOU loss.

    This class is an integration of Centernet cls loss and IOU loss.

    Args:
        heatmap_loss: heatmap loss calculation module.
        iou_loss: IOU loss calculation module.
    """

    def __init__(
        self,
        heatmap_loss: nn.Module,
        iou_loss: nn.Module,
    ):
        super().__init__()
        self.heatmap_loss = heatmap_loss  # Focal loss
        self.iou_loss = iou_loss  # GIOU or CIOU

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute losses of the head.

        Args:
            preds: A dict of head outs. Should have keys below.
                1. heatmap_pred: center predict heatmaps for all levels with
                shape (B, num_classes, H, W).
                2. wh_pred: wh predicts for all levels with shape (B, 2, H, W).
                3. offset_pred: offset predicts for all levels with shape
                (B, 2, H, W).

            target: A dict of training target, keys listed below.
                1. heatmap_pred: center predict heatmap with shape
                (B, num_classes, H, W).
                2. bbox_pred: bbox predict with shape (B, 4, H, W).
                3. gt_bboxes: bbox gt for each image with shape (num_gts, 4)
                in [tl_x, tl_y, br_x, br_y] format.
                4. gt_labels: class indices corresponding to each box.
                5. wh_offset_target_weight: specify positions that need bbox
                iou loss calculation.
                6. ig_bboxes_mask: A mask specifies where to ignore heatmap
                loss.

        Returns:
            A dict containing keys below:
                loss_center_heatmap: Loss of center heatmap.
                loss_iou: Loss of iou.

        """
        center_heatmap_pred = preds["heatmap_pred"]
        bbox_pred = preds["bbox_pred"]
        target_result, avg_factor = target
        center_heatmap_target = target_result["center_heatmap_target"]
        bbox_target = target_result["bbox_target"]
        wh_offset_target_weight = target_result["wh_offset_target_weight"]
        ig_bboxes_mask = target_result.get("ig_bboxes_mask", None)

        # Since the channel of wh_target and offset_target is 2, the avg_factor
        # of loss_center_heatmap is always 1/2 of loss_wh and loss_offset.
        loss_center_heatmap = self.heatmap_loss(
            center_heatmap_pred,
            center_heatmap_target,
            valid_classes_list=target_result["valid_classes_list"],
            heatmap_hard_mask=target_result["heatmap_hard_mask"],
            ig_bboxes_mask=ig_bboxes_mask,
        )

        bbox_target_weight = (
            wh_offset_target_weight.permute(0, 2, 3, 1)
            .contiguous()
            .repeat(1, 1, 1, 4)
        )

        loss_iou = self.iou_loss(
            bbox_pred.permute(0, 2, 3, 1).contiguous(),
            bbox_target.permute(0, 2, 3, 1).contiguous(),
            weight=bbox_target_weight,
            avg_factor=avg_factor,
        )

        return {
            "loss_center_heatmap": loss_center_heatmap,
            "loss_iou": loss_iou,
        }


@OBJECT_REGISTRY.register
class CenterNetFocalLoss(nn.Module):
    """Focal loss for CenterNet.

    This loss is able to deal with single-class labeled multiclass
    classification task.

    Args:
        loss_name: The key of loss in return dict.
        alpha: A weighting factor for pos-sample, (1-alpha) is for neg-sample.
        gamma: Gamma used in focal loss to compress the contribution of easy
            examples.
        loss_weight: Global weight of loss. Defaults is 1.0.
        eps: A small value to avoid zero in log function.

    """

    def __init__(
        self,
        loss_name: str,
        alpha: float = 2.0,
        gamma: float = 4.0,
        loss_weight: float = 1.0,
        eps: float = 1e-10,
    ):
        super(CenterNetFocalLoss, self).__init__()
        self.loss_name = loss_name
        self.alpha = alpha
        self.gamma = gamma
        self.loss_weight = loss_weight
        self.eps = eps

    def gaussian_focal_loss(
        self,
        pred,
        gaussian_target,
        valid_classes_list,
        heatmap_hard_mask,
        grad_tensor,
        ig_bboxes_mask,
    ):
        """Gaussian focal loss.

        Gaussian focal loss for single-class labeled multi-class
        classification.
        """

        #      "label_mask" is a [B, NC, H, W] mask, for each class
        # in each instance, only the "valid" classes will be marked True.
        label_mask = torch.zeros_like(pred).to(torch.bool)
        for batch_id, valid_classes in enumerate(valid_classes_list):
            label_mask[batch_id, valid_classes] = True

        pos_pos = gaussian_target.eq(1)

        #      Generate masks for negative loss according to label_mask.
        # a. For valid classes, use any target < 1 as neg loss target.
        neg_pos1 = torch.logical_and(gaussian_target.lt(1), label_mask)
        # b. For invalid classes, use union of all where gaussion target > 0.
        neg_pos2 = (
            gaussian_target.gt(0)
            .any(dim=1, keepdim=True)
            .repeat(1, gaussian_target.shape[1], 1, 1)
        )
        # c. Get union of a and b as the negative loss position mask.
        neg_pos = torch.logical_or(neg_pos1, neg_pos2)

        #      Negative loss weight, for positions in invalid class where the
        # valid class channels have positive response, use the corresponding
        # gaussian value as their weight.
        neg_weights = (1 - gaussian_target).pow(self.gamma)
        label_mask_neg = torch.logical_not(label_mask)
        gaussian_target_pos = gaussian_target.max(axis=1)[0]
        num_class = pred.shape[1]
        for bi, label_mask_neg_i in enumerate(label_mask_neg):
            num_valid_class = len(valid_classes_list[bi])
            neg_weights[bi][label_mask_neg_i] = (
                gaussian_target_pos[bi]
                .unsqueeze(0)
                .repeat(1, num_class - num_valid_class, 1, 1)
                .view(-1)
                .pow(self.gamma)  # TODO: gamma 4 is too small
            )

        pos_loss = (
            -(pred + self.eps).log()
            * (1 - pred).pow(self.alpha)
            * pos_pos
            * grad_tensor
        )
        neg_loss = (
            -(1 - pred + self.eps).log()
            * pred.pow(self.alpha)
            * neg_weights
            * neg_pos
            * grad_tensor
        )

        if heatmap_hard_mask is not None:
            neg_loss[heatmap_hard_mask.bool()] = 0

        if ig_bboxes_mask is not None:
            pos_loss[ig_bboxes_mask.bool()] = 0
            neg_loss[ig_bboxes_mask.bool()] = 0

        total_pos = pos_pos.float().sum()
        pos_factor = 1 / total_pos if total_pos > 0 else 0
        neg_factor = pos_factor
        pos_loss = pos_loss.sum((0, 2, 3))
        neg_loss = neg_loss.sum((0, 2, 3))
        loss = pos_loss * pos_factor + neg_loss * neg_factor
        return loss

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        grad_tensor: Optional[Tensor] = None,
        valid_classes_list: Optional[Tensor] = None,
        heatmap_hard_mask: Optional[Tensor] = None,
        ig_bboxes_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Forward function.

        Args:
            logits: The predicted logits before sigmoid.
            labels: The learning target of the prediction in gaussian
                distribution.
            grad_tensor: A grad scale tensor. Default is None.
            valid_classes_list: A list of list gives information about valid
                classes in instances in a batch. This information is to mask
                invalid classes's loss. Default is full class for each sample.
            heatmap_hard_mask: A mask tensor whick tells hard instance in
                heatmap. Default is None.
            ig_bboxes_mask: A mask tensor whick tells ignore regions
                in heatmap.
        """
        logits = logits.sigmoid()
        if grad_tensor is None:
            grad_tensor = torch.ones_like(logits)
        if isinstance(self.loss_weight, torch.Tensor):
            self.loss_weight = self.loss_weight.to(logits.device)
        bs, nc = logits.shape[:2]
        if valid_classes_list is None:
            valid_classes_list = [list(range(nc)) for _ in range(bs)]
        loss_reg = self.loss_weight * self.gaussian_focal_loss(
            logits,
            labels,
            valid_classes_list,
            heatmap_hard_mask,
            grad_tensor,
            ig_bboxes_mask,
        )
        loss_reg = loss_reg.sum()

        result_dict = {}
        result_dict[self.loss_name] = loss_reg
        return result_dict
