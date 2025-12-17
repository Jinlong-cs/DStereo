from collections import OrderedDict
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from hat.registry import OBJECT_REGISTRY
from .utils import create_meshgrid, get_mask, norm_pts, pupil_center_from_mask

__all__ = ["PupilSegLoss"]


@OBJECT_REGISTRY.register
class PupilSegLoss(nn.Module):
    """Pupil segmentation loss module.

    Args:
        loss_weights: Dict of loss weight.
        do_self_corr: Whether to add self-correlation loss. Defaults to False.
        mask_shape: Shape of mask, (H, W) or [H, W]. Defaults to None.
    """

    def __init__(
        self,
        loss_weights: Dict,
        do_self_corr: bool = False,
        mask_shape: Optional[Sequence] = None,
    ):
        super().__init__()
        self.loss_weights = loss_weights
        self.do_self_corr = do_self_corr
        self.mask_shape = mask_shape
        if self.do_self_corr:
            assert (
                self.mask_shape is not None
            ), "if `do_self_corr` is true, then mask_shape cannot be None!"
            H, W = self.mask_shape
            self.mesh = create_meshgrid(H, W, normalized_coordinates=True)
            self.mesh.requires_grad = False

    def get_seg_loss(self, mask_pred, mask_gt, spat_weights, dist_map, alpha):
        """Measure the error between mask_pred and mask_gt."""
        B = mask_pred.shape[0]
        l_sl = self.surface_loss(torch.sigmoid(mask_pred), dist_map)
        l_ce = F.binary_cross_entropy_with_logits(
            mask_pred.view(B, -1),
            mask_gt.view(B, -1).float(),
            weight=spat_weights.view(B, -1),
            reduction="mean",
        )
        l_gd = self.gdice_loss(torch.sigmoid(mask_pred), mask_gt)
        loss_seg = alpha * l_sl + (1 - alpha) * l_gd + l_ce
        return loss_seg

    def surface_loss(self, mask, dist_map):
        r"""Boundary loss, which measure the distance from each point to the boundary.

        For classes with no groundtruth,
        dist_map would ideally be filled with 0s.

        Reference:
            Boundary loss for highly unbalanced segmentation

        Formula:
            L = \sum_{mask_{n} * dist_map_{n}},
            where n represents the position of the pixel,
            and dist_map represents the normalized distance
            from the pixel point to the pupil boundary.

        """
        score = mask.flatten(start_dim=1) * dist_map.flatten(start_dim=1)
        score = torch.mean(score)
        return score

    def gdice_loss(self, mask_pred, mask_gt):
        r"""Generalised Dice Loss (common segmentation loss).

        Reference:
            Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations # noqa

        Formula:
            L = 1-2\frac{\sum_{l=1}^Lw_l\sum_nr_{ln}p_{ln}}{\sum_{l=1}^Lw_l\sum_nr_{ln}p_{ln}+p_{ln}},  # noqa
            where w is the weight for each category,
            r_{ln} is the gt of class l in the pixel n,
            and p_{ln} is the predicted value.

        """
        bg_pred = torch.ones_like(mask_pred) - mask_pred
        bg_gt = torch.ones_like(mask_gt) - mask_gt
        mask_pred = torch.flatten(mask_pred, start_dim=1, end_dim=-1)
        bg_pred = torch.flatten(bg_pred, start_dim=1, end_dim=-1)
        mask_gt = torch.flatten(mask_gt, start_dim=1, end_dim=-1)
        bg_gt = torch.flatten(bg_gt, start_dim=1, end_dim=-1)
        pupil_weight = 1.0 / (torch.sum(mask_gt, dim=1) ** 2).clamp(1e-5)
        bg_weight = 1.0 / (torch.sum(bg_gt, dim=1) ** 2).clamp(1e-5)
        numerator_pupil = pupil_weight * torch.sum(mask_pred * mask_gt, dim=1)
        numerator_bg = bg_weight * torch.sum(bg_pred * bg_gt, dim=1)
        denominator_pupil = pupil_weight * torch.sum(
            mask_pred + mask_gt, dim=1
        )
        denominator_bg = bg_weight * torch.sum(bg_pred + bg_gt, dim=1)
        dice_metric = (
            2.0
            * (numerator_pupil + numerator_bg)
            / (denominator_pupil + denominator_bg)
        )
        return torch.mean(1 - dice_metric.clamp(1e-5))

    def self_consistency_loss(self, mask, ellipse_param):
        r"""Correction loss based on self consistency KL divergence.

        Measure the consistency of the mask predicted by the decoder branch
        and the ellipse parameters predicted by the regression branch


        Formula:
            L = \sum_nP(n)ln(\frac{P(n)}{Q(n)}),
            where n represents the position of the pixel,
            P(n) is the predicted value of the decoder branch,
            Q(n) is the predicted value converted from the
            elliptic parameter predicted by the regression branch.

        """
        mask = F.logsigmoid(mask)
        mesh = self.mesh.to(mask.device)
        pup_mask = get_mask(mesh, ellipse_param)[1]
        loss = F.kl_div(mask, pup_mask, reduction="mean")
        return loss

    def forward(self, model_out: dict, data: dict):
        alpha = self.loss_weights["surface_loss_ratio"]
        assert (
            alpha >= 0 and alpha <= 1
        ), "`surface_loss_ratio` must be between 0 and 1!"
        loss = OrderedDict()
        mask_pred = model_out["mask_pred"]
        mask_pred = mask_pred.squeeze(1)
        ellipse_param_pred = model_out["ellipse_param_pred"]
        gt_pupil_center = data["gt_pupil_center"]
        gt_norm_pupil_ellipse_param = data["gt_norm_pupil_ellipse_param"]
        gt_pupil_mask = data["gt_pupil_mask"]
        spat_weights = data["spat_weights"]
        dist_map = data["dist_map"]

        # process the pred ellipse param
        bc = ellipse_param_pred.shape[0]
        ellipse_param_pred = ellipse_param_pred.reshape(bc, -1)
        pup_c = torch.tanh(ellipse_param_pred[:, 0:2])
        pup_param = torch.sigmoid(ellipse_param_pred[:, 2:4])
        pup_angle = ellipse_param_pred[:, 4]
        ellipse_param_pred = torch.cat(
            [pup_c, pup_param, pup_angle.unsqueeze(1)], dim=1
        )

        # seg2pt loss
        pupil_mask_center = pupil_center_from_mask(mask_pred, temperature=4)
        seg2pt_loss = torch.mean(
            F.l1_loss(
                pupil_mask_center,
                norm_pts(gt_pupil_center, gt_pupil_mask.shape[1:]),
                reduction="none",
            )
        )

        # ellipse loss
        ellipse_loss = F.l1_loss(
            ellipse_param_pred,
            gt_norm_pupil_ellipse_param.view(-1, 5),
            reduction="mean",
        )

        # segmentation loss
        seg_loss = self.get_seg_loss(
            mask_pred, gt_pupil_mask, spat_weights, dist_map, alpha
        )

        loss.update({"seg2pt_loss": seg2pt_loss * self.loss_weights["seg2pt"]})
        loss.update(
            {"ellipse_loss": ellipse_loss * self.loss_weights["ellipse"]}
        )
        loss.update({"seg_loss": seg_loss * self.loss_weights["seg"]})

        if self.do_self_corr:
            # Uses ellipse center from segmentation but other params from regression # noqa
            el_pred = torch.cat(
                [pupil_mask_center, ellipse_param_pred[:, 2:5]], dim=1
            )
            selfcorr_loss = self.self_consistency_loss(mask_pred, el_pred)
            loss.update(
                {
                    "selfcorr_loss": selfcorr_loss
                    * self.loss_weights["selfcorr"]
                }
            )
        return loss
