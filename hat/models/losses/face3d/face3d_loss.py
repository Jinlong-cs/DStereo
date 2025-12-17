from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from hat.registry import OBJECT_REGISTRY

__all__ = ["Face3dLoss"]


@OBJECT_REGISTRY.register
class Face3dLoss(nn.Module):
    """Constaints for face3d reconstruction.

    Args:
        loss_weights: dict of loss weight.
        lpips: lpips module for evaluating face similarity. Defaults to None.
        consistency_nums: shape consistency constraints. The sample will be
            copied and fed into different augmentations when it is more than 1.
            Defaults to 1.
    """

    def __init__(
        self,
        loss_weights: Dict,
        lpips: Optional[nn.Module] = None,
        consistency_nums: int = 1,
    ):
        super().__init__()
        self.lpips = lpips
        self.loss_weights = loss_weights
        self.consistency_nums = consistency_nums
        if self.lpips is not None:
            for param in self.lpips.parameters():
                param.requires_grad = False

    def img_ldmk_loss(self, pred, label, loss_weights):
        """Landmark loss in image space."""
        loss = {}
        if loss_weights["img_ldmk"] > 0:
            pr_ldmk = pred["img_ldmk"][..., :2]
            gt_ldmk = label["gt_ldmk"][..., :2]
            loss["img_ldmk"] = (
                F.smooth_l1_loss(pr_ldmk, gt_ldmk) * loss_weights["img_ldmk"]
            )
        return loss

    def cam_verts_loss(self, pred, label, loss_weights):
        """Vertices loss in camera space."""
        loss = {}
        if loss_weights["cam_verts"] > 0:
            pr_verts = pred["real_verts"]
            gt_verts = label["gt_verts"]
            loss["cam_verts"] = (
                F.smooth_l1_loss(pr_verts, gt_verts)
                * loss_weights["cam_verts"]
            )
        return loss

    def cam_ldmk_loss(self, pred, label, loss_weights):
        """3D landmark loss in camera space."""
        loss = {}
        if loss_weights["cam_ldmk"] > 0:
            pr_cam_ldmk = pred["cam_ldmk"]
            gt_cam_ldmk = label["gt_cam_ldmk"]
            loss["cam_ldmk"] = (
                F.smooth_l1_loss(pr_cam_ldmk, gt_cam_ldmk)
                * loss_weights["cam_ldmk"]
            )
        return loss

    def latent_loss(self, pred, label, loss_weights):
        """3dmm coefficient loss."""
        loss = []
        for key in [
            "global_pose",
            "transl",
            "shape",
            "jaw",
            "exp",
            "light",
            "tex",
        ]:
            if loss_weights[key] > 0:
                loss.append(
                    F.l1_loss(pred[key], label[key]) * loss_weights[key]
                )
        loss = sum(loss)
        return {"flame": loss} if loss > 0 else {}

    def photometric_loss(self, pred, label, loss_weights):
        """Pixel-wise reconstructed image loss."""
        loss = {}
        if loss_weights["photo"] > 0:
            pr_img = pred["pr_img"]
            gt_img = label["gt_img"]
            mask = pred["pr_mask"]  # pr_mask = render_mask * gt_mask
            loss["photo"] = (
                (pr_img - gt_img).abs().sum()
                / (mask.sum() + 1e-6)
                * loss_weights["photo"]
            )
        return loss

    def depth_loss(self, pred, label, loss_weights):
        """Pixel-wise reconstructed depth image loss."""
        loss = {}
        if loss_weights["depth"] > 0:
            pr_depth = pred["pr_depth"]
            gt_depth = label["gt_depth"]
            mask = pred["pr_mask"]
            loss["depth"] = (
                (gt_depth * mask - pr_depth).abs().sum()
                / (mask.sum() + 1e-6)
                * loss_weights["depth"]
            )
        return loss

    def perception_loss(self, pred, label, loss_weights):
        """Image similarity loss for gt_img and reconstructed image."""
        loss = {}
        if loss_weights["lpips"] > 0:
            pr_img = pred["pr_img"]
            gt_img = label["gt_img"]
            loss["lpips"] = (
                self.lpips(pr_img, gt_img, normalize=True).mean()
                * loss_weights["lpips"]
            )
        return loss

    def regularization(self, data, loss_weights):
        """Face priors for regularization."""
        loss = OrderedDict()
        if loss_weights["shape_reg"] > 0:
            loss["shape_reg"] = (
                torch.norm(data["shape"], dim=1).mean()
                * loss_weights["shape_reg"]
            )
        if loss_weights["exp_reg"] > 0:
            loss["exp_reg"] = (
                torch.norm(data["exp"], dim=1).mean() * loss_weights["exp_reg"]
            )
        if loss_weights["tex_reg"] > 0:
            loss["tex_reg"] = (
                torch.norm(data["tex"], dim=1).mean() * loss_weights["exp_reg"]
            )
        return loss

    def shape_consistency_loss(self, data, loss_weights):
        """Keep shape consistency on the same sample with different augs."""
        loss = {}
        if loss_weights["shape_cons"] > 0:
            loss_shape = 0
            batch_size = data["shape"].shape[0]
            shape = data["shape"].reshape(
                batch_size // self.consistency_nums, self.consistency_nums, -1
            )
            for cons_num_i in range(1, self.consistency_nums):
                shape_cons = (
                    F.l1_loss(shape[:, 0, :], shape[:, cons_num_i, :])
                    * loss_weights["shape_cons"]
                )
                loss_shape += shape_cons
            loss["shape_cons"] = loss_shape

        return loss

    def eye3d_loss(self, pred, label, loss_weights):
        """Eye 3D position loss in mm."""
        loss = {}
        if loss_weights["eye3d"] > 0:
            loss_eye3d_left = F.smooth_l1_loss(
                pred["real_eye3d_left"], label["eye3d_left"]
            )
            loss_eye3d_right = F.smooth_l1_loss(
                pred["real_eye3d_right"], label["eye3d_right"]
            )
            loss["eye3d"] = (
                loss_eye3d_left + loss_eye3d_right
            ) * loss_weights["eye3d"]
        return loss

    def eyelid_loss(self, pred, label, loss_weights):  # noqa
        """Eyelid loss in DECA.

        Refer to https://arxiv.org/abs/2012.04012
        """
        # TODO: Add eyelid loss from DECA
        return {}

    def forward(self, data, loss_weights=None):
        if loss_weights is None:
            loss_weights = self.loss_weights

        pred = data["pred"]
        label = data["label"]

        loss = OrderedDict()
        loss.update(self.img_ldmk_loss(pred, label, loss_weights))
        loss.update(self.cam_ldmk_loss(pred, label, loss_weights))
        loss.update(self.eye3d_loss(pred, label, loss_weights))
        loss.update(self.cam_verts_loss(pred, label, loss_weights))
        loss.update(self.latent_loss(pred, label, loss_weights))
        loss.update(self.photometric_loss(pred, label, loss_weights))
        loss.update(self.perception_loss(pred, label, loss_weights))
        loss.update(self.shape_consistency_loss(pred, loss_weights))
        loss.update(self.regularization(pred, loss_weights))
        loss.update(self.eyelid_loss(pred, label, loss_weights))

        total_loss = 0
        for _, v in loss.items():
            total_loss += v
        loss["total_loss"] = total_loss
        return loss
