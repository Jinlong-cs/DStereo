from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn

from hat.core.face3d.lbs import batch_rodrigues
from hat.registry import OBJECT_REGISTRY

__all__ = ["Human3dLoss"]


@OBJECT_REGISTRY.register
class Human3dLoss(nn.Module):
    def __init__(
        self,
        loss_weights: Dict,
        disable_hips: bool = False,
    ):
        super().__init__()
        self.loss_weights = loss_weights
        self.disable_hips = disable_hips
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def keypoint3d_loss(
        self, pred, label, keypoint3d_conf, has_pose_3d, loss_weights
    ):
        loss = {}
        conf = keypoint3d_conf.unsqueeze(-1).clone()
        gt_keypoints_3d = label["gt_keypoint_3d"][has_pose_3d == 1]
        conf = conf[has_pose_3d == 1]
        pred_keypoints_3d = pred["pr_keypoint_3d"][has_pose_3d == 1]
        if self.disable_hips:
            conf[:, 2, :] = 0
            conf[:, 3, :] = 0

        if len(gt_keypoints_3d) > 0:
            if not self.disable_hips:
                # pelvis
                gt_root_points = (
                    gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]
                ) / 2
                pred_root_points = (
                    pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]
                ) / 2
            else:
                # nose
                gt_root_points = gt_keypoints_3d[:, 19, :]
                pred_root_points = pred_keypoints_3d[:, 19, :]
            gt_keypoints_3d = gt_keypoints_3d - gt_root_points[:, None, :]
            pred_keypoints_3d = (
                pred_keypoints_3d - pred_root_points[:, None, :]
            )
            loss["loss_keypoints_3d"] = (
                conf
                * torch.nn.MSELoss(reduction="none")(
                    pred_keypoints_3d, gt_keypoints_3d
                )
                * loss_weights["keypoint3d_loss_weight"]
            ).mean()
        else:
            loss["loss_keypoints_3d"] = (
                torch.FloatTensor(1).fill_(0.0).to(self.device)
                * loss_weights["keypoint3d_loss_weight"]
            )
        return loss

    def keypoint2d_loss(self, pred, label, keypoint2d_conf, loss_weights):
        loss = {}
        conf = keypoint2d_conf.unsqueeze(-1).clone()
        loss["loss_keypoints"] = (
            conf
            * torch.nn.MSELoss(reduction="none")(
                pred["pr_keypoint"], label["gt_keypoint"]
            )
            * loss_weights["keypoint_loss_weight"]
        ).mean()
        return loss

    def verts_loss(self, pred, label, has_smpl, loss_weights):
        loss = {}
        pred_vertices_with_shape = pred["pr_verts"][has_smpl == 1]
        gt_vertices_with_shape = label["gt_verts"][has_smpl == 1]
        if len(gt_vertices_with_shape) > 0:
            loss["loss_shape"] = (
                torch.nn.L1Loss()(
                    pred_vertices_with_shape, gt_vertices_with_shape
                )
                * loss_weights["shape_loss_weight"]
            )
        else:
            loss["loss_shape"] = (
                torch.FloatTensor(1).fill_(0.0).to(self.device)
                * loss_weights["shape_loss_weight"]
            )
        return loss

    def beta_loss(self, pred, label, has_smpl, loss_weights):
        loss = {}
        pred_betas_valid = pred["pr_betas"][has_smpl == 1]
        gt_betas_valid = label["gt_betas"][has_smpl == 1]
        if len(gt_betas_valid) > 0:
            loss["loss_regr_betas"] = (
                torch.nn.MSELoss()(pred_betas_valid, gt_betas_valid)
                * loss_weights["beta_loss_weight"]
            )
        else:
            loss["loss_regr_betas"] = (
                torch.FloatTensor(1).fill_(0.0).to(self.device)
                * loss_weights["beta_loss_weight"]
            )
        return loss

    def pose_loss(self, pred, label, has_smpl, loss_weights):
        loss = {}
        pred_rotmat_valid = pred["pr_pose"][has_smpl == 1]
        gt_rotmat_valid = batch_rodrigues(
            label["gt_pose"].view(-1, 3), 1e-8
        ).view(-1, 24, 3, 3)[has_smpl == 1]
        if len(gt_rotmat_valid) > 0:
            loss["loss_regr_pose"] = (
                torch.nn.MSELoss()(pred_rotmat_valid, gt_rotmat_valid)
                * loss_weights["pose_loss_weight"]
            )
        else:
            loss["loss_regr_pose"] = (
                torch.FloatTensor(1).fill_(0.0).to(self.device)
                * loss_weights["pose_loss_weight"]
            )
        return loss

    def cam_loss(self, pred, loss_weights):
        loss = {}
        loss["loss_regr_cam"] = (
            (torch.exp(-pred["pr_cam"][:, 0] * 10)) ** 2
        ).mean() * loss_weights["cam_loss_weight"]
        return loss

    def forward(
        self,
        data,
        keypoint2d_conf,
        keypoint3d_conf,
        has_pose_3d,
        has_smpl,
        loss_weights=None,
    ):
        if loss_weights is None:
            loss_weights = self.loss_weights

        pred = data["pred"]
        label = data["label"]

        loss = OrderedDict()
        loss.update(
            self.keypoint2d_loss(pred, label, keypoint2d_conf, loss_weights)
        )
        loss.update(
            self.keypoint3d_loss(
                pred, label, keypoint3d_conf, has_pose_3d, loss_weights
            )
        )
        loss.update(self.verts_loss(pred, label, has_smpl, loss_weights))
        loss.update(self.beta_loss(pred, label, has_smpl, loss_weights))
        loss.update(self.pose_loss(pred, label, has_smpl, loss_weights))
        loss.update(self.cam_loss(pred, loss_weights))

        return loss
