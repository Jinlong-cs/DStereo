import logging
from typing import Dict

import torch
import torch.nn as nn

try:
    from kornia.geometry.conversions import rotation_matrix_to_angle_axis
except ImportError:
    pass

from hat.core.face3d.lbs import rot6d_to_rotmat
from hat.models.task_modules.face3d.geometry import batch_orth_proj
from hat.models.task_modules.human3d.utils import FitsDict
from hat.models.task_modules.human3d.utils.geometry import (
    estimate_translation,
    perspective_projection,
)
from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages

__all__ = ["SPIN"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class SPIN(nn.Module):
    """Structure of SPIN.

    Args:
        model: Model used in SPIN.
        mode: "train", "val" mode. In train mode, return
            loss. In val mode, return prediction for evaluation.
        smpl: SMPL module.
        shape_loss: shape loss function.
        keypoint_3d_loss: 3d keypoints loss function.
        smpl_loss: SMPL loss function.
        keypoint_loss: 2d keypoint loss function.
        smplify: SMPLify module.
        img_res: Img resolution.
        loss_weights: loss weight about shape loss, keypoint loss,
            keypoint 3d loss, smpl loss(pose loss and beta loss), cam loss.
            Dict like
            {
                "shape_loss_weight": 0,
                "keypoint_loss_weight": 5,
                "keypoint3d_loss_weight": 5,
                "pose_loss_weight": 1.,
                "beta_loss_weight": 0.001,
                "cam_loss_weight: 1.,
            }
        smplify_threshold: Threshold about whether a fit is valid.
            Defaults to 100.
        focal_length: Camera focal length. Defaults to 5000..
        run_smplify: Whether to run SMPLify. Defaults to True.
        use_weakproj: Whether convert perspective projection to
            weak projection. Defaults to False.
        gt_train_weight: Training weight about gt keypoints.
            Defaults to 1..
        fits_dict: Class FitsDict, Use to save and load the fitting results.
            Defaults to None.
    """

    @require_packages("kornia")
    def __init__(
        self,
        model: nn.Module,
        mode: str,
        smpl: nn.Module,
        human3d_loss: nn.Module,
        smplify: nn.Module,
        img_res: int,
        loss_weights: Dict[str, float] = None,
        smplify_threshold: float = 100.0,
        focal_length: float = 5000.0,
        run_smplify: bool = True,
        use_weakproj: bool = False,
        gt_train_weight: float = 1.0,
        fits_dict: FitsDict = None,
    ):
        super().__init__()
        assert mode in ["train", "val", "deploy"]
        self.model = model
        self.mode = mode
        self.smpl = smpl
        self.human3d_loss = human3d_loss
        self.focal_length = focal_length
        self.img_res = img_res
        self.loss_weights = loss_weights
        self.smplify_threshold = smplify_threshold
        self.run_smplify = run_smplify
        self.use_weakproj = use_weakproj
        self.gt_train_weight = gt_train_weight

        self.smplify = smplify

        self.fits_dict = fits_dict

        if mode == "train":
            assert (
                fits_dict is not None
            ), "fits dict is necessary when in train mode"

    def finalize(self):
        self.fits_dict.save()

    def forward(self, data):
        img = data["img"]
        batch_size = img.shape[0]

        # Feed images in the network to predict camera and SMPL parameters
        pred_pose, pred_betas, pred_camera = self.model(img)

        if self.mode == "train":
            # 2D keypoints
            gt_keypoints_2d = data["gt_ldmk"]
            # 2D keypoints validity
            gt_keypoints_2d_attr = data["gt_ldmk_attr"]
            # SMPL pose parameters
            gt_pose = data["gt_smpl_pose"]
            # SMPL beta parameters
            gt_betas = data["gt_smpl_betas"]
            # 3D pose
            gt_joints = data["gt_ldmk_3d"]
            # 3D pose validity
            gt_joints_attr = data["gt_ldmk_3d_attr"]
            # flag that indicates whether SMPL parameters are valid
            has_smpl = data["has_smpl"].byte()
            # flag that indicates whether 3D pose is valid
            has_pose_3d = data["has_ldmk_3d"].byte()
            # flag that indicates whether image was flipped
            is_flipped = data["is_flipped"]
            # rotation angle used for data augmentation
            rot_angle = data["rot_angle"]
            # name of the dataset the image comes from
            dataset_name = data["dataset_name"]
            # index of example inside its dataset
            indices = data["sample_index"]

            pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

            pred_output = self.smpl(
                betas=pred_betas,
                body_pose=pred_rotmat[:, 1:],
                global_orient=pred_rotmat[:, 0].unsqueeze(1),
                pose2rot=False,
            )
            pred_vertices = pred_output.vertices
            pred_joints = pred_output.joints

            # Convert Weak Perspective Camera [s, tx, ty] to
            # camera translation [tx, ty, tz].
            pred_cam_t = torch.stack(
                [
                    pred_camera[:, 1],
                    pred_camera[:, 2],
                    2
                    * self.focal_length
                    / (self.img_res * pred_camera[:, 0] + 1e-9),
                ],
                dim=-1,
            )

            camera_center = torch.zeros(batch_size, 2, device=img.device)

            if self.use_weakproj:
                pred_keypoints_2d = batch_orth_proj(pred_joints, pred_camera)
            else:
                pred_keypoints_2d = perspective_projection(
                    pred_joints,
                    rotation=torch.eye(3, device=pred_joints.device)
                    .unsqueeze(0)
                    .expand(batch_size, -1, -1),
                    translation=pred_cam_t,
                    focal_length=self.focal_length,
                    camera_center=camera_center,
                )
                # Normalize keypoints to [-1,1]
                pred_keypoints_2d = pred_keypoints_2d / (self.img_res / 2.0)

            # Get GT vertices and model joints.
            # Note that gt_model_joints is different from
            # gt_joints as it comes from SMPL.
            gt_out = self.smpl(
                betas=gt_betas,
                body_pose=gt_pose[:, 3:],
                global_orient=gt_pose[:, :3],
            )
            gt_model_joints = gt_out.joints
            gt_vertices = gt_out.vertices

            # Get current best fits from the dictionary
            opt_pose, opt_betas, opt_validity = self.fits_dict[
                (
                    dataset_name,
                    indices.cpu(),
                    rot_angle.cpu(),
                    is_flipped.cpu(),
                )
            ]
            opt_pose = opt_pose.to(img.device)
            opt_betas = opt_betas.to(img.device)
            opt_output = self.smpl(
                betas=opt_betas,
                body_pose=opt_pose[:, 3:],
                global_orient=opt_pose[:, :3],
            )
            opt_vertices = opt_output.vertices
            opt_joints = opt_output.joints
            # assume that non valid opt has GT values
            if len(has_smpl[opt_validity == 0]) > 0 and self.run_smplify:
                assert min(
                    has_smpl[opt_validity == 0]
                ), "All opt params should be True."

            # De-normalize 2D keypoints from [-1,1] to pixel space
            gt_keypoints_2d_orig = gt_keypoints_2d.clone()
            gt_keypoints_2d_orig = (
                0.5 * self.img_res * (gt_keypoints_2d_orig + 1)
            )

            # Estimate camera translation given the model joints and
            # 2D keypoints by minimizing a weighted least squares loss.
            # gt_cam_t = estimate_translation(
            #     gt_model_joints,
            #     gt_keypoints_2d_orig,
            #     gt_keypoints_2d_attr,
            #     focal_length=self.focal_length,
            #     img_size=self.img_res,
            # )

            if self.run_smplify:
                # Convert predicted rotation matrices to axis-angle
                # N * 3 * 3
                pred_rotmat_ = pred_rotmat.detach().view(-1, 3, 3).detach()
                pred_pose = (
                    rotation_matrix_to_angle_axis(pred_rotmat_)
                    .contiguous()
                    .view(batch_size, -1)
                )
                pred_pose[torch.isnan(pred_pose)] = 0.0

                # Run SMPLify optimization starting from the network prediction
                (
                    new_opt_vertices,
                    new_opt_joints,
                    new_opt_pose,
                    new_opt_betas,
                    new_opt_cam_t,
                    new_opt_joint_loss,
                ) = self.smplify(
                    pred_pose.detach(),
                    pred_betas.detach(),
                    pred_cam_t.detach(),
                    0.5
                    * self.img_res
                    * torch.ones(batch_size, 2, device=pred_pose.device),
                    gt_keypoints_2d_orig,
                    gt_keypoints_2d_attr,
                )
                new_opt_joint_loss = new_opt_joint_loss.mean(dim=-1)

                # Will update the dictionary for the examples
                # where the new loss is less than the current one.
                opt_cam_t = estimate_translation(
                    opt_joints,
                    gt_keypoints_2d_orig,
                    gt_keypoints_2d_attr,
                    focal_length=self.focal_length,
                    img_size=self.img_res,
                )
                opt_joint_loss = self.smplify.get_fitting_loss(
                    opt_pose,
                    opt_betas,
                    opt_cam_t,
                    0.5
                    * self.img_res
                    * torch.ones(batch_size, 2, device=opt_pose.device),
                    gt_keypoints_2d_orig,
                    gt_keypoints_2d_attr,
                ).mean(dim=-1)
                update = new_opt_joint_loss < opt_joint_loss

                opt_joint_loss[update] = new_opt_joint_loss[update]
                opt_vertices[update, :] = new_opt_vertices[update, :]
                opt_joints[update, :] = new_opt_joints[update, :]
                opt_pose[update, :] = new_opt_pose[update, :]
                opt_betas[update, :] = new_opt_betas[update, :]
                # opt_cam_t[update, :] = new_opt_cam_t[update, :]

                self.fits_dict[
                    (
                        dataset_name,
                        indices.cpu(),
                        rot_angle.cpu(),
                        is_flipped.cpu(),
                        update.cpu(),
                    )
                ] = (
                    opt_pose.cpu(),
                    opt_betas.cpu(),
                )

                self.finalize()

            # Replace extreme betas with zero betas
            opt_betas[(opt_betas.abs() > 3).any(dim=-1)] = 0.0

            # Replace the optimized parameters with the
            # ground truth parameters, if available.
            opt_vertices[has_smpl, :, :] = gt_vertices[has_smpl, :, :]
            # opt_cam_t[has_smpl, :] = gt_cam_t[has_smpl, :]
            opt_joints[has_smpl, :, :] = gt_model_joints[has_smpl, :, :]
            opt_pose[has_smpl, :] = gt_pose[has_smpl, :]
            opt_betas[has_smpl, :] = gt_betas[has_smpl, :]

            # Assert whether a fit is valid by comparing the
            # joint loss with the threshold.
            if self.run_smplify:
                valid_fit = (opt_joint_loss < self.smplify_threshold).to(
                    img.device
                )
            else:
                valid_fit = torch.zeros_like(
                    has_smpl, device=img.device, dtype=torch.bool
                )
            # Add the examples with GT parameters to the list of valid fits
            valid_fit = valid_fit | has_smpl

            data["pred"] = {
                "pr_cam": pred_camera,
                "pr_pose": pred_rotmat,
                "pr_betas": pred_betas,
                "pr_verts": pred_vertices,
                "pr_keypoint": pred_keypoints_2d,
                "pr_keypoint_3d": pred_joints,
            }
            data["label"] = {
                "gt_pose": opt_pose,
                "gt_betas": opt_betas,
                "gt_verts": opt_vertices,
                "gt_keypoint": gt_keypoints_2d,
                "gt_keypoint_3d": gt_joints,
            }
            loss = self.human3d_loss(
                data,
                gt_keypoints_2d_attr,
                gt_joints_attr,
                has_pose_3d,
                valid_fit,
                self.loss_weights,
            )

        # Pack output arguments
        if self.mode == "train":
            outputs = loss
        elif self.mode == "val":
            outputs = {
                "pred_pose": pred_pose,
                "pred_betas": pred_betas,
                "pred_camera": pred_camera,
            }
        else:
            outputs = {
                "pred_pose": pred_pose,
                "pred_betas": pred_betas,
                "pred_camera": pred_camera,
            }
            outputs = list(outputs.values())
        return outputs

    def fuse_model(self):
        if self.model is not None and hasattr(self.model, "fuse_model"):
            self.model.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        if self.model is not None and hasattr(self.model, "set_qconfig"):
            self.model.set_qconfig()
