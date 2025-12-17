# For the GMM prior, we use the GMM implementation of SMPLify-X
# https://github.com/vchoutas/smplify-x/blob/master/smplifyx/prior.py
from typing import Dict

import torch
import torch.nn as nn

from hat.models.task_modules.human3d.smplify.losses import (
    body_fitting_loss,
    camera_fitting_loss,
)
from hat.models.task_modules.human3d.smplify.prior import MaxMixturePrior
from hat.models.task_modules.human3d.utils import (
    human3d_constants as constants,
)
from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class SMPLify:
    """Implementation of single-stage SMPLify.

    Args:
        prior_folder: Path of prior file.
        smpl_model: Class of smpl module.
        loss_weights: Loss weight dict of each item. Like:
            loss_weights={
                "pose_prior_weight": 4.78,
                "shape_prior_weight": 5,
                "angle_prior_weight": 15.2,
            }
        camera_step_size: Number of camera trans fitting step.
            Defaults to 1e-2.
        body_step_size: Number of body fitting step. Defaults to 1e-2.
        camera_num_iters: Iter number of camera trans. Defaults to 100.
        body_num_iters: Iter number of body. Defaults to 100.
        focal_length: Focal length. Defaults to 5000.
        use_depth_loss: Whether use depth loss when fitting camera trans.
            This will lead camera's depth close to the estimated depth.
            Defaults to True.
        device: Device to use, cpu or cuda. Defaults to cuda.
    """

    def __init__(
        self,
        prior_folder: str,
        smpl_model: nn.Module,
        loss_weights: Dict,
        camera_step_size: float = 1e-2,
        body_step_size: float = 1e-2,
        camera_num_iters: int = 100,
        body_num_iters: int = 100,
        focal_length: int = 5000,
        use_depth_loss: bool = True,
        device: str = "cuda",
    ):
        assert device in ["cpu", "cuda"]
        # Store options
        self.loss_weights = loss_weights
        self.device = torch.device(device)
        self.focal_length = focal_length
        self.camera_step_size = camera_step_size
        self.body_step_size = body_step_size
        self.prior_folder = prior_folder
        self.use_depth_loss = use_depth_loss

        # Ignore the the following joints for the fitting process
        ign_joints = ["Right Hip", "Left Hip"]
        self.ign_joints = [constants.JOINT_IDS[i] for i in ign_joints]
        self.camera_num_iters = camera_num_iters
        self.body_num_iters = body_num_iters
        # GMM pose prior
        self.pose_prior = MaxMixturePrior(
            prior_folder=self.prior_folder,
            num_gaussians=8,
            dtype=torch.float32,
        ).to(self.device)
        # Load SMPL model
        self.smpl = smpl_model.to(self.device)

    def __call__(
        self,
        init_pose,
        init_betas,
        init_cam_t,
        camera_center,
        keypoints_2d,
        keypoints_2d_conf,
    ):
        """Perform body fitting.

        Input:
            init_pose: SMPL pose estimate
            init_betas: SMPL betas estimate
            init_cam_t: Camera translation estimate
            camera_center: Camera center location
            keypoints_2d: Keypoints used for the optimization
        Returns:
            vertices: Vertices of optimized shape
            joints: 3D joints of optimized shape
            pose: SMPL pose parameters of optimized shape
            betas: SMPL beta parameters of optimized shape
            camera_translation: Camera translation
            reprojection_loss: Final joint reprojection loss
        """

        # Make camera translation a learnable parameter
        camera_translation = init_cam_t.clone()

        # Get joint confidence
        joints_2d = keypoints_2d
        joints_conf = keypoints_2d_conf

        # Split SMPL pose to body pose and global orientation
        body_pose = init_pose[:, 3:].detach().clone()
        global_orient = init_pose[:, :3].detach().clone()
        betas = init_betas.detach().clone()

        # Step 1: Optimize camera translation and body orientation
        # Optimize only camera translation and body orientation
        body_pose.requires_grad = False
        betas.requires_grad = False
        global_orient.requires_grad = True
        camera_translation.requires_grad = True

        camera_opt_params = [global_orient, camera_translation]
        camera_optimizer = torch.optim.Adam(
            camera_opt_params, lr=self.camera_step_size, betas=(0.9, 0.999)
        )

        for i in range(self.camera_num_iters):  # noqa B007
            smpl_output = self.smpl(
                global_orient=global_orient, body_pose=body_pose, betas=betas
            )
            model_joints = smpl_output.joints
            loss = camera_fitting_loss(
                model_joints,
                camera_translation,
                init_cam_t,
                camera_center,
                joints_2d,
                joints_conf,
                focal_length=self.focal_length,
                use_depth_loss=self.use_depth_loss,
            )
            camera_optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            camera_optimizer.step()

        # Fix camera translation after optimizing camera
        camera_translation.requires_grad = False

        # Step 2: Optimize body joints
        # Optimize only the body pose and global orientation of the body
        body_pose.requires_grad = True
        betas.requires_grad = True
        global_orient.requires_grad = True
        camera_translation.requires_grad = False
        body_opt_params = [body_pose, betas, global_orient]

        # For joints ignored during fitting, set the confidence to 0
        joints_conf[:, self.ign_joints] = 0.0

        body_optimizer = torch.optim.Adam(
            body_opt_params, lr=self.body_step_size, betas=(0.9, 0.999)
        )
        for i in range(self.body_num_iters):  # noqa B007
            smpl_output = self.smpl(
                global_orient=global_orient, body_pose=body_pose, betas=betas
            )
            model_joints = smpl_output.joints
            loss = body_fitting_loss(
                body_pose,
                betas,
                model_joints,
                camera_translation,
                camera_center,
                joints_2d,
                joints_conf,
                self.pose_prior,
                focal_length=self.focal_length,
                pose_prior_weight=self.loss_weights["pose_prior_weight"],
                shape_prior_weight=self.loss_weights["shape_prior_weight"],
                angle_prior_weight=self.loss_weights["angle_prior_weight"],
            )
            body_optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            body_optimizer.step()

        # Get final loss value
        with torch.no_grad():
            smpl_output = self.smpl(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas,
                return_full_pose=True,
            )
            model_joints = smpl_output.joints

            reprojection_loss = body_fitting_loss(
                body_pose,
                betas,
                model_joints,
                camera_translation,
                camera_center,
                joints_2d,
                joints_conf,
                self.pose_prior,
                focal_length=self.focal_length,
                output="reprojection",
            )

        vertices = smpl_output.vertices.detach()
        joints = smpl_output.joints.detach()
        pose = torch.cat([global_orient, body_pose], dim=-1).detach()
        betas = betas.detach()

        return (
            vertices,
            joints,
            pose,
            betas,
            camera_translation,
            reprojection_loss,
        )

    def get_fitting_loss(
        self,
        pose,
        betas,
        cam_t,
        camera_center,
        keypoints_2d,
        keypoints_2d_conf,
    ):
        """Given body and camera parameters, compute reprojection loss value.

        Input:
            pose: SMPL pose parameters
            betas: SMPL beta parameters
            cam_t: Camera translation
            camera_center: Camera center location
            keypoints_2d: Keypoints used for the optimization
        Returns:
            reprojection_loss: Final joint reprojection loss
        """

        # Get joint confidence
        joints_2d = keypoints_2d
        joints_conf = keypoints_2d_conf
        # For joints ignored during fitting, set the confidence to 0
        joints_conf[:, self.ign_joints] = 0.0

        # Split SMPL pose to body pose and global orientation
        body_pose = pose[:, 3:]
        global_orient = pose[:, :3]

        with torch.no_grad():
            smpl_output = self.smpl(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas,
                return_full_pose=True,
            )
            model_joints = smpl_output.joints
            reprojection_loss = body_fitting_loss(
                body_pose,
                betas,
                model_joints,
                cam_t,
                camera_center,
                joints_2d,
                joints_conf,
                self.pose_prior,
                focal_length=self.focal_length,
                output="reprojection",
            )

        return reprojection_loss
