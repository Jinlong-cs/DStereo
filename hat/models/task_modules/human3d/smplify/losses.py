import torch

from hat.models.task_modules.human3d.utils import (
    human3d_constants as constants,
)
from hat.models.task_modules.human3d.utils.geometry import (
    perspective_projection,
)


def gmof(x, sigma):
    """Geman-McClure error function."""
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def angle_prior(pose):
    """Angle prior that penalizes unnatural bending of the knees and elbows."""

    # 55: left elbow
    # 58: right elbow
    # 12: left knee
    # 15: right knee
    # We subtract 3 because pose does not
    # include the global rotation of the model
    # Indices for the roration angle of
    return (
        torch.exp(
            pose[:, [55 - 3, 58 - 3, 12 - 3, 15 - 3]]
            * torch.tensor([1.0, -1.0, -1, -1.0], device=pose.device)
        )
        ** 2
    )


def body_fitting_loss(
    body_pose,
    betas,
    model_joints,
    camera_t,
    camera_center,
    joints_2d,
    joints_conf,
    pose_prior,
    focal_length=5000,
    sigma=100,
    pose_prior_weight=4.78,
    shape_prior_weight=5,
    angle_prior_weight=15.2,
    output="sum",
):
    """Loss function for body fitting."""

    batch_size = body_pose.shape[0]
    rotation = (
        torch.eye(3, device=body_pose.device)
        .unsqueeze(0)
        .expand(batch_size, -1, -1)
    )
    projected_joints = perspective_projection(
        model_joints, rotation, camera_t, focal_length, camera_center
    )

    # Weighted robust reprojection error
    reprojection_error = gmof(projected_joints - joints_2d, sigma)
    reprojection_loss = (joints_conf ** 2) * reprojection_error.sum(dim=-1)

    # Pose prior loss
    pose_prior_loss = (pose_prior_weight ** 2) * pose_prior(body_pose, betas)

    # Angle prior for knees and elbows
    angle_prior_loss = (angle_prior_weight ** 2) * angle_prior(body_pose).sum(
        dim=-1
    )

    # Regularizer to prevent betas from taking large values
    shape_prior_loss = (shape_prior_weight ** 2) * (betas ** 2).sum(dim=-1)

    total_loss = (
        reprojection_loss.sum(dim=-1)
        + pose_prior_loss
        + angle_prior_loss
        + shape_prior_loss
    )

    if output == "sum":
        return total_loss.sum()
    elif output == "reprojection":
        return reprojection_loss


def camera_fitting_loss(
    model_joints,
    camera_t,
    camera_t_est,
    camera_center,
    joints_2d,
    joints_conf,
    focal_length=5000,
    depth_loss_weight=100,
    use_depth_loss=True,
):
    """Loss function for camera optimization."""

    # Project model joints
    batch_size = model_joints.shape[0]
    rotation = (
        torch.eye(3, device=model_joints.device)
        .unsqueeze(0)
        .expand(batch_size, -1, -1)
    )
    projected_joints = perspective_projection(
        model_joints, rotation, camera_t, focal_length, camera_center
    )

    gt_joints = ["Right Hip", "Left Hip", "Right Shoulder", "Left Shoulder"]
    gt_joints_ind = [constants.JOINT_IDS[joint] for joint in gt_joints]

    reprojection_error_gt = (
        (joints_2d[:, gt_joints_ind] - projected_joints[:, gt_joints_ind]) ** 2
    ).sum(dim=-1) * (joints_conf[:, gt_joints_ind] ** 2)

    reprojection_loss = reprojection_error_gt.sum(dim=-1)

    # Loss that penalizes deviation from depth estimate
    depth_loss = (depth_loss_weight ** 2) * (
        camera_t[:, 2] - camera_t_est[:, 2]
    ) ** 2

    if use_depth_loss:
        total_loss = reprojection_loss + depth_loss
    else:
        total_loss = reprojection_loss
    return total_loss.sum()
