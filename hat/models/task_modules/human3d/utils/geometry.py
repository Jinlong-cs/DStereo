"""Geometry utils.

Useful geometric operations, e.g.
Perspective projection and a differentiable Rodrigues formula
Parts of the code are taken from https://github.com/MandyMo/pytorch_HMR
"""
import numpy as np
import torch


def perspective_projection(
    points, rotation, translation, focal_length, camera_center
):
    """Compute the perspective projection of a set of points.

    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.0
    K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum("bij,bkj->bki", rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum("bij,bkj->bki", K, projected_points)

    return projected_points[:, :, :-1]


def estimate_translation_np(
    S, joints_2d, joints_conf, focal_length=5000, img_size=224
):
    """Find camera translation.

    Find camera translation.that brings 3D joints S closest to
    2D the corresponding joints_2d.

    Input:
        S: (24, 3) 3D joint locations
        joints: (24, 3) 2D joint locations and confidence
    Returns:
        (3,) camera translation vector
    """

    num_joints = S.shape[0]
    # focal length
    f = np.array([focal_length, focal_length])
    # optical center
    center = np.array([img_size / 2.0, img_size / 2.0])

    # transformations
    Z = np.reshape(np.tile(S[:, 2], (2, 1)).T, -1)
    XY = np.reshape(S[:, 0:2], -1)
    O = np.tile(center, num_joints)  # noqa
    F = np.tile(f, num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf), (2, 1)).T, -1)

    # least squares
    Q = np.array(
        [
            F * np.tile(np.array([1, 0]), num_joints),
            F * np.tile(np.array([0, 1]), num_joints),
            O - np.reshape(joints_2d, -1),
        ]
    ).T
    c = (np.reshape(joints_2d, -1) - O) * Z - F * XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W, Q)
    c = np.dot(W, c)

    # square matrix
    A = np.dot(Q.T, Q)
    b = np.dot(Q.T, c)

    # solution
    trans = np.linalg.solve(A, b)

    return trans


def estimate_translation(
    joints_3d, joints_2d, joints_conf, focal_length=5000.0, img_size=224.0
):
    """Find camera translation.

    Find camera translation that brings 3D joints S closest to
    2D the corresponding joints_2d.

    Input:
        joints_3d: (B, N, 3) 3D joint locations
        joints_2d: (B, N, 3) 2D joint locations and confidence
    Returns:
        (B, 3) camera translation vectors
    """

    device = joints_3d.device

    joints_3d = joints_3d.cpu().numpy()
    joints_2d = joints_2d.cpu().numpy()
    trans = np.zeros((joints_3d.shape[0], 3), dtype=np.float32)
    # Find the translation for each example in the batch
    for i in range(joints_3d.shape[0]):
        joints_3d_i = joints_3d[i]
        joints_i = joints_2d[i]
        conf_i = joints_conf[i]
        trans[i] = estimate_translation_np(
            joints_3d_i,
            joints_i,
            conf_i,
            focal_length=focal_length,
            img_size=img_size,
        )
    return torch.from_numpy(trans).to(device)
