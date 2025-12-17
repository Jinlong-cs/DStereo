from typing import Optional

try:
    import kornia
except ImportError:
    kornia = None

import torch

from hat.utils.package_helper import require_packages
from .lbs import transform_mat

__all__ = ["perspective_camera"]


@require_packages("kornia")
def perspective_camera(
    rot_mat: torch.Tensor,
    transl: torch.Tensor,
    cam_ldmk: torch.Tensor,
    cam_verts: torch.Tensor,
    intri: torch.Tensor,
    dist: Optional[torch.Tensor] = None,
):
    # TODO: (yuhao.dou) make it simple

    # transform matrix
    camera_transform = transform_mat(rot_mat, transl.unsqueeze(dim=-1))
    # Convert the points to homogeneous coordinates
    homog_coord_ldmk = torch.ones(
        list(cam_ldmk.shape)[:-1] + [1],
        dtype=cam_ldmk.dtype,
        device=cam_ldmk.device,
    )
    points_h_ldmk = torch.cat([cam_ldmk, homog_coord_ldmk], dim=-1)
    homog_coord_vt = torch.ones(
        list(cam_verts.shape)[:-1] + [1],
        dtype=cam_verts.dtype,
        device=cam_verts.device,
    )
    points_h_vt = torch.cat([cam_verts, homog_coord_vt], dim=-1)
    # local space to camera space
    projected_points_ldmk = torch.einsum(
        "bki,bji->bjk", [camera_transform, points_h_ldmk]
    )
    cam_ldmk = projected_points_ldmk.clone()[..., :3]
    projected_points_vt = torch.einsum(
        "bki,bji->bjk", [camera_transform, points_h_vt]
    )
    # cam_points_vt = projected_points_vt.clone()
    # camera space to z=1 space
    img_points_ldmk = torch.div(
        projected_points_ldmk[:, :, :2],
        projected_points_ldmk[:, :, 2:3],
    )
    img_points_vt = torch.div(
        projected_points_vt[:, :, :2],
        projected_points_vt[:, :, 2:3],
    )
    # distort points
    if dist is not None:
        img_points_ldmk = kornia.geometry.calibration.distort_points(
            img_points_ldmk,
            torch.eye(3, device=img_points_ldmk.device)[None],
            dist,
        )
        img_points_vt = kornia.geometry.calibration.distort_points(
            img_points_vt,
            torch.eye(3, device=img_points_vt.device)[None],
            dist,
        )
    cx = intri[..., 0:1, 2]  # princial point in x (Bx1)
    cy = intri[..., 1:2, 2]  # princial point in y (Bx1)
    fx = intri[..., 0:1, 0]  # focal in x (Bx1)
    fy = intri[..., 1:2, 1]  # focal in y (Bx1)
    camera_mat = torch.zeros(
        [img_points_ldmk.size(0), 2, 2],
        dtype=torch.float32,
        device=img_points_ldmk.device,
    )
    camera_mat[..., 0:1, 0] = fx
    camera_mat[..., 1:2, 1] = fy
    camera_center = torch.zeros(
        [img_points_ldmk.size(0), 1, 2],
        dtype=torch.float32,
        device=img_points_ldmk.device,
    )
    camera_center[..., :, 0] = cx
    camera_center[..., :, 1] = cy
    img_points_ldmk = (
        torch.einsum("bki,bji->bjk", [camera_mat, img_points_ldmk])
        + camera_center
    )
    img_points_vt = (
        torch.einsum("bki,bji->bjk", [camera_mat, img_points_vt])
        + camera_center
    )
    img_verts = torch.cat(
        (img_points_vt, projected_points_vt[..., 2:3]), dim=-1
    )
    return img_points_ldmk, cam_ldmk, img_verts
