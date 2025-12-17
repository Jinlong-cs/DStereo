import cv2
import numpy as np
import pytest
import torch

from hat.core.utils_3d import project_func_pinhole


@pytest.mark.parametrize(
    [
        "pts_3d",
        "cam_intrinsic",
        "cam_distcoeff",
        "num_cam",
    ],
    [
        pytest.param(
            np.array([[3.1, 2.1, 18.8]]),
            np.array(
                [
                    [
                        [461.0, 0.0, 478.9],
                        [0.0, 461.0, 271.2],
                        [0.0, 0.0, 1.0],
                    ],
                    [
                        [1200.3, 0.0, 506.9],
                        [0.0, 1200.3, 348.2],
                        [0.0, 0.0, 1.0],
                    ],
                ]
            ),
            np.array(
                [
                    [1.89, 2.43, 2.91e-04, -8.39e-05, 0.19, 2.23, 3.07, 0.80],
                    [1.89, 2.43, 2.91e-04, -8.39e-05, 0.19, 2.23, 3.07, 0.80],
                ]
            ),
            2,
        ),
    ],
)
def test_project_func_pinhole(pts_3d, cam_intrinsic, cam_distcoeff, num_cam):
    pts_num = 10
    pts_3d = pts_3d + np.random.randn(pts_num, 3) * 2
    pts_2d_cv2 = []
    for cam_idx in range(num_cam):
        rvec, _ = cv2.Rodrigues(np.identity(3, np.float32))
        tvec = np.zeros(shape=(3, 1), dtype=np.float32)
        image_pts = cv2.projectPoints(
            pts_3d[:, :3],
            np.array(rvec),
            tvec,
            cam_intrinsic[cam_idx, :3, :3],
            cam_distcoeff[cam_idx],
        )[0]
        pts_2d_cv2.append(np.squeeze(image_pts))
    pts_2d_cv2 = np.stack(pts_2d_cv2, axis=0)

    cam_intrinsic = torch.tensor(cam_intrinsic).reshape(num_cam, 3, 3)
    cam_distcoeff = torch.tensor(cam_distcoeff).reshape(num_cam, 8)

    pts_3d = torch.tensor(pts_3d).expand(num_cam, pts_num, 3)

    pts_2d = project_func_pinhole(pts_3d, cam_intrinsic, cam_distcoeff)
    assert (torch.abs(pts_2d - torch.from_numpy(pts_2d_cv2)) < 1e-6).all()


if __name__ == "__main__":
    pytest.main(["-s", __file__])
