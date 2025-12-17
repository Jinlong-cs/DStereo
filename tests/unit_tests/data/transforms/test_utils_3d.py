import numpy as np
import pytest

from hat.core.box3d_utils import get_3dbox_corners
from hat.core.utils_3d import (
    compute_box_parametric,
    get_3dbox_dense_points,
    move_in_same_period_scalar,
)


@pytest.mark.parametrize(
    ["alpha", "beta", "theta"],
    [
        pytest.param(0, np.pi * 2.0, 0),
        pytest.param(0, np.pi * 3 / 2, -np.pi / 2),
        pytest.param(0, np.pi * 8 + 0.01, 0.01),
        pytest.param(0, -np.pi * 8 - 0.01, -0.01),
    ],
)
def test_move_in_same_period_scalar(alpha, beta, theta):
    beta = move_in_same_period_scalar(alpha, beta)
    assert abs(beta - theta) < 1e-4


def test_get_3dbox_corners():
    loc = [0, 0, 0]
    dim = [4, 2, 3]
    rot_y = 0
    pts = get_3dbox_corners(loc, dim, rot_y, coord_system="cv_camera")

    assert abs(pts[0, 0] - 2.0) < 1e-4
    assert abs(pts[0, 2] + 1.0) < 1e-4
    assert abs(pts[4, 1] + 3) < 1e-4

    rot_y = -np.pi * 0.5
    pts = get_3dbox_corners(loc, dim, rot_y, coord_system="cv_camera")
    assert abs(pts[0, 2] - 2.0) < 1e-4
    assert abs(abs(pts[0, 0]) - 1.0) < 1e-4
    assert abs(pts[4, 1] + 3) < 1e-4

    yaw = 0.0
    pts = get_3dbox_corners(loc, dim, yaw, coord_system="vcs")
    assert abs(abs(pts[0, 0]) - 2.0) < 1e-4
    assert abs(abs(pts[0, 1]) - 1.0) < 1e-4
    assert abs(abs(pts[4, 2]) - 3) < 1e-4

    yaw = np.pi * 0.5
    pts = get_3dbox_corners(loc, dim, yaw, coord_system="vcs")
    assert abs(abs(pts[0, 0]) - 1.0) < 1e-4
    assert abs(abs(pts[0, 1]) - 2.0) < 1e-4
    assert abs(abs(pts[4, 2]) - 3) < 1e-4


def test_compute_box_parametric_in_camera():
    loc = np.array([1.0, 2.0, 0.0])
    dim = [4, 2, 3]
    rot_y = np.pi
    pts = get_3dbox_corners(loc, dim, rot_y, coord_system="cv_camera")

    h_loc, h_rot_y = compute_box_parametric(pts, coord_system="cv_camera")
    assert np.all(np.abs(np.array(loc) - h_loc) < 1e-4)
    assert h_rot_y < np.pi and h_rot_y >= -np.pi
    assert abs(h_rot_y - rot_y) < 1e-4

    loc = np.array([1.0, 2.0, 1.0])
    dim = [4, 2, 3]
    yaw = -np.pi * 0.5
    pts = get_3dbox_corners(loc, dim, yaw, coord_system="vcs")
    h_loc, h_yaw = compute_box_parametric(pts, coord_system="vcs")
    assert np.all(np.abs(loc - h_loc) < 1e-4)
    assert abs(h_yaw - yaw) < 1e-4


def test_get_3dbox_dense_points():
    # This test is trival, real test should be visualized
    loc = np.array([0.0, 0.0, 0.0])
    dim = [4, 2, 3]
    yaw = 0

    pts_dense = get_3dbox_dense_points(loc, dim, yaw, 50)
    assert pts_dense.shape[0] == 12 * 50
    assert pts_dense.shape[1] == 3


if __name__ == "__main__":
    pytest.main(["-s", __file__])
