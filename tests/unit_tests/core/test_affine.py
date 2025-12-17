import numpy as np
import pytest

from hat.core.affine import (
    affine_transform,
    get_affine_transform,
    get_vcs2bev_img_mat,
)
from hat.core.virtual_camera import IPMCamera


@pytest.mark.parametrize(
    ["size", "rotation", "out_size", "inverse", "target_res"],
    [
        pytest.param(
            (1920, 1080),
            0,
            [240, 128],
            False,
            np.array([[0.125, 0.0, 0.0], [0.0, 0.125, -3.5]]),
        ),
        pytest.param(
            (1920, 1080),
            0,
            [960, 512],
            False,
            np.array([[0.5, 0.0, 0.0], [0.0, 0.5, -14.0]]),
        ),
    ],
)
def test_get_affine_transform(size, rotation, out_size, inverse, target_res):
    res = get_affine_transform(size, rotation, out_size, inverse)
    assert (res == target_res).all()
    P1 = get_affine_transform(size, rotation, out_size, False)
    P2 = get_affine_transform(size, rotation, out_size, True)
    assert ((P1 * P2)[:2, :2] == np.identity(2)).all()

    # test center_shift
    M0 = get_affine_transform(
        size=(256, 256),
        rotation=0,
        out_size=(128, 120),
        center_shift=(0, -8),
    )
    tar_0 = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
    assert (M0 == tar_0).all()

    # test pre_resize_scale
    M0 = get_affine_transform(
        size=(256, 256),
        rotation=0,
        out_size=(60, 60),
        pre_resize_scale=0.5,
        center_shift=(68, 68),
    )
    M0 = np.round(M0, 6)
    tar_0 = np.array([[0.5, 0.0, -68.0], [0.0, 0.5, -68.0]])
    assert (M0 == tar_0).all()


@pytest.mark.parametrize(
    ["pt", "t", "target_res"],
    [
        pytest.param(
            [np.array([100, 200])],
            np.array([[0.125, 0.0, 0.0], [0.0, 0.125, -3.5]]),
            np.array([[12.5, 21.5]]),
        ),
        pytest.param(
            [np.array([100, 200])],
            np.array([[1, 0, 0], [0, 1, 0]]),
            np.array([[100.0, 200.0]]),
        ),
        pytest.param(
            [np.array([100, 200])],
            np.array([[1, 0, 0], [0, 1, 1]]),
            np.array([[100.0, 201.0]]),
        ),
    ],
)
def test_affine_transform(pt, t, target_res):
    res = affine_transform(pt, t)
    assert (res == target_res).all()


@pytest.mark.parametrize(
    ["vcs_range", "bev_size"],
    [
        pytest.param(
            (-31.8, -57.6, 102.6, 57.6),
            (224, 192),
        ),
        pytest.param(
            (-31.8, -57.6, 102.6, 57.6),
            (224, 384),
        ),
        pytest.param(
            (-12.8, -12.8, 25.6, 12.8),
            (192, 128),
        ),
        pytest.param(
            (-35.2, -38.4, 105.6, 38.4),
            (352, 192),
        ),
    ],
)
def test_get_vcs2bev_img_mat(vcs_range, bev_size):
    new_matrix = get_vcs2bev_img_mat(vcs_range, bev_size)
    ipm_cam = IPMCamera(image_size=bev_size[::-1], vcs_range=vcs_range)
    old_matrix = ipm_cam.get_poseMat_vcsgnd2pixel(ipm_cam)
    assert np.allclose(new_matrix, old_matrix, atol=1e-6)
