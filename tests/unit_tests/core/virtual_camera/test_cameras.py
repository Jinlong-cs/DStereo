# Copyright (c) Horizon Robotics. All rights reserved.
# The unit test for VIRTUAL_CAMERAS


import os

import cv2
import numpy as np
import pytest

from hat.core.virtual_camera import (
    CylindricalCamera,
    FisheyeCamera,
    IPMCamera,
    PinholeCamera,
    SphericalCamera,
)
from hat.core.virtual_camera.utils import ImagePointsInterpolation
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH

UT_DATA_ROOT = os.path.join(HAT_BUCKET_PATH, "unit_test_data")


def create_Fisheye_cam(calib_path):
    camera = FisheyeCamera.init_cam_param_by_file(
        calib_path=calib_path, is_virtual=False
    )
    return camera


def create_Cylindrical_cam(calib_path):
    camera = CylindricalCamera.init_cam_param_by_file(
        calib_path=calib_path, is_virtual=True
    )
    camera.set_cam_param(
        image_size=[1920, 1280],
        principle_point=[960, 512],
        fx=472,
        fy=472,
    )
    return camera


def create_Spherical_cam(calib_path):
    camera = SphericalCamera.init_cam_param_by_file(
        calib_path=calib_path, is_virtual=True
    )
    camera.set_cam_param(
        image_size=[1920, 1280],
        principle_point=[960, 512],
        fx=472,
        fy=472,
    )
    return camera


def create_Pinhole_Camera(calib_path):
    camera = PinholeCamera.init_cam_param_by_file(
        calib_path=calib_path, is_virtual=True
    )
    return camera


@pytest.mark.skipif(
    not HAT_BUCKET_EXISTS,
    reason=f"requiring {HAT_BUCKET_PATH} bucket",
)  # noqa: E501
@pytest.mark.parametrize(
    [
        "image_path",
        "calib_cam_path",
    ],
    [
        pytest.param(
            os.path.join(
                UT_DATA_ROOT,
                "test_virtual_camera/fisheye/fisheye_rear.jpg",  # noqa: E501
            ),
            os.path.join(
                UT_DATA_ROOT,
                "test_virtual_camera/fisheye/fisheye_rear.json",  # noqa: E501
            ),
        ),
        pytest.param(
            os.path.join(
                UT_DATA_ROOT,
                "test_virtual_camera/pinhole/frontleft.jpg",  # noqa: E501
            ),
            os.path.join(
                UT_DATA_ROOT,
                "test_virtual_camera/pinhole/frontleft.json",  # noqa: E501
            ),
        ),
    ],
)
# more visual cases check:
def test_VirtualCameras_coordinate_projection(
    image_path,
    calib_cam_path,
):
    image = cv2.imread(image_path)
    if "fisheye" in image_path:
        src_cam = FisheyeCamera.init_cam_param_by_file(
            calib_path=calib_cam_path, is_virtual=False
        )
    else:
        src_cam = PinholeCamera.init_cam_param_by_file(
            calib_path=calib_cam_path, is_virtual=False
        )

    # pixel_points shape is (n, 2)
    src_pts = np.array(
        [[847, 573], [556, 574], [100, 782], [432, 832]],
        dtype=np.float64,
    )

    # Example 1: src_cam -> virtual cylinder
    cylindrical_cam = create_Cylindrical_cam(calib_cam_path)
    cylindrical_image = src_cam.project_image2dstCam(  # noqa F841
        cylindrical_cam, image
    )
    # src_cam pts project to cylindrical_cam
    cyl_pts = src_cam.project_pixel2dstCam(cylindrical_cam, src_pts)
    cyl_pts2camcoord = cylindrical_cam.project_pixel2cam(cyl_pts)
    cyl_camcoord2pts = cylindrical_cam.project_cam2pixel(cyl_pts2camcoord)
    assert (
        cyl_pts.astype(int) == cyl_camcoord2pts.astype(int)
    ).all(), "assert error"

    cyl_pts2vcscoord = cylindrical_cam.project_pixel2vcs(cyl_pts)
    cyl_vcscoord2pts = cylindrical_cam.project_vcs2pixel(cyl_pts2vcscoord)
    assert (
        cyl_pts.astype(int) == cyl_vcscoord2pts.astype(int)
    ).all(), "assert error"

    # Example 2: src_cam -> ipm
    ipm_cam = IPMCamera(
        image_size=(896, 896),
        vcs_range=(-10.36, -8.96, 7.56, 8.96),
    )
    # warp image
    ipm_image = src_cam.project_image2dstCam(ipm_cam, image)  # noqa F841

    ipm_pts = src_cam.project_pixel2dstCam(ipm_cam, src_pts)
    ipm_pts2camcoord = ipm_cam.project_pixel2cam(ipm_pts)
    ipm_camcoord2pts = ipm_cam.project_cam2pixel(ipm_pts2camcoord)
    assert (
        ipm_pts.astype(int) == ipm_camcoord2pts.astype(int)
    ).all(), "assert error"

    ipm_pts2vcscoord = ipm_cam.project_pixel2vcs(ipm_pts)
    ipm_vcscoord2pts = ipm_cam.project_vcs2pixel(ipm_pts2vcscoord)
    assert (
        ipm_pts.astype(int) == ipm_vcscoord2pts.astype(int)
    ).all(), "assert error"

    # Example 3: src_cam -> pinhole
    pinhole_cam = create_Pinhole_Camera(calib_cam_path)
    pinhole_image = src_cam.project_image2dstCam(  # noqa F841
        pinhole_cam, image
    )
    pinhole_pts = src_cam.project_pixel2dstCam(pinhole_cam, src_pts)
    pinhole_pts2camcoord = pinhole_cam.project_pixel2cam(pinhole_pts)
    pinhole_camcoord2pts = pinhole_cam.project_cam2pixel(pinhole_pts2camcoord)
    assert (
        pinhole_pts.astype(int) == pinhole_camcoord2pts.astype(int)
    ).all(), "assert error"

    pinhole_pts2vcscoord = pinhole_cam.project_pixel2vcs(pinhole_pts)
    pinhole_vcscoord2pts = pinhole_cam.project_vcs2pixel(pinhole_pts2vcscoord)
    assert (
        pinhole_pts.astype(int) == pinhole_vcscoord2pts.astype(int)
    ).all(), "assert error"

    # Example 4: src_cam -> virtual spherical
    spherical_cam = create_Spherical_cam(calib_cam_path)
    spherical_image = src_cam.project_image2dstCam(  # noqa F841
        spherical_cam, image
    )


@pytest.mark.skipif(
    not HAT_BUCKET_EXISTS,
    reason="requiring AUTO_JENKINS_TEST bucket",
)  # noqa: E501
@pytest.mark.parametrize(
    [
        "image_path",
        "calib_cam_path",
    ],
    [
        pytest.param(
            os.path.join(
                UT_DATA_ROOT,
                "test_virtual_camera/fisheye/fisheye_back.jpg",  # noqa: E501
            ),
            os.path.join(
                UT_DATA_ROOT,
                "test_virtual_camera/fisheye/fisheye_back.json",  # noqa: E501
            ),
        ),
        pytest.param(
            os.path.join(
                UT_DATA_ROOT,
                "test_virtual_camera/pinhole/frontleft.jpg",  # noqa: E501
            ),
            os.path.join(
                UT_DATA_ROOT,
                "test_virtual_camera/pinhole/frontleft.json",  # noqa: E501
            ),
        ),
    ],
)
def test_ImagePointsInterpolation(
    image_path,
    calib_cam_path,
):
    image = cv2.imread(image_path)
    if "fisheye" in image_path:
        src_cam = FisheyeCamera.init_cam_param_by_file(
            calib_path=calib_cam_path, is_virtual=False
        )
    else:
        src_cam = PinholeCamera.init_cam_param_by_file(
            calib_path=calib_cam_path, is_virtual=False
        )
    # Example 1: src_cam -> virtual cylinder
    cylindrical_cam = create_Cylindrical_cam(calib_cam_path)
    cylindrical_image = src_cam.project_image2dstCam(  # noqa F841
        cylindrical_cam, image
    )
    point_interpolator = ImagePointsInterpolation(
        cylindrical_cam, pt_interval=5
    )

    # manually annotated points:
    # [[x0, y0], [x1, y1], [x2, y2], [x3, y3], ...]
    input_points = np.array(
        [[847, 573.2], [1056, 574.2], [1580, 782.2], [325, 832.2]],
        dtype=np.float32,
    )
    point_sequence = point_interpolator.anno_via_point_interpolation(
        input_points
    )

    # only for visualization
    image_out = point_interpolator.visualize_results(  # noqa F841
        cylindrical_image,
        np.array(input_points, dtype=np.float32),
        point_sequence,
    )

    # example3: World Points Projection
    vcs_points = np.array(
        [
            [
                -4.60946345,
                1.39410797,
                0.35820493,
            ],
            [-2.00946345, 1.37119823, 0.35820493],
            [-2.00946345, 1.37119823 + 1.7, 0.35820493],
            [
                -4.60946345,
                1.39410797 + 1.7,
                0.35820493,
            ],
        ]
    )
    pixel_points = cylindrical_cam.project_vcs2pixel(vcs_points)

    image_out = point_interpolator.visualize_results(  # noqa F841
        cylindrical_image,
        np.array(pixel_points, dtype=np.float64),
        pixel_points,
    )


if __name__ == "__main__":
    pytest.main(["-s", __file__])
