import os
from copy import deepcopy

import cv2
import numpy as np
import pytest
import torch

from hat.core.differentiable_camera import (
    DifferentiableCameraBase,
    DifferentiableCylindricalCamera,
    DifferentiableFisheyeCamera,
    DifferentiablePinholeCamera,
)
from hat.core.virtual_camera.cameras import (
    CylindricalCamera,
    FisheyeCamera,
    IPMCamera,
    PinholeCamera,
)
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH

UT_DATA_ROOT = os.path.join(HAT_BUCKET_PATH, "unit_test_data")


@pytest.mark.skipif(
    not HAT_BUCKET_EXISTS,
    reason=f"requiring {UT_DATA_ROOT} bucket",
)
@pytest.mark.parametrize(
    [
        "image_path",
        "calib_cam_path",
        "num_iters",
    ],
    [
        pytest.param(
            os.path.join(
                UT_DATA_ROOT,
                "test_virtual_camera/pinhole/frontleft.jpg",
            ),
            os.path.join(
                UT_DATA_ROOT,
                "test_virtual_camera/pinhole/frontleft.json",
            ),
            5,
        ),
        pytest.param(
            os.path.join(
                UT_DATA_ROOT,
                "test_virtual_camera/pinhole/weissen.jpg",
            ),
            os.path.join(
                UT_DATA_ROOT,
                "test_virtual_camera/pinhole/weissen.json",
            ),
            5,
        ),
        pytest.param(
            os.path.join(
                UT_DATA_ROOT,
                "test_virtual_camera/pinhole/senyun.jpg",
            ),
            os.path.join(
                UT_DATA_ROOT,
                "test_virtual_camera/pinhole/senyun.json",
            ),
            10,
        ),
        pytest.param(
            os.path.join(
                UT_DATA_ROOT,
                "test_virtual_camera/fisheye/fisheye_back.jpg",
            ),
            os.path.join(
                UT_DATA_ROOT,
                "test_virtual_camera/fisheye/fisheye_back.json",
            ),
            10,
        ),
        pytest.param(
            os.path.join(
                UT_DATA_ROOT,
                "test_virtual_camera/fisheye/fisheye_rear.jpg",
            ),
            os.path.join(
                UT_DATA_ROOT,
                "test_virtual_camera/fisheye/fisheye_rear.json",
            ),
            10,
        ),
        pytest.param(
            os.path.join(
                UT_DATA_ROOT,
                "test_virtual_camera/fisheye/back.png",
            ),
            os.path.join(
                UT_DATA_ROOT,
                "test_virtual_camera/fisheye/back.json",
            ),
            10,
        ),
        pytest.param(
            os.path.join(
                UT_DATA_ROOT,
                "test_virtual_camera/fisheye/front.png",
            ),
            os.path.join(
                UT_DATA_ROOT,
                "test_virtual_camera/fisheye/front.json",
            ),
            10,
        ),
        pytest.param(
            os.path.join(
                UT_DATA_ROOT,
                "test_virtual_camera/fisheye/left.png",
            ),
            os.path.join(
                UT_DATA_ROOT,
                "test_virtual_camera/fisheye/left.json",
            ),
            10,
        ),
        pytest.param(
            os.path.join(
                UT_DATA_ROOT,
                "test_virtual_camera/fisheye/right.png",
            ),
            os.path.join(
                UT_DATA_ROOT,
                "test_virtual_camera/fisheye/right.json",
            ),
            10,
        ),
    ],
)
def test_Differentiable_cameras(
    image_path,
    calib_cam_path,
    num_iters,
):
    image = cv2.imread(image_path)
    image_ = torch.tensor(np.array([image, image])).permute(0, 3, 1, 2).float()
    if "fisheye" not in image_path:
        src_cam = PinholeCamera.init_cam_param_by_file(
            calib_path=calib_cam_path, is_virtual=False
        )
        diff_src_cam = DifferentiablePinholeCamera.init_from_cameras_list(
            [src_cam]
        )
        diff_src_cam.num_iters = num_iters
        # case2 undistort transform
        src_cam_undist = deepcopy(src_cam)
        src_cam_undist.distcoeffs = None
        diff_src_cam_undist = DifferentiableCameraBase.init_from_cameras_list(
            [src_cam_undist, src_cam_undist]
        )
        diff_src_cam_undist.num_iters = num_iters

    else:
        src_cam = FisheyeCamera.init_cam_param_by_file(
            calib_path=calib_cam_path, is_virtual=False
        )
        diff_src_cam = DifferentiableFisheyeCamera.init_from_cameras_list(
            [src_cam]
        )
        diff_src_cam.num_iters = num_iters
        # case2 undistort transform
        src_cam_undist = deepcopy(src_cam)
        src_cam_undist.distcoeffs = None
        diff_src_cam_undist = (
            DifferentiableFisheyeCamera.init_from_cameras_list(
                [src_cam_undist, src_cam_undist]
            )
        )
        diff_src_cam_undist.num_iters = num_iters

    diff_src_cam = diff_src_cam.join_cameras(diff_src_cam)

    ipm_cam = IPMCamera(
        image_size=(448, 896),
        vcs_range=(-25.6, -12.8, 25.6, 12.8),
    )
    diff_ipm_cam = DifferentiableCameraBase.init_from_cameras_list([ipm_cam])
    diff_ipm_cam = diff_ipm_cam.join_cameras(diff_ipm_cam)

    cyl_cam = CylindricalCamera.init_cam_param_by_matrix(
        image_size=(1408, 1152),
        camera_matrix=np.array([[440, 0, 704], [0, 440, 576], [0, 0, 1]]),
        poseMat_vcs2cam=deepcopy(src_cam.poseMat_vcs2cam),
        is_virtual=True,
    )
    diff_cyl_cam = DifferentiableCylindricalCamera.init_from_cameras_list(
        [cyl_cam]
    )
    diff_cyl_cam = diff_cyl_cam.join_cameras(diff_cyl_cam)

    # resize example
    diff_src_cam_resize = deepcopy(diff_src_cam)
    diff_src_cam_resize = diff_src_cam_resize.resize(
        scale_x=0.25, scale_y=0.25
    )
    image_resize = diff_src_cam.project_image2dstCam(
        diff_src_cam_resize,
        image_,
    )
    image0 = image_resize[0].permute(1, 2, 0).cpu().numpy().astype("uint8")

    # ipm example
    uv_map = diff_src_cam.generate_mapping(diff_ipm_cam)
    image_ipm = diff_src_cam.project_image2dstCam(diff_ipm_cam, image_, uv_map)
    image0 = image_ipm[0].permute(1, 2, 0).cpu().numpy().astype("uint8")
    image1 = src_cam.project_image2dstCam(ipm_cam, image)
    diff_pixel_rate = sum(
        np.abs(
            image0.reshape(-1).astype("float")
            - image1.reshape(-1).astype("float")
        )
        > 5
    ) / (image0.reshape(-1).shape[0])
    assert diff_pixel_rate < 1e-2, "Please check consistency!"

    # to cylindrical example
    uv_map = diff_src_cam.generate_mapping(diff_cyl_cam)
    image_cyl_tensor = diff_src_cam.project_image2dstCam(
        diff_cyl_cam, image_, uv_map
    )
    image_cyl_dvc = (
        image_cyl_tensor[0].permute(1, 2, 0).cpu().numpy().astype("uint8")
    )
    image_cyl_vc = src_cam.project_image2dstCam(cyl_cam, image)
    diff_pixel_rate = sum(
        np.abs(
            image_cyl_dvc.reshape(-1).astype(float)
            - image_cyl_vc.reshape(-1).astype(float)
        )
        > 5
    ) / (image_cyl_dvc.reshape(-1).shape[0] * 3)
    assert diff_pixel_rate < 1e-2, "Please check consistency!"

    # cylindrical to ipm example
    uv_map = diff_cyl_cam.generate_mapping(diff_ipm_cam)
    image_ipm = diff_cyl_cam.project_image2dstCam(
        diff_ipm_cam, image_cyl_tensor, uv_map
    )
    image_ipm_dvc = image_ipm[0].permute(1, 2, 0).cpu().numpy().astype("uint8")
    image_ipm_vc = cyl_cam.project_image2dstCam(ipm_cam, image_cyl_vc)
    diff_pixel_rate = sum(
        np.abs(
            image_ipm_dvc.reshape(-1).astype(float)
            - image_ipm_vc.reshape(-1).astype(float)
        )
        > 5
    ) / (image_ipm_dvc.reshape(-1).shape[0] * 3)
    assert diff_pixel_rate < 1e-2, "Please check consistency!"

    # undist example
    image_undist = diff_src_cam.project_image2dstCam(
        diff_src_cam_undist, image_
    )
    image0 = image_undist[0].permute(1, 2, 0).cpu().numpy().astype("uint8")

    if not isinstance(src_cam, FisheyeCamera):
        image1 = src_cam.project_image2dstCam(src_cam_undist, image)
        diff_pixel_rate = sum(
            np.abs(
                image0.reshape(-1).astype("float")
                - image1.reshape(-1).astype("float")
            )
            > 5
        ) / (image0.reshape(-1).shape[0])
        assert diff_pixel_rate < 1e-2, "Please check consistency!"

    # rebuild ipm
    image_ipm_rebuild = diff_ipm_cam.project_image2dstCam(
        diff_ipm_cam, image_ipm
    )
    image0 = image_ipm[0].permute(1, 2, 0).cpu().numpy().astype("uint8")
    image1 = (
        image_ipm_rebuild[0].permute(1, 2, 0).cpu().numpy().astype("uint8")
    )
    diff_pixel_rate = sum(
        np.abs(
            image0.reshape(-1).astype("float")
            - image1.reshape(-1).astype("float")
        )
        > 5
    ) / (image0.reshape(-1).shape[0])
    assert diff_pixel_rate < 1e-2, "Please check consistency!"


if __name__ == "__main__":
    pytest.main(["-s", __file__])
