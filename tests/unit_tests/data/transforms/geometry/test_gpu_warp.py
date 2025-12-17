import os
from copy import deepcopy

import cv2
import numpy as np
import pytest
import torch

from hat.core.virtual_camera.cameras import (
    CylindricalCamera,
    FisheyeCamera,
    PinholeCamera,
)
from hat.data.transforms.geometry import DiffCamImageProjector
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH

UT_DATA_ROOT = os.path.join(HAT_BUCKET_PATH, "unit_test_data")


@pytest.mark.skipif(
    not HAT_BUCKET_EXISTS,
    reason=f"requiring {UT_DATA_ROOT} bucket",
)
@pytest.mark.parametrize(
    [
        "image_list",
        "cam_param_list",
    ],
    [
        pytest.param(
            [
                os.path.join(
                    UT_DATA_ROOT,
                    "test_virtual_camera/pinhole/weissen.jpg",
                ),
                os.path.join(
                    UT_DATA_ROOT,
                    "test_virtual_camera/fisheye/fisheye_rear.jpg",
                ),
                os.path.join(
                    UT_DATA_ROOT,
                    "test_virtual_camera/pinhole/weissen.jpg",
                ),
            ],
            [
                os.path.join(
                    UT_DATA_ROOT,
                    "test_virtual_camera/pinhole/weissen.json",
                ),
                os.path.join(
                    UT_DATA_ROOT,
                    "test_virtual_camera/fisheye/fisheye_rear.json",
                ),
                os.path.join(
                    UT_DATA_ROOT,
                    "test_virtual_camera/pinhole/weissen.json",
                ),
            ],
        ),
        pytest.param(
            [
                os.path.join(
                    UT_DATA_ROOT,
                    "test_virtual_camera/pinhole/frontleft.jpg",
                ),
                os.path.join(
                    UT_DATA_ROOT,
                    "test_virtual_camera/fisheye/fisheye_back.jpg",
                ),
                os.path.join(
                    UT_DATA_ROOT,
                    "test_virtual_camera/fisheye/fisheye_back.jpg",
                ),
            ],
            [
                os.path.join(
                    UT_DATA_ROOT,
                    "test_virtual_camera/pinhole/frontleft.json",
                ),
                os.path.join(
                    UT_DATA_ROOT,
                    "test_virtual_camera/fisheye/fisheye_back.json",
                ),
                os.path.join(
                    UT_DATA_ROOT,
                    "test_virtual_camera/fisheye/fisheye_back.json",
                ),
            ],
        ),
    ],
)
def test_gpu_warp(image_list, cam_param_list):
    gpu_warp_merge = DiffCamImageProjector(
        num_cal_iters=10,
        merge_data_by_camera=True,
        src_cam_key="source_cam",
        dst_cam_key="warp_cam",
        from_meta=False,
        image_key="img",
    )
    gpu_warp = DiffCamImageProjector(
        num_cal_iters=10,
        merge_data_by_camera=False,
        src_cam_key="source_cam",
        dst_cam_key="warp_cam",
        from_meta=False,
        image_key="img",
    )
    image_list = [cv2.imread(image_path) for image_path in image_list]
    image_tensor_list = [
        torch.from_numpy(image).permute(2, 0, 1) for image in image_list
    ]
    src_cam_list = []
    dst_cam_list = []
    for cam_path in cam_param_list:
        src_cam = FisheyeCamera if "fisheye" in cam_path else PinholeCamera
        src_cam = src_cam.init_cam_param_by_file(cam_path)
        dst_cam = CylindricalCamera.init_cam_param_by_matrix(
            image_size=(1408, 1152),
            camera_matrix=np.array([[440, 0, 704], [0, 440, 576], [0, 0, 1]]),
            poseMat_vcs2cam=deepcopy(src_cam.poseMat_vcs2cam),
            is_virtual=True,
        )
        src_cam_list.append(src_cam)
        dst_cam_list.append(dst_cam)

    data = {
        "img": image_tensor_list,
        "source_cam": src_cam_list,
        "warp_cam": dst_cam_list,
    }

    merge_gpu_warped_data = gpu_warp_merge(deepcopy(data))
    gpu_warped_data = gpu_warp(deepcopy(data))
    diff_rate = (
        torch.abs(
            merge_gpu_warped_data["img"].float()
            - gpu_warped_data["img"].float()
        ).sum()
        / gpu_warped_data["img"].float().sum()
    )
    assert diff_rate < 1e-6, "diff rate beyond tolerance: {}".format(diff_rate)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
