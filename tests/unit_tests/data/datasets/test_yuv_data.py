import os

import numpy as np
import pytest

from hat.data.datasets.yuv_data import YUVFrames
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH

try:
    import pyramid_resizer
except ImportError:
    pyramid_resizer = None


pe_config = {
    "is_with_pe": True,
    "pe_stride": 8,
    "pe_c": 3,
    "pe_h": 80,
    "pe_w": 120,
    "img_resize": 2,
    "input_hw": (640, 960),
    "default_intrinsic_mat": np.array(
        [
            [1.1143467e03, 0.0000000e00, 9.7890454e02, 0.0000000e00],
            [0.0000000e00, 1.1143467e03, 6.7061133e02, 0.0000000e00],
            [0.0000000e00, 0.0000000e00, 1.0000000e00, 0.0000000e00],
        ],
        dtype=np.float32,
    ),
    "default_distort": np.array(
        [
            -5.8614337e-01,
            2.2130845e-01,
            1.1750473e-04,
            1.7102613e-04,
            7.1054570e-02,
            -1.8689558e-01,
            -9.3175270e-02,
            2.2248814e-01,
        ],
        dtype=np.float32,
    ),
    "default_pitch": 0,
    "default_roll": 0,
    "default_camera_z": 1,
    "crop_roi_3d": None,
    "is_with_relu": False,
    "verbose": 1,
}


@pytest.mark.skipif(pyramid_resizer is None, reason="need pyramid_resizer")
@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
@pytest.mark.parametrize(
    "enable_calib_all, pe_config",
    [(True, pe_config), (False, pe_config), (True, None), (False, None)],
)
def test_yuv_frame(enable_calib_all, pe_config):
    bucket_path = HAT_BUCKET_PATH
    data_root = os.path.join(bucket_path, "users/xuewu.lin")
    im_hw = (640, 960)
    img_dir = os.path.join(data_root, "resize_2/yuv_img/")
    calib_path = os.path.join(data_root, "camera_0.json")

    dataset = YUVFrames(
        img_dir,
        im_hw=im_hw,
        calib_path=calib_path,
        pe_config=pe_config,
        enable_calib_all=enable_calib_all,
    )
    for data in dataset:
        assert "img" in data
        assert "img_id" in data
        assert data.get("img_height") == im_hw[0]
        assert data.get("img_width") == im_hw[1]
        assert data.get("layout") == "chw"
        if pe_config is not None:
            assert "coordinate_map" in data

        if enable_calib_all:
            assert "calib_all" in data
        else:
            assert "calib_all" not in data
