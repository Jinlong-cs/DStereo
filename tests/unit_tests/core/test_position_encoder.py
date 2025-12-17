# Copyright (c) Horizon Robotics. All rights reserved.
import json
import os

import numpy as np
import pytest

from hat.core.position_embedding_utils import PositionEncoder

pred_batch_size = 16
default_calib = np.array(
    [
        [
            [1114.34668, 0, 978.904541, 0],
            [0, 1114.34668, 670.611328, 0],
            [0, 0, 1, 0],
        ]
    ]
    * pred_batch_size
)

default_distCoeffs = np.array(
    [
        [
            -0.586143374,
            0.221308455,
            1.17504729e-04,
            1.71026128e-04,
            7.10545704e-02,
            -0.186895579,
            -9.31752697e-02,
            0.222488135,
        ]
    ]
    * pred_batch_size
)


@pytest.mark.parametrize(
    [
        "pe_config",
        "calib_params",
    ],
    [
        pytest.param(
            dict(
                is_with_pe=True,
                pe_stride=8,
                pe_c=3,
                pe_h=640 // 8,
                pe_w=960 // 8,
                img_resize=2,
                input_hw=(640, 960),
                default_intrinsic_mat=default_calib[0],
                default_distort=default_distCoeffs[0],
                default_pitch=0,
                default_roll=0,
                default_camera_z=1,
                crop_roi_3d=None,
                verbose=0,
            ),
            None,
        ),
        pytest.param(
            dict(
                is_with_pe=True,
                pe_stride=4,
                pe_c=3,
                pe_h=(1124 - 100) // 8,
                pe_w=(1920 - 0) // 8,
                img_resize=2,
                input_hw=(640, 960),
                default_intrinsic_mat=default_calib[0],
                default_distort=default_distCoeffs[0],
                default_pitch=0,
                default_roll=0,
                default_camera_z=1,
                crop_roi_3d=(0, 100, 1920, 1124),
                verbose=1,
            ),
            None,
        ),
        pytest.param(
            dict(
                is_with_pe=True,
                pe_stride=4,
                pe_c=3,
                pe_h=(1124 - 100) // 8,
                pe_w=(1920 - 0) // 8,
                img_resize=2,
                input_hw=(540, 960),
                default_intrinsic_mat=default_calib[0],
                default_distort=default_distCoeffs[0],
                default_pitch=0,
                default_roll=0,
                default_camera_z=1,
                crop_roi_3d=(0, 0, 1920, 1024),
                verbose=1,
            ),
            "/horizon-bucket/auto_jenkins_test/hdflow_workspace/test_3d/extrinsic_stat/camera_json/LX086/20220530/frontleft.json",  # noqa
        ),
    ],
)
def test_positionencoder(
    pe_config,
    calib_params,
):
    position_encoder = PositionEncoder(
        pe_stride=pe_config["pe_stride"],
        input_hw=pe_config["input_hw"],
        img_resize=pe_config["img_resize"],
        pe_h=pe_config["pe_h"],
        pe_w=pe_config["pe_w"],
        default_intrinsic_mat=pe_config["default_intrinsic_mat"],
        default_distort=pe_config["default_distort"],
        default_pitch=pe_config["default_pitch"],
        default_roll=pe_config["default_roll"],
        default_camera_z=pe_config["default_camera_z"],
        crop_roi=pe_config["crop_roi_3d"],
        verbose=pe_config["verbose"],
    )

    if calib_params is not None and os.path.exists(calib_params):
        with open(calib_params, "r") as fin:
            calib_params = json.load(fin)
    else:
        calib_params = None

    coordinate3d_map = position_encoder(calib_params)

    assert coordinate3d_map.shape == (
        pe_config["pe_c"],
        pe_config["pe_h"],
        pe_config["pe_w"],
    )

    position_encoder.reset()
