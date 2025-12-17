# Copyright (c) Horizon Robotics. All rights reserved.

import os

import cv2
import numpy as np
import pytest

from hat.data.datasets.model_eval_raw import ModelEvalRawDataset

eval_bucket_root = "/horizon-bucket/auto_eval/adas_eval/eval_platform/fs"
pred_batch_size = 1
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
        "data_path",
        "pe_config",
        "use_dataset_extrinsic",
    ],
    [
        pytest.param(
            os.path.join(eval_bucket_root, "6029905/datasets"),
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
                verbose=1,
            ),
            True,
        ),
        pytest.param(
            os.path.join(eval_bucket_root, "6029905/datasets"),
            dict(
                is_with_pe=False,
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
                verbose=1,
            ),
            True,
        ),
        pytest.param(
            os.path.join(eval_bucket_root, "6029905/datasets"),
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
                verbose=0,
            ),
            False,
        ),
        pytest.param(
            os.path.join(eval_bucket_root, "6029905/datasets"),
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
            False,
        ),
    ],
)
def test_model_eval_raw_dataset(
    data_path,
    pe_config,
    use_dataset_extrinsic,
):
    if os.path.exists(data_path):
        dataset = ModelEvalRawDataset(
            data_path,
            to_rgb=True,
            buf_only=True,
            return_orig_img=True,
            image_types=[".jpeg", ".png", ".jpg"],
            pe_config=pe_config,
            use_dataset_extrinsic=use_dataset_extrinsic,
        )
        if pe_config["is_with_pe"] and len(dataset) > 0:
            try:
                item = dataset[0]
                assert "coordinate_map" in item
                assert item["coordinate_map"].shape == (
                    pe_config["pe_c"],
                    pe_config["pe_h"],
                    pe_config["pe_w"],
                )
            except cv2.error:
                pass

        dataset.position_encoder.reset()
