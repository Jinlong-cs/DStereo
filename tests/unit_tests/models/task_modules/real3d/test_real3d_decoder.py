# Copyright (c) Horizon Robotics. All rights reserved.

import numpy as np
import torch

from hat.models.task_modules.real3d import Real3DDecoder
from hat.models.task_modules.real3d.decoder import get_orientation


def gen_fake_data_pred():
    return dict(
        hm=torch.zeros(1, 3, 128, 240, dtype=torch.float32),
        dep=torch.zeros(1, 1, 128, 240, dtype=torch.float32),
        rot=torch.zeros(1, 2, 128, 240, dtype=torch.float32),
        dim=torch.zeros(1, 3, 128, 240, dtype=torch.float32),
        loc_offset=torch.zeros(1, 2, 128, 240, dtype=torch.float32),
        wh=torch.zeros(1, 2, 128, 240, dtype=torch.float32),
    )


def gen_fake_data_target():
    return dict(
        calibration=torch.tensor(
            [
                [
                    [2.4110e03, 0.0000e00, 1.8908e03, 0.0000e00],
                    [0.0000e00, 2.4110e03, 1.1042e03, 0.0000e00],
                    [0.0000e00, 0.0000e00, 1.0000e00, 0.0000e00],
                ]
            ]
        ),
        image_transform={
            "M": torch.tensor(
                [[[0.2500, -0.0000, 0.0000], [0.0000, 0.2500, -14.0000]]]
            ),
            "original_size": [
                torch.tensor([3840], dtype=torch.int64),
                torch.tensor([2160], dtype=torch.int64),
            ],
            "input_size": [
                torch.tensor([3840], dtype=torch.int64),
                torch.tensor([2160], dtype=torch.int64),
            ],
        },
        dist_coeffs=torch.tensor([[0.02, -0.03, 0.002, 0.003]]),
    )


def test_real3d_decoder():
    pred = gen_fake_data_pred()
    target = gen_fake_data_target()
    real3d_decoder = Real3DDecoder(
        focal_length_default=2411,
        topk=40,
        max_pooling_kernel_size=3,
        undistort=True,
        fisheye=True,
    )
    results = real3d_decoder(pred, target)

    assert "category_id" in results
    assert "score" in results
    assert "bbox" in results
    assert "center" in results
    assert "alpha" in results
    assert "location" in results
    assert "rotation_y" in results

    assert results["category_id"].shape[0] == pred["dep"].shape[0]


def test_real3d_decoder_nms():
    pred = gen_fake_data_pred()
    target = gen_fake_data_target()
    nms_kwargs = {
        "iou_threshold": 0.7,
        "replace": True,
    }
    real3d_decoder = Real3DDecoder(
        focal_length_default=2411,
        topk=40,
        max_pooling_kernel_size=3,
        undistort=True,
        fisheye=True,
        nms_kwargs=nms_kwargs,
    )
    results = real3d_decoder(pred, target)

    assert "nms_keep" in results
    assert "score" in results

    assert results["nms_keep"].shape == results["score"].shape


def test_get_orientation():
    b, n = 2, 10
    pred_alpha_z = torch.zeros((b, n, 12), dtype=torch.float32)
    location = torch.zeros((b, n, 3), dtype=torch.float32)
    bin_center = [0, np.pi / 2, np.pi, -np.pi / 2]

    # alpha_z offet = -30 °
    # alpha_z = -30 °
    fake_alpha_z = np.array(
        [0, 1, 0, 0] + [0, 0, -0.5, 0.866] + [0] * 4, np.float32
    )
    pred_alpha_z[0, 0] = torch.from_numpy(fake_alpha_z)

    # theta = -30°
    fake_loc = np.array([1, 0, 3 ** 0.5], np.float32)
    location[0, 0] = torch.from_numpy(fake_loc)
    ret = get_orientation(pred_alpha_z, location, bin_center)
    roty, alpha_x, alpha_z, theta = ret

    assert abs(roty[0, 0] - 0 * np.pi / 180) < 5e-5
    assert abs(alpha_x[0, 0] - -30 * np.pi / 180) < 5e-5
    assert abs(alpha_z[0, 0] - 60 * np.pi / 180) < 5e-5
    assert abs(theta[0, 0] - -30 * np.pi / 180) < 5e-5
