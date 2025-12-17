# -*- coding: utf-8 -*-
# Copyright (c) Horizon Robotics. All rights reserved.

import numpy as np
import pytest
import torch

from hat.data.transforms.real3d import (
    Real3dTargetGenerator,
    angle2multibin,
    format_angle,
    roty2alpha_z,
)
from hat.registry import build_from_registry
from tests.utils import gen_fake_transforms_data


@pytest.mark.parametrize(
    ["size"], [pytest.param([240, 128]), pytest.param([960, 512])]
)
def test_image_transform(size):
    cfg = dict(
        type="ImageTransform",
        size=size,
    )
    transform_image = build_from_registry(cfg)
    data = gen_fake_transforms_data(1920, 1080, "hwc")
    data_res = transform_image(data)
    assert (
        data_res["img"].shape[0] == size[1]
        and data_res["img"].shape[1] == size[0]
    )
    assert "image_transform" in data_res
    assert "M" in data_res["image_transform"]
    assert "original_size" in data_res["image_transform"]
    assert "input_size" in data_res["image_transform"]


def test_image_to_tensor():
    cfg = dict(
        type="ImageToTensor",
        from_numpy=True,
    )
    image_to_tensor = build_from_registry(cfg)
    data = gen_fake_transforms_data(1920, 1080, "hwc")
    data_res = image_to_tensor(data)
    assert isinstance(data_res["img"], torch.Tensor)


@pytest.mark.parametrize(
    ["rgb_input", "use_yuv_v2"],
    [
        pytest.param(True, True),
        pytest.param(True, False),
        pytest.param(False, False),
    ],
)
def test_image_to_yuv(rgb_input, use_yuv_v2):
    cfg = dict(
        type="ImageBgrToYuv444",
        rgb_input=rgb_input,
        use_yuv_v2=use_yuv_v2,
    )
    image_to_tensor = build_from_registry(cfg)
    data = {"img": torch.randint(0, 255, (3, 512, 960))}
    data_res = image_to_tensor(data)
    assert data_res["img"].shape == (3, 512, 960)


@pytest.mark.parametrize(
    ["times"], [pytest.param(1), pytest.param(2), pytest.param(5)]
)
def test_parse_data_real3D_multitask(times):
    cfg = dict(
        type="ParseDataReal3DMultitask",
        times=times,
    )
    parse_data_real3D_multitask = build_from_registry(cfg)
    data = gen_fake_transforms_data(1920, 1080, "hwc")
    data_res = parse_data_real3D_multitask(data)  # noqa: F841


def test_real3d_target_generator():
    head_channels = dict(hm=3, rot=2, dep=1, dim=3, loc_offset=2, wh=2)
    real3d_target_generator = Real3dTargetGenerator(  # noqa: F841
        3,
        1.0,
        (1920, 1080),
        {},
        head_channels,
        undistort=False,
        fisheye=False,
    )


def test_format_angle():
    angle = 3.2
    assert abs(format_angle(angle) - -3.083185) < 1e-5


def test_roty2alpha_z():
    loc = [1, 0, 3 ** 0.5]  # [X,Y,Z] -->theta=-30°
    roty = -30 * np.pi / 180  # -30°
    # alpha_z = 30°
    # alpha_x = -60°
    assert round(roty2alpha_z(roty, loc), 4) == round(30 * np.pi / 180, 4)


def test_angle2multibin():
    angle = 60 * np.pi / 180
    bin_centers = [0, np.pi / 2, np.pi, -np.pi / 2]
    bin_cls, bin_offset = angle2multibin(angle, bin_centers, 0)
    assert bin_cls[1] == 1
    assert abs(bin_offset[1] - (-30 * np.pi / 180)) < 1e-6
