# Copyright (c) Horizon Robotics. All rights reserved.

import numpy as np
import pytest

from hat.data.transforms.faceid import (
    Contrast,
    GaussianBlur,
    JPEGCompress,
    MotionBlur,
    RandomDownSample,
    RandomGray,
    SpatialVariantBrightness,
)
from tests.utils import (
    check,
    check_range,
    check_shape,
    check_type,
    gen_fake_transforms_data,
)


@pytest.mark.parametrize(
    ["p", "only_one_channel"],
    [pytest.param(1, True), pytest.param(0.08, False)],
)
def test_random_gray(p, only_one_channel):
    h = 112
    w = 112
    data = gen_fake_transforms_data(h, w, layout="hwc")
    src_data = data.copy()
    random_gray = RandomGray(p=p, only_one_channel=only_one_channel)
    aug_data = random_gray(data).copy()
    if only_one_channel:
        check(aug_data["img"], check_shape, shape=(h, w, 1))
    else:
        check(aug_data["img"], check_shape, shape=(h, w, 3))
    check(aug_data["img"], check_range, min=0, max=255)
    check(aug_data["img"], check_type, instance=np.ndarray)
    assert aug_data["img"].dtype == src_data["img"].dtype


@pytest.mark.parametrize(
    ["p", "max_quality", "min_quality"],
    [
        pytest.param(1, 95, 30),
        pytest.param(0.2, 95, 30),
    ],
)
def test_jpeg_compress(p, max_quality, min_quality):
    h = 112
    w = 112
    data = gen_fake_transforms_data(h, w, layout="hwc")
    src_data = data.copy()
    jpeg_compress = JPEGCompress(p, max_quality, min_quality)
    aug_data = jpeg_compress(data).copy()
    check(aug_data["img"], check_shape, shape=(h, w, 3))
    check(aug_data["img"], check_range, min=0, max=255)
    check(aug_data["img"], check_type, instance=np.ndarray)
    assert aug_data["img"].dtype == src_data["img"].dtype


@pytest.mark.parametrize(
    ["p", "brightness", "max_template_type", "online_template"],
    [
        pytest.param(1, 0.5, 3, False),
    ],
)
def test_random_spatial_variant_brightness(
    p, brightness, max_template_type, online_template
):
    h = 112
    w = 112
    data = gen_fake_transforms_data(h, w, layout="hwc")
    src_data = data.copy()
    random_spatial_variant_brightness = SpatialVariantBrightness(
        p=p,
        brightness=brightness,
        max_template_type=max_template_type,
        online_template=online_template,
    )
    aug_data = random_spatial_variant_brightness(data).copy()
    check(aug_data["img"], check_shape, shape=(h, w, 3))
    check(aug_data["img"], check_range, min=0, max=255)
    check(aug_data["img"], check_type, instance=np.ndarray)
    assert aug_data["img"].dtype == src_data["img"].dtype


@pytest.mark.parametrize(
    ["p", "kernel_size_min", "kernel_size_max", "sigma_min", "sigma_max"],
    [
        pytest.param(1, 2, 9, 0, 0),
    ],
)
def test_random_gaussian_blur(
    p, kernel_size_min, kernel_size_max, sigma_min, sigma_max
):
    h = 112
    w = 112
    data = gen_fake_transforms_data(h, w, layout="hwc")
    src_data = data.copy()
    random_gaussian_blur = GaussianBlur(
        p=p,
        kernel_size_max=kernel_size_max,
        kernel_size_min=kernel_size_min,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    aug_data = random_gaussian_blur(data).copy()
    check(aug_data["img"], check_shape, shape=(h, w, 3))
    check(aug_data["img"], check_range, min=0, max=255)
    check(aug_data["img"], check_type, instance=np.ndarray)
    assert aug_data["img"].dtype == src_data["img"].dtype


@pytest.mark.parametrize(
    ["p", "length_min", "length_max", "angle_min", "angle_max"],
    [
        pytest.param(1, 9, 18, 1, 359),
    ],
)
def test_random_motion_blur(p, length_min, length_max, angle_min, angle_max):
    h = 112
    w = 112
    data = gen_fake_transforms_data(h, w, layout="hwc")
    src_data = data.copy()
    random_motion_blur = MotionBlur(
        p=p,
        length_min=length_min,
        length_max=length_max,
        angle_min=angle_min,
        angle_max=angle_max,
    )
    aug_data = random_motion_blur(data).copy()
    check(aug_data["img"], check_shape, shape=(h, w, 3))
    check(aug_data["img"], check_range, min=0, max=255)
    check(aug_data["img"], check_type, instance=np.ndarray)
    assert aug_data["img"].dtype == src_data["img"].dtype


@pytest.mark.parametrize(
    ["p", "data_shape", "min_downsample_width", "inter_method"],
    [
        pytest.param(1.0, (3, 112, 112), 60, 1),
        pytest.param(0.08, (3, 112, 112), 30, 2),
    ],
)
def test_random_downsample(p, data_shape, min_downsample_width, inter_method):
    h = 112
    w = 112
    data = gen_fake_transforms_data(h, w, layout="hwc")
    src_data = data.copy()
    random_downsample = RandomDownSample(
        p=p,
        data_shape=data_shape,
        min_downsample_width=min_downsample_width,
        inter_method=inter_method,
    )
    aug_data = random_downsample(data).copy()
    check(aug_data["img"], check_shape, shape=(h, w, 3))
    check(aug_data["img"], check_range, min=0, max=255)
    check(aug_data["img"], check_type, instance=np.ndarray)
    assert aug_data["img"].dtype == src_data["img"].dtype


@pytest.mark.parametrize(
    ["p", "contrast"],
    [
        pytest.param(1.0, 0.5),
    ],
)
def test_random_contrast(p, contrast):
    h = 112
    w = 112
    data = gen_fake_transforms_data(h, w, layout="hwc")
    src_data = data.copy()
    random_contrast = Contrast(
        p=p,
        contrast=contrast,
    )
    aug_data = random_contrast(data).copy()
    check(aug_data["img"], check_shape, shape=(h, w, 3))
    check(aug_data["img"], check_range, min=0, max=255)
    check(aug_data["img"], check_type, instance=np.ndarray)
    assert aug_data["img"].dtype == src_data["img"].dtype
