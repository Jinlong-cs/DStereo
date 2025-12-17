# -*- coding: utf-8 -*-
# Copyright (c) Horizon Robotics. All rights reserved.

import numpy as np
import pytest

from hat.data.transforms.functional_img import (
    demosaic,
    image_normalize,
    image_pad,
    imresize,
    imresize_warp_when_nearest,
    random_flip,
)
from tests.utils import gen_fake_transforms_data

try:
    import hat_sim
except Exception:
    hat_sim = None


@pytest.mark.parametrize("img_h, img_w", [(320, 512)])
@pytest.mark.parametrize("pattern", ["GRBG", "RGGB", "BGGR", "GBRG"])
def test_demosaic(img_h, img_w, pattern):
    if hat_sim is None:
        return

    assert img_h > 4
    assert img_w > 4
    in_img = np.random.randint(
        0, 255, (img_h, img_w, 1), dtype="uint8"
    ).astype(np.float32)
    out_img = demosaic(in_img, img_h, img_w, pattern)

    assert out_img.shape[2] == 3
    assert out_img.shape[0] == img_h
    assert out_img.shape[1] == img_w


@pytest.mark.skipif(hat_sim is None, reason="need hat-sim")
@pytest.mark.parametrize("w, h", [(125, 100), [250, 200]])
@pytest.mark.parametrize("layout", ["hwc", "chw", "hw"])
@pytest.mark.parametrize("divisor", [1, 32])
@pytest.mark.parametrize("keep_ratio", [True])
@pytest.mark.parametrize("return_scale", [True])
@pytest.mark.parametrize("interpolation", ["nearest", "bilinear"])
def test_imresize_warp_when_nearest(
    w,
    h,
    layout,
    divisor,
    keep_ratio,
    return_scale,
    interpolation,
):
    # generate fake img
    src_w, src_h = 200, 128
    img = gen_fake_transforms_data(src_w, src_h, layout)
    img = img["img"]
    # Test layout in ["hwc", "chw", "hw"]
    with pytest.raises(AssertionError):
        resized_img, w_scale, h_scale = imresize_warp_when_nearest(
            img,
            w,
            h,
            "nchw",
            divisor,
            keep_ratio,
            return_scale,
            interpolation,
        )
    resized_img, w_scale, h_scale = imresize_warp_when_nearest(
        img,
        w,
        h,
        layout,
        divisor,
        keep_ratio,
        return_scale,
        interpolation,
    )
    if layout == "hwc":
        new_h, new_w, _ = resized_img.shape
    elif layout == "chw":
        _, new_h, new_w = resized_img.shape
    elif layout == "hw":
        new_h, new_w = resized_img.shape

    assert isinstance(resized_img, np.ndarray)
    # Test new_w and new_h can be divisible by divisor
    assert np.isclose(int(new_w / divisor), new_w / divisor)
    assert np.isclose(int(new_h / divisor), new_h / divisor)
    # Test new_w and new_h are we want
    assert int(h_scale * src_h) == new_h
    assert int(w_scale * src_w) == new_w
    # Test w_scale is equal to h_scale when keep_ratio and divisor is 1
    if keep_ratio and divisor == 1:
        assert np.isclose(w_scale, h_scale), (w_scale, h_scale)


@pytest.mark.skipif(hat_sim is None, reason="need hat-sim")
@pytest.mark.parametrize("w, h", [(125, 100), [250, 200]])
@pytest.mark.parametrize("layout", ["hwc", "chw", "hw"])
@pytest.mark.parametrize("divisor", [1, 32])
@pytest.mark.parametrize("keep_ratio", [True, False])
@pytest.mark.parametrize("return_scale", [True])
@pytest.mark.parametrize("interpolation", ["nearest", "bilinear"])
@pytest.mark.parametrize("raw_scaler_enable", [True, False])
@pytest.mark.parametrize("sample1c_enable", [True, False])
@pytest.mark.parametrize("raw_pattern", ["RGGB"])
def test_imresize(
    w,
    h,
    layout,
    divisor,
    keep_ratio,
    return_scale,
    interpolation,
    raw_scaler_enable,
    sample1c_enable,
    raw_pattern,
):
    # generate fake img
    src_w, src_h = 200, 128
    img = gen_fake_transforms_data(src_w, src_h, layout)
    img = img["img"]
    # Test layout in ["hwc", "chw", "hw"]
    with pytest.raises(AssertionError):
        resized_img, _, w_scale, h_scale = imresize(
            img,
            w,
            h,
            "nchw",
            divisor,
            keep_ratio,
            return_scale,
            interpolation,
            raw_scaler_enable,
            sample1c_enable,
            raw_pattern,
        )
    resized_img, _, w_scale, h_scale = imresize(
        img,
        w,
        h,
        layout,
        divisor,
        keep_ratio,
        return_scale,
        interpolation,
        raw_scaler_enable,
        sample1c_enable,
        raw_pattern,
    )
    if layout == "hwc":
        new_h, new_w, _ = resized_img.shape
    elif layout == "chw":
        _, new_h, new_w = resized_img.shape
    elif layout == "hw":
        new_h, new_w = resized_img.shape

    assert isinstance(resized_img, np.ndarray)
    # Test new_w and new_h can be divisible by divisor
    assert np.isclose(int(new_w / divisor), new_w / divisor)
    assert np.isclose(int(new_h / divisor), new_h / divisor)
    # Test new_w and new_h are we want
    assert int(h_scale * src_h) == new_h
    assert int(w_scale * src_w) == new_w
    # Test w_scale is equal to h_scale when keep_ratio and divisor is 1
    if keep_ratio and divisor == 1:
        assert np.isclose(w_scale, h_scale), (w_scale, h_scale)


@pytest.mark.parametrize("return_tensor", [True, False])
@pytest.mark.parametrize("layout", ["hwc", "chw", "hw"])
@pytest.mark.parametrize("px", [0, 1])
@pytest.mark.parametrize("py", [0, 1])
@pytest.mark.parametrize("raw_pattern", ["GRBG", "RGGB"])
def test_random_flip(return_tensor, layout, px, py, raw_pattern):
    # generate fake img
    src_w, src_h = 200, 128
    img = gen_fake_transforms_data(
        src_w, src_h, layout, return_tensor=return_tensor
    )
    img = img["img"]
    # Test layout in ["hwc", "chw", "hw"]
    with pytest.raises(AssertionError):
        flipped_img, _, flipped_pattern = random_flip(
            img, "nchw", px, py, raw_pattern
        )
    flipped_img, _, flipped_pattern = random_flip(
        img, layout, px, py, raw_pattern
    )
    flipped_flipped_img, _, flipped_flipped_pattern = random_flip(
        flipped_img, layout, px, py, flipped_pattern
    )
    assert flipped_img.shape == img.shape
    assert flipped_flipped_img.shape == img.shape
    assert (flipped_flipped_img == img).all()
    assert flipped_flipped_pattern == raw_pattern


@pytest.mark.parametrize("return_tensor", [False, True])
@pytest.mark.parametrize(
    "layout, shape, pad_val",
    [
        ("hwc", None, 10),
        ("hwc", (250, 250), 10),
        ("hwc", (250, 250, 3), 10),
        ("hwc", (250, 250, 3), (10, 20, 30)),
        ("chw", None, 10),
        ("chw", (250, 250), 10),
        ("chw", (3, 250, 250), 10),
        ("chw", (3, 250, 250), (10, 20, 30)),
        ("hw", None, 10),
        ("hw", (250, 250), 10),
    ],
)
@pytest.mark.parametrize("divisor", [1, 32])
def test_image_pad(return_tensor, layout, shape, pad_val, divisor):
    # generate fake img
    src_w, src_h = 200, 128
    img = gen_fake_transforms_data(
        src_w, src_h, layout, return_tensor=return_tensor
    )
    img = img["img"]
    # Test layout in ["hwc", "chw", "hw"]
    with pytest.raises(AssertionError):
        pad_image = image_pad(img, "nchw", shape, divisor, pad_val)
    pad_image = image_pad(img, layout, shape, divisor, pad_val)
    if layout == "hwc":
        pad_h, pad_w, _ = pad_image.shape
        assert np.isclose(int(pad_w / divisor), pad_w / divisor)
        assert np.isclose(int(pad_h / divisor), pad_h / divisor)
        assert (img == pad_image[:src_h, :src_w, :]).all()
    elif layout == "chw":
        _, pad_h, pad_w = pad_image.shape
        assert np.isclose(int(pad_w / divisor), pad_w / divisor)
        assert np.isclose(int(pad_h / divisor), pad_h / divisor)
        assert (img == pad_image[:, :src_h, :src_w]).all()
    elif layout == "hw":
        pad_h, pad_w = pad_image.shape
        assert np.isclose(int(pad_w / divisor), pad_w / divisor)
        assert np.isclose(int(pad_h / divisor), pad_h / divisor)
        assert (img == pad_image[:src_h, :src_w]).all()


@pytest.mark.parametrize("return_tensor", [False, True])
@pytest.mark.parametrize("layout", ["hwc", "chw"])
@pytest.mark.parametrize("mean", [0.0, [0.0, 0.0, 0.0]])
@pytest.mark.parametrize("std", [1.0, [1.0, 1.0, 1.0]])
def test_image_normalize(return_tensor, layout, mean, std):
    # generate fake img
    src_w, src_h = 200, 128
    img = gen_fake_transforms_data(
        src_w, src_h, layout, return_tensor=return_tensor
    )
    img = img["img"]
    # Test layout in ["hwc", "chw"]
    with pytest.raises(AssertionError):
        norm_img = image_normalize(img, mean, std, "nchw")
    # Test std != 0
    with pytest.raises(ValueError):
        norm_img = image_normalize(img, mean, 0.0, layout)
    norm_img = image_normalize(img, mean, std, layout)
    assert isinstance(norm_img, type(img))
    assert norm_img.shape == img.shape


if __name__ == "__main__":
    pytest.main(["-s", __file__])
