# -*- coding: utf-8 -*-
# Copyright (c) Horizon Robotics. All rights reserved.


import pytest
import torch

from hat.data.transforms.elevation import (
    CropElevation,
    NormalizeElevation,
    PrepareDataElevation,
    ResizeElevation,
    ToTensorElevation,
)
from hat.utils.package_helper import check_packages_available
from tests.utils import (
    check,
    check_range,
    check_shape,
    check_type,
    gen_fake_np_data,
    gen_fake_pil_data,
)


def generate_fake_data(h=1080, w=1920, frames=3):

    data = dict()
    data["pil_imgs"] = [
        [
            gen_fake_pil_data(
                (h, w, 3), mode="RGB", dtype="uint8", low=0, high=255
            )
        ]
        for j in range(frames)
    ]

    data["color_imgs"] = [
        [
            gen_fake_pil_data(
                (h, w, 3), mode="RGB", dtype="uint8", low=0, high=255
            )
        ]
        for j in range(frames)
    ]

    data["gt_gamma"] = [gen_fake_np_data((h, w), dtype="float32")]
    data["gt_height"] = [gen_fake_np_data((h, w), dtype="float32")]
    data["gt_depth"] = [gen_fake_np_data((h, w), dtype="float32")]

    data["mask"] = [
        [gen_fake_pil_data((h, w), mode="I")] for j in range(frames)
    ]
    data["obj_mask"] = [
        [gen_fake_pil_data((h, w), mode="I")] for j in range(frames)
    ]
    data["ground_mask"] = [
        [gen_fake_pil_data((h, w), mode="I")] for j in range(frames)
    ]

    data["intrinsics"] = gen_fake_np_data((3, 3), dtype="float32")
    data["camera_high"] = [gen_fake_np_data((1), dtype="float32")]
    data["timestamp"] = gen_fake_np_data((1), dtype="float32")

    data["ground_norm"] = [
        gen_fake_np_data((3, 1), dtype="float32"),
        gen_fake_np_data((3, 1), dtype="float32"),
    ]
    data["ground_homo"] = [
        gen_fake_np_data((1, 3, 3), dtype="float32"),
        gen_fake_np_data((1, 3, 3), dtype="float32"),
    ]
    data["rotation"] = [
        gen_fake_np_data((3, 3), dtype="float32"),
        gen_fake_np_data((3, 3), dtype="float32"),
    ]
    data["transition"] = [
        gen_fake_np_data((3, 1), dtype="float32"),
        gen_fake_np_data((3, 1), dtype="float32"),
    ]
    return data


@pytest.mark.skipif(
    not check_packages_available("torchvision", raise_exception=False),
    reason="need torchvision",
)
@pytest.mark.parametrize(
    [
        "size",
        "interpolation",
    ],
    [
        pytest.param((540, 960), "nearest"),
        pytest.param((540, 960), "bilinear"),
    ],
)
def test_resize(size, interpolation):
    resize_elevation = ResizeElevation(size=size, interpolation=interpolation)

    h, w = 1080, 1920
    frames = 3
    data = generate_fake_data(h, w, frames)

    resize_data = resize_elevation(data)

    check(resize_data["pil_imgs"], check_shape, shape=size)
    check(resize_data["mask"], check_shape, shape=size)
    check(resize_data["gt_gamma"], check_shape, shape=size)
    check(resize_data["gt_height"], check_shape, shape=size)
    check(resize_data["gt_depth"], check_shape, shape=size)

    check(resize_data["intrinsics"], check_shape, shape=(3, 3))
    check(resize_data["ground_norm"], check_shape, shape=(3, 1))
    check(resize_data["ground_homo"], check_shape, shape=(1, 3, 3))
    check(resize_data["camera_high"], check_shape, shape=(1,))
    check(resize_data["timestamp"], check_shape, shape=(1,))
    check(resize_data["rotation"], check_shape, shape=(3, 3))
    check(resize_data["transition"], check_shape, shape=(3, 1))


@pytest.mark.skipif(
    not check_packages_available("torchvision", raise_exception=False),
    reason="need torchvision",
)
@pytest.mark.parametrize(
    ["height", "width", "top", "left", "resized"],
    [
        pytest.param(512, 960, None, None, True),
        pytest.param(512, 960, 28, None, True),
        pytest.param(512, 960, 0, None, True),
        pytest.param(512, 960, [0], None, True),
        pytest.param(512, 960, None, None, False),
        pytest.param(512, 960, 28, None, False),
        pytest.param(512, 960, 0, None, False),
        pytest.param(512, 960, [0], None, False),
    ],
)
def test_crop_elevation(height, width, top, left, resized):
    crop_elevation = CropElevation(height, width, top, left, resized)

    h, w = 540, 960
    frames = 3
    data = generate_fake_data(h, w, frames)
    data["size"] = [(540, 960)]
    crop_data = crop_elevation(data)

    check(crop_data["pil_imgs"], check_shape, shape=(height, width))
    check(crop_data["mask"], check_shape, shape=(height, width))
    check(crop_data["gt_gamma"], check_shape, shape=(height, width))
    check(crop_data["gt_height"], check_shape, shape=(height, width))
    check(crop_data["gt_depth"], check_shape, shape=(height, width))

    check(crop_data["intrinsics"], check_shape, shape=(3, 3))
    check(crop_data["ground_norm"], check_shape, shape=(3, 1))
    check(crop_data["ground_homo"], check_shape, shape=(1, 3, 3))
    check(crop_data["camera_high"], check_shape, shape=(1,))
    check(crop_data["timestamp"], check_shape, shape=(1,))
    check(crop_data["rotation"], check_shape, shape=(3, 3))
    check(crop_data["transition"], check_shape, shape=(3, 1))


@pytest.mark.skipif(
    not check_packages_available("torchvision", raise_exception=False),
    reason="need torchvision",
)
@pytest.mark.parametrize(
    ["mean", "std", "gamma_scale"],
    [pytest.param(128, 128, 1.0), pytest.param(128, 128, 1000.0)],
)
def test_normalize_elevation(mean, std, gamma_scale):
    to_tensor_elevation = ToTensorElevation(True, True)
    normlaize_elevation = NormalizeElevation(mean, std, gamma_scale)

    h, w = 540, 960
    frames = 3
    data = generate_fake_data(h, w, frames)
    data["size"] = [(540, 960)]

    tensor_data = to_tensor_elevation(data)
    gamma = tensor_data["gt_gamma"]
    normal_data = normlaize_elevation(tensor_data)

    assert (
        normal_data["gt_gamma"][0].data.sum()
        == (gamma[0] * gamma_scale).sum().data
    )


@pytest.mark.skipif(
    not check_packages_available("torchvision", raise_exception=False),
    reason="need torchvision",
)
@pytest.mark.parametrize(
    ["to_yuv", "with_color_imgs", "use_yuv_v2"],
    [
        pytest.param(True, True, True),
        pytest.param(True, False, False),
        pytest.param(False, True, False),
        pytest.param(False, False, False),
    ],
)
def test_to_tensor_elevation(to_yuv, with_color_imgs, use_yuv_v2):
    to_tensor_elevation = ToTensorElevation(
        to_yuv, with_color_imgs, use_yuv_v2
    )

    h, w = 540, 960
    frames = 3
    data = generate_fake_data(h, w, frames)
    data["size"] = [(540, 960)]

    tensor_data = to_tensor_elevation(data)

    check(tensor_data["imgs"], check_type, instance=torch.Tensor)
    if with_color_imgs:
        check(tensor_data["color_imgs"], check_type, instance=torch.Tensor)
    check(tensor_data["mask"], check_type, instance=torch.Tensor)
    check(tensor_data["gt_gamma"], check_type, instance=torch.Tensor)
    check(tensor_data["gt_height"], check_type, instance=torch.Tensor)
    check(tensor_data["gt_depth"], check_type, instance=torch.Tensor)

    check(tensor_data["intrinsics"], check_type, instance=torch.Tensor)
    check(tensor_data["ground_norm"], check_type, instance=torch.Tensor)
    check(tensor_data["ground_homo"], check_type, instance=torch.Tensor)
    check(tensor_data["camera_high"], check_type, instance=torch.Tensor)
    check(tensor_data["timestamp"], check_type, instance=torch.Tensor)
    check(tensor_data["rotation"], check_type, instance=torch.Tensor)
    check(tensor_data["transition"], check_type, instance=torch.Tensor)
    assert "pil_imgs" not in tensor_data

    if to_yuv:
        check(tensor_data["imgs"], check_range, min=0, max=255.0)
    else:
        check(tensor_data["imgs"], check_range, min=0, max=1.0)

    if with_color_imgs:
        assert "color_imgs" in tensor_data
        check(tensor_data["color_imgs"], check_range, min=0, max=1.0)


@pytest.mark.skipif(
    not check_packages_available("torchvision", raise_exception=False),
    reason="need torchvision",
)
@pytest.mark.parametrize(
    ["organize_data_type"],
    [
        pytest.param("elevation_train"),
        pytest.param("elevation_val"),
        pytest.param("inference"),
    ],
)
def test_to_prepare_data3dv(organize_data_type):
    # generate fake data
    to_tensor_elevation = ToTensorElevation(True, True)
    prepare_elevation = PrepareDataElevation(organize_data_type)

    h, w = 1080, 1920
    frames = 3
    data = generate_fake_data(h, w, frames)
    data["size"] = [(h, w)]

    tensor_data = to_tensor_elevation(data)
    prepare_data = prepare_elevation(tensor_data)

    check(prepare_data["img"], check_shape, shape=(1, 3, h, w))
    if organize_data_type == "elevation_train":
        check(prepare_data["extra_img"], check_shape, shape=(1, 3, h, w))

    check(prepare_data["color_imgs"], check_shape, shape=(1, 3, h, w))
    check(prepare_data["mask"], check_shape, shape=(1, h, w))
    check(prepare_data["ground_mask"], check_shape, shape=(1, h, w))
    check(prepare_data["obj_mask"], check_shape, shape=(1, h, w))
    check(prepare_data["gt_gamma"], check_shape, shape=(1, h, w))
    check(prepare_data["gt_height"], check_shape, shape=(1, h, w))
    check(prepare_data["gt_depth"], check_shape, shape=(1, h, w))
