import numpy as np
import pytest
import torch

from hat.data.transforms.avspeech.images import (
    BrightnessContrast,
    CoarseDropout,
    FancyPCA,
    HueSaturateValue,
    ImageListNormalize,
    ImageListRandomCrop,
    ImageListRandomFlip,
    ImageListStack,
    ImageListToYUV444,
    TimeMask,
    ToGray,
)
from hat.utils.package_helper import check_packages_available


@pytest.mark.skipif(
    not check_packages_available("torchvision", raise_exception=False),
    reason="need torchvision",
)
def test_images():
    # 构造数据
    batch_size = 2
    image_shape = (96, 96, 3)  # H*W*C
    src_shape = (batch_size, *image_shape)  # T*H*W*C
    tgt_shape = (2, 3, 96, 96)  # T*C*H*W
    images = np.random.uniform(0, 256, src_shape)
    data = {"images": images}

    # 测试 BrightnessContrast
    data = BrightnessContrast(
        brightness_limit=0.4,
        contrast_limit=0.4,
        brightness_by_max=False,
        p=1.0,
    )(data)
    assert len(data["images"]) == batch_size
    assert (image.shape == image_shape for image in data["images"])

    # 测试 CoarseDropout
    data = CoarseDropout(
        max_holes=1,
        max_height=96,
        max_width=30,
        fill_value=[183, 197, 248],
        p=1.0,
    )(data)
    assert len(data["images"]) == batch_size
    assert (image.shape == image_shape for image in data["images"])

    # 测试 HueSaturateValue
    data = HueSaturateValue(
        hue_shift_limit=10,
        sat_shift_limit=30,
        val_shift_limit=0,
        p=1.0,
    )(data)
    assert len(data["images"]) == batch_size
    assert (image.shape == image_shape for image in data["images"])

    # 测试 FancyPCA
    data = FancyPCA(alpha=0.8, p=1.0)(data)
    assert len(data["images"]) == batch_size
    assert (image.shape == image_shape for image in data["images"])

    # 测试 ToGray
    data = ToGray(p=1.0, rgb_data=False)(data)
    assert len(data["images"]) == batch_size
    assert (image.shape == image_shape for image in data["images"])

    # 测试 ImageListStack
    data = ImageListStack(hwc2chw=True)(data)
    assert data["images"].shape == tgt_shape
    assert "images_lens" in data

    # 测试 ImageListRandomFlip
    data = ImageListRandomFlip(p=1.0)(data)
    assert data["images"].shape == tgt_shape

    # 测试 ImageListRandomCrop
    data = ImageListRandomCrop(
        p=1.0,
        scale=(0.8, 1.0),
        ratio=(1.0, 1.0),
        size=(96, 96),
    )(data)
    assert data["images"].shape == tgt_shape

    # 测试 TimeMask
    data = TimeMask(
        p=1.0,
        max_frame=30,
        num_mask=2,
        replace_with_zero=False,
    )(data)
    assert data["images"].shape == tgt_shape

    # 测试 ImageListToYUV444
    data = ImageListToYUV444(layout="bgr")(data)
    assert data["images"].shape == tgt_shape

    # 测试 ImageListNormalize
    data = ImageListNormalize(
        mean=[128.0, 128.0, 128.0],
        std=[128.0, 128.0, 128.0],
    )(data)
    assert data["images"].shape == tgt_shape
    assert data["images"].dtype == torch.float32


if __name__ == "__main__":
    test_images()
