# -*- coding: utf-8 -*-
# Copyright (c) Horizon Robotics. All rights reserved.
import numpy as np
import torch

from hat.data.transforms.video_classification import (
    JitterScaleVideo,
    NormalizeVideo,
    RandomCropVideo,
    RandomFlipVideo,
    UniformCropVideo,
)


def test_normalizevideo():
    transform = NormalizeVideo(
        mean=(0.45, 0.45, 0.45),
        std=(0.225, 0.225, 0.225),
        tensor_shape=(1, 1, 1, 3),
    )
    image = np.random.randint(0, 250, size=(8, 240, 320, 3)).astype(np.uint8)
    fake_data = {"img": image, "labels": torch.randint(10, (1,))}
    result = transform(fake_data)
    assert result["img"].shape == (3, 8, 240, 320)


def test_jitterscalevideo():
    transform = JitterScaleVideo(
        min_size=224,
        max_size=320,
    )
    image = torch.randn(3, 8, 240, 320)
    fake_data = {"img": image, "labels": torch.randint(10, (1,))}
    result = transform(fake_data)
    assert min(result["img"].shape[2:]) >= 224
    assert min(result["img"].shape[2:]) <= 320


def test_randomcropvideo():
    transform = RandomCropVideo(
        target_size=224,
    )
    image = torch.randn(3, 8, 240, 320)
    fake_data = {"img": image, "labels": torch.randint(10, (1,))}
    result = transform(fake_data)
    assert result["img"].shape == (3, 8, 224, 224)


def test_uniformcropvideo():
    transform = UniformCropVideo(
        target_size=224,
    )
    image = torch.randn(3, 8, 240, 320)
    fake_data = {"img": image, "labels": torch.randint(10, (1,))}
    result = transform(fake_data)
    assert result["img"].shape == (3, 24, 224, 224)


def test_randomflipvideo():
    transform = RandomFlipVideo(
        px=0.5,
    )
    image = torch.randn(3, 8, 240, 320)
    fake_data = {"img": image, "labels": torch.randint(10, (1,))}
    result = transform(fake_data)
    assert result["img"].shape == (3, 8, 240, 320)
