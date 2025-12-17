# -*- coding: utf-8 -*-
# Copyright (c) Horizon Robotics. All rights reserved.
import numpy as np
import pytest
import torch
from PIL import Image

from hat.data.transforms.classification import TimmMixup, TimmTransforms
from hat.registry import build_from_registry
from tests.utils import gen_fake_transforms_data

try:
    import timm
except ImportError:
    timm = None


def test_convert_layer():
    for keys in [None, ["img"], ["imgs"]]:
        data = {}
        h, w, c = 1080, 1920, 3
        data["img"] = gen_fake_transforms_data(
            w, h, "hwc", return_tensor=True
        )["img"]
        data["imgs"] = [
            gen_fake_transforms_data(w, h, "hwc", return_tensor=True)["img"],
            gen_fake_transforms_data(w, h, "hwc", return_tensor=True)["img"],
        ]
        cfg = dict(type="ConvertLayout", hwc2chw=True, keys=keys)
        convert_layer = build_from_registry(cfg)
        data_res = convert_layer(data)
        if keys is None:
            keys = ["img"]
        for key in keys:
            assert key in data
            if isinstance(data_res[key], list):
                for img in data_res[key]:
                    img_c, img_h, img_w = img.shape
                    assert img_c == c and img_h == h and img_w == w
            if isinstance(data_res[key], torch.Tensor):
                img = data_res[key]
                img_c, img_h, img_w = img.shape
                assert img_c == c and img_h == h and img_w == w


@pytest.mark.skipif(timm is None, reason="timm is required")
def test_timm_transforms():
    transform = TimmTransforms(
        input_size=224,
        is_training=True,
        color_jitter=0.4,
        auto_augment="rand-m9-mstd0.5-inc1",
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
        mean=[0, 0, 0],
        std=[1.0, 1.0, 1.0],
        interpolation="bicubic",
    )
    image = Image.fromarray(
        np.random.randint(0, 250, size=(256, 334, 3)).astype(np.uint8)
    )
    fake_data = {"img": image, "labels": torch.randint(10, (1,))}
    result = transform(fake_data)
    assert result["img"].shape == (3, 224, 224)


@pytest.mark.skipif(timm is None, reason="timm is required")
def test_timm_mixup():
    mixup = TimmMixup(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        cutmix_minmax=None,
        prob=1.0,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=0.1,
        num_classes=1000,
    )
    fake_data = {
        "img": torch.randn(10, 3, 224, 224).cuda(),
        "labels": torch.randint(10, (10,)).cuda(),
    }
    result = mixup(fake_data)
    assert result["img"].shape == (10, 3, 224, 224)
