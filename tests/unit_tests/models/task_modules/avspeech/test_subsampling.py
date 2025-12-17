# Copyright (c) Horizon Robotics. All rights reserved.
import torch

from hat.models.task_modules.avspeech.subsampling import (
    CausalSubsampling2,
    Conv2dSubsampling4,
    Conv2dSubsampling8,
)


def test_conv2d_subsampling4():
    x = torch.randn(2, 1, 100, 80)
    subsampling = Conv2dSubsampling4(80, 256, 0.1)
    y = subsampling(x)
    assert y.shape == (2, 256, 1, 24)  # (100 - 3) // 4 = 24


def test_conv2d_subsampling8():
    x = torch.randn(2, 1, 100, 80)
    subsampling = Conv2dSubsampling8(80, 256, 0.1)
    y = subsampling(x)
    assert y.shape == (2, 256, 1, 11)  # (100 - 7) // 8 = 11


def test_causal_subsampling2():
    x = torch.randn(2, 80, 1, 100)
    subsampling = CausalSubsampling2(80, 256, 0.1)
    y = subsampling(x)
    assert y.shape == (2, 256, 1, 50)  # 100 // 2 = 50
