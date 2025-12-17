# Copyright (c) Horizon Robotics. All rights reserved.
import pytest
import torch

from hat.models.task_modules.lutnet import LutNet


@pytest.mark.parametrize(
    [
        "in_channels",
        "output_bit",
        "lut_size",
        "down_size",
        "pregamma",
        "track_coarse_steps",
        "norm_symmetric",
        "alpha",
    ],
    [
        pytest.param(3, 16, 64, None, None, None, True, 0.9),
        pytest.param(3, 16, 64, None, None, None, False, 0.9),
        pytest.param(3, 16, 64, (256, 480), None, 10, True, 0.9),
        pytest.param(3, 16, 64, (256, 256), None, 10, True, 0.9),
        pytest.param(3, 16, 64, (256, 256), 1 / 3.0, 10, True, 0.9),
    ],
)
def test_lutnet(
    in_channels,
    output_bit,
    lut_size,
    down_size,
    pregamma,
    track_coarse_steps,
    norm_symmetric,
    alpha,
):
    model = LutNet(
        in_channels=in_channels,
        output_bit=output_bit,
        lut_size=lut_size,
        down_size=down_size,
        pregamma=pregamma,
        track_coarse_steps=track_coarse_steps,
        norm_symmetric=norm_symmetric,
        alpha=alpha,
    )
    batch_size = 4
    height = 512
    width = 960
    x = torch.rand(batch_size, in_channels, height, width)
    y = model(x)
    assert y.shape == (batch_size, in_channels, height, width)
    assert y.min() >= 0 and y.max() <= 1 or y.min() >= -1 and y.max() <= 1
