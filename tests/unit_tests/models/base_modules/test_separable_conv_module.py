import numpy as np
import torch

from hat.models.base_modules.separable_conv_module import (
    SeparableConvModule2d,
    SeparableGroupConvModule2d,
)


def test_SeparableConvModule2d():
    data_shape = np.random.randint(10, 20, size=4)
    fake_data = torch.tensor(
        np.random.random(size=data_shape) * 2 - 1,
        dtype=torch.float,
    )
    sc_module = SeparableConvModule2d(
        fake_data.shape[1],
        20,
        (3, 3),
    )
    output = sc_module(fake_data)
    assert output is not None


def test_SeparableGroupConvModule2d():
    data_shape = np.random.randint(10, 20, size=4)
    fake_data = torch.tensor(
        np.random.random(size=data_shape) * 2 - 1,
        dtype=torch.float,
    )
    sc_module = SeparableGroupConvModule2d(
        fake_data.shape[1],
        20,
        (3, 3),
    )
    output = sc_module(fake_data)
    assert output is not None
