import numpy as np
import pytest
import torch

from hat.models.base_modules.basic_vargconvnet_module import VargConvNetBlock


@pytest.mark.parametrize(
    ["in_channels", "out_channels", "stride", "groups"],
    [
        pytest.param(32, 32, 1, 8),
        pytest.param(16, 16, 2, 8),
        pytest.param(16, 16, 1, 4),
    ],
)
def test_basic_vargblockv2(in_channels, out_channels, stride, groups):
    data_shape = [1, in_channels, 32, 32]
    fake_data = torch.tensor(
        np.random.random(size=data_shape),
        dtype=torch.float,
    )
    module = VargConvNetBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        stride=stride,
        bn_kwargs={},
        groups=groups,
    )
    output = module(fake_data)
    assert output is not None
