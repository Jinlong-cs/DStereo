# Copyright (c) Horizon Robotics. All rights reserved.
import pytest

from hat.models.task_modules.isp_me import NISP
from tests.utils import gen_fake_torch_randn_data


@pytest.mark.parametrize(
    [
        "in_channels",
        "out_channels",
        "block_num",
        "use_bias",
        "bn_kwargs",
        "groups",
    ],
    [
        pytest.param(3, 3, 0, False, {}, 1),
        pytest.param(3, 16, 0, False, {}, 1),
        pytest.param(3, 3, 1, False, {}, 1),
        pytest.param(3, 16, 1, False, {}, 1),
        pytest.param(3, 3, 2, True, {}, 2),
        pytest.param(3, 16, 2, True, {}, 4),
    ],
)
def test_nisp(
    in_channels,
    out_channels,
    block_num,
    use_bias,
    bn_kwargs,
    groups,
):
    batch_size = 2
    height = 512
    width = 960
    x = gen_fake_torch_randn_data(
        (batch_size, in_channels, height, width),
    )
    model = NISP(
        in_channels, out_channels, block_num, use_bias, bn_kwargs, groups
    )
    data = {}
    data["img"] = x
    y = model(data)
    assert y.shape == (batch_size, out_channels, height, width)
