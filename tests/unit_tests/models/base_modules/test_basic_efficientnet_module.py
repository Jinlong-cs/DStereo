import collections

import numpy as np
import pytest
import torch
import torch.nn as nn

from hat.models.base_modules.basic_efficientnet_module import (
    MBConvBlock,
    SEBlock,
)
from hat.registry import build_from_registry


@pytest.mark.parametrize(
    ["act_layer"],
    [  # ReLU
        pytest.param(dict(type="ReLU", inplace=True)),
        # ReLU6
        pytest.param(dict(type="ReLU6", inplace=True)),
        # Swish
        pytest.param(dict(type="SiLU", inplace=True)),
        # Swish
        pytest.param(dict(type=nn.SiLU, inplace=True)),
    ],
)
def test_seblock(act_layer):
    data_shape = np.random.randint(10, 20, size=4)
    fake_data = torch.tensor(
        np.random.random(size=data_shape),
        dtype=torch.float,
    )

    se_module = SEBlock(
        in_channels=fake_data.shape[1],
        num_squeezed_channels=fake_data.shape[1] * 2,
        out_channels=fake_data.shape[1],
        act_layer=build_from_registry(act_layer),
    )
    output = se_module(fake_data)

    assert output is not None


@pytest.mark.parametrize(
    ["use_se_block", "act_layer"],
    [  # Not Use SEBlock, ReLU
        pytest.param(False, dict(type="ReLU", inplace=True)),
        # Use SEBlock, ReLU6
        pytest.param(True, dict(type="ReLU6", inplace=True)),
        # Use SEBlock, SiLU
        pytest.param(True, dict(type="SiLU", inplace=True)),
        # Use SEBlock, SiLU
        pytest.param(True, dict(type=nn.SiLU, inplace=True)),
    ],
)
def test_mbconvblock(use_se_block, act_layer):
    data_shape = np.random.randint(10, 20, size=4)
    fake_data = torch.tensor(
        np.random.random(size=data_shape),
        dtype=torch.float,
    )
    BlockArgs = collections.namedtuple(
        "BlockArgs",
        [
            "kernel_size",
            "num_repeat",
            "in_filters",
            "out_filters",
            "expand_ratio",
            "id_skip",
            "strides",
            "se_ratio",
        ],
    )
    fake_block_args = BlockArgs(
        kernel_size=3,
        num_repeat=2,
        in_filters=fake_data.shape[1],
        out_filters=24,
        expand_ratio=6,
        id_skip=True,
        strides=2,
        se_ratio=0.25,
    )

    mbconv_module = MBConvBlock(
        fake_block_args,
        bn_kwargs={},
        use_se_block=use_se_block,
        act_layer=build_from_registry(act_layer),
    )

    output = mbconv_module(fake_data)

    assert output is not None

    output = mbconv_module(fake_data, drop_connect_rate=0.2)

    assert output is not None


if __name__ == "__main__":
    pytest.main(["-s", __file__])
