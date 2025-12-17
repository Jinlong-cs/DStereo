import numpy as np
import torch

from hat.models.base_modules.basic_resnet_module import (
    BasicResBlock,
    BottleNeck,
)


def test_BasicResBlock():
    data_shape = np.random.randint(10, 20, size=4)
    fake_data = torch.tensor(
        np.random.random(size=data_shape),
        dtype=torch.float,
    )
    bb_module = BasicResBlock(
        fake_data.shape[1],
        40,
        bn_kwargs={},
        stride=1,
    )
    output = bb_module(fake_data)
    assert output is not None


def test_BottleNeck():
    data_shape = np.random.randint(10, 20, size=4)
    fake_data = torch.tensor(
        np.random.random(size=data_shape),
        dtype=torch.float,
    )
    bn_module = BottleNeck(
        fake_data.shape[1],
        40,
        bn_kwargs={},
        stride=1,
    )
    output = bn_module(fake_data)
    assert output is not None
