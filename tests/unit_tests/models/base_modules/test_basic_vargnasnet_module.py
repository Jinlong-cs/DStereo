# Copyright (c) Horizon Robotics. All rights reserved.
import numpy as np
import pytest
import torch

from hat.models.base_modules.basic_vargnasnet_module import VargNASNetBlock


@pytest.mark.parametrize(
    ["config_i"],
    [
        pytest.param([[32, 32], "varg_k3f1", [], 0, 1]),
        pytest.param([[32, 24], "vargr_k3", ["varg_k3"], 1, 2]),
        pytest.param([[24, 40], "vargr_k5", [], 0, 2]),
        pytest.param([[40, 56], "vargr_k3", ["varg_k5", "varg_k3"], 2, 2]),
        pytest.param([[56, 72], "vargr_k3", ["varg_k3", "varg_k3"], 2, 1]),
        pytest.param([[72, 96], "vargr_k3", ["varg_k3"], 1, 2]),
        pytest.param([[96, 160], "vargr_k5", [], 0, 1]),
    ],
)
def test_varg_nasnet_block(config_i):
    data_shape = [1, config_i[0][0], 32, 32]
    fake_data = torch.tensor(
        np.random.random(size=data_shape),
        dtype=torch.float,
    )
    bvg_module = VargNASNetBlock(
        in_ch=config_i[0][0],
        block_ch=config_i[0][1],
        head_op=config_i[1],
        stack_ops=config_i[2],
        stride=config_i[-1],
        bias=True,
        bn_kwargs={},
    )
    output = bvg_module(fake_data)
    assert output is not None
