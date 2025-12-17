import numpy as np
import pytest
import torch

from hat.models.base_modules.basic_vargnet_module import (
    BasicVarGBlock,
    BasicVarGBlockV2,
    ExtendVarGNetFeatures,
    OnePathResUnit,
    TwoPathResUnit,
)


@pytest.mark.parametrize(
    ["merge_branch", "group_base"],
    [
        pytest.param(True, 8),
        pytest.param(False, 4),
    ],
)
def test_basic_vargblock(merge_branch, group_base):
    data_shape = [40, 40, 40, 40]
    fake_data = torch.tensor(
        np.random.random(size=data_shape),
        dtype=torch.float,
    )
    bvg_module = BasicVarGBlock(
        fake_data.shape[1],
        40,
        40,
        stride=1,
        bn_kwargs={},
        merge_branch=merge_branch,
        group_base=group_base,
    )
    output = bvg_module(fake_data)
    assert output is not None


@pytest.mark.parametrize(
    ["in_channels", "out_channels", "stride", "merge_branch", "group_base"],
    [
        pytest.param(16, 32, 1, True, 8),
        pytest.param(32, 32, 1, True, 8),
        pytest.param(16, 32, 2, True, 8),
        pytest.param(16, 16, 2, True, 8),
        pytest.param(16, 32, 1, False, 8),
        pytest.param(16, 32, 1, False, 4),
    ],
)
def test_basic_vargblockv2(
    in_channels, out_channels, stride, merge_branch, group_base
):
    data_shape = [1, in_channels, 32, 32]
    fake_data = torch.tensor(
        np.random.random(size=data_shape),
        dtype=torch.float,
    )
    bvg_module = BasicVarGBlockV2(
        in_channels=in_channels,
        mid_channels=out_channels,
        out_channels=out_channels,
        stride=stride,
        bn_kwargs={},
        merge_branch=merge_branch,
        group_base=group_base,
    )
    output = bvg_module(fake_data)
    assert output is not None


def test_one_path_resunit():
    data_shape = [16, 16, 16, 16]
    fake_data = torch.tensor(
        np.random.random(size=data_shape),
        dtype=torch.float,
    )
    opr_module = OnePathResUnit(
        dw_num_filter=16,
        group_base=4,
        pw_num_filter=16,
        pw_num_filter2=16,
        bn_kwargs={},
        stride=1,
        is_dim_match=True,
    )
    output = opr_module(fake_data)
    assert output is not None


def test_two_path_resunit():
    data_shape = [1, 64, 64, 64]
    fake_data = torch.tensor(
        np.random.random(size=data_shape),
        dtype=torch.float,
    )
    opr_module = TwoPathResUnit(
        dw_num_filter=64,
        group_base=4,
        pw_num_filter=128,
        pw_num_filter2=128,
        bn_kwargs={},
        stride=2,
        is_dim_match=False,
        use_bias=True,
        pw_with_act=False,
        factor=1.0,
    )
    output = opr_module(fake_data)
    assert output is not None


def test_extend_vargnet_features():
    data_shape = [16, 16, 16, 16]
    fake_data = torch.tensor(
        np.random.random(size=data_shape),
        dtype=torch.float,
    )
    evf_module = ExtendVarGNetFeatures(
        prev_channel=16, channels=16, num_units=2, group_base=4, bn_kwargs={}
    )
    output = evf_module([fake_data])
    assert output is not None
