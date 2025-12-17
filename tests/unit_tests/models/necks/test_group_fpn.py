from typing import List

import pytest
import torch

from hat.models.necks.group_fpn import GroupFPN


@pytest.mark.parametrize(
    ["feature_dim", "in_channels", "min_output_stride", "group_base"],
    [
        pytest.param(128, [32, 64, 128, 256], 4, 2),
        pytest.param(120, [40, 80, 160, 240], 8, 4),
        pytest.param(120, [40, 80, 160, 320], 16, 8),
        pytest.param(256, [16, 32, 64, 128], 32, 16),
    ],
)
def test_group_fpn(
    feature_dim,
    in_channels,
    min_output_stride,
    group_base,
):
    inputs = [
        torch.randn((2, in_channels[0], 32, 32)),
        torch.randn((2, in_channels[1], 16, 16)),
        torch.randn((2, in_channels[2], 8, 8)),
        torch.randn((2, in_channels[3], 4, 4)),
    ]
    model = GroupFPN(
        feature_dim=feature_dim,
        in_channels=in_channels,
        min_output_stride=min_output_stride,
        group_base=group_base,
    )
    outputs = model(inputs)
    assert len(outputs) == 4
    assert isinstance(outputs, List)
