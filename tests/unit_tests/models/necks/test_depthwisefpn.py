from typing import List

import pytest
import torch

from hat.models.necks.depthwise_fpn import DepthWiseFPN


@pytest.mark.parametrize(
    ["feature_dim", "in_channels", "min_output_stride"],
    [
        pytest.param(128, [40, 80, 160, 320], 4),
        pytest.param(128, [40, 80, 160, 320], 8),
        pytest.param(128, [40, 80, 160, 320], 16),
        pytest.param(128, [40, 80, 160, 320], 32),
    ],
)
def test_depthwise_fpn(feature_dim, in_channels, min_output_stride):
    inputs = [
        torch.randn((2, 40, 32, 32)),
        torch.randn((2, 80, 16, 16)),
        torch.randn((2, 160, 8, 8)),
        torch.randn((2, 320, 4, 4)),
    ]
    model = DepthWiseFPN(
        feature_dim=feature_dim,
        in_channels=in_channels,
        min_output_stride=min_output_stride,
    )
    outputs = model(inputs)
    assert len(outputs) == 4
    assert isinstance(outputs, List)
