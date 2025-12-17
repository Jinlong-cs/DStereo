# Copyright (c) Horizon Robotics. All rights reserved.

import pytest
import torch

from hat.models.necks.ufpn import UFPN
from tests.unit_tests.models.base import (
    qat_test_with_multi_inputs,
    qtensor_test,
)


def generate_fake_inputs(input_size, in_strides, in_channels, output_strides):
    inputs = [
        torch.randn(1, in_ch, input_size[1] // in_st, input_size[0] // in_st)
        for (in_ch, in_st) in zip(in_channels, in_strides)
    ]

    out_stride = output_strides[0]
    pe_feat_shape = (
        1,
        3,
        input_size[1] // out_stride,
        input_size[0] // out_stride,
    )
    coordinate_map = torch.randn(pe_feat_shape, dtype=torch.float32)

    return inputs, coordinate_map


def test_fpn_error():
    # Test len(in_strides) == len(in_channels) == len(out_channels)
    model = None
    with pytest.raises(AssertionError):
        model = UFPN(
            in_strides=[8, 16, 32],
            in_channels=[64, 128, 256],
            out_channels=[32, 64, 128, 256],
            group_base=4,
            bn_kwargs={},
        )
    assert model is None


@pytest.mark.parametrize(
    [
        "in_strides",
        "in_channels",
        "out_channels",
        "output_strides",
        "group_base",
        "is_with_relu",
        "pe_stride",
    ],
    [
        pytest.param(
            [8, 16, 32, 64],
            [16, 32, 64, 128],
            [16, 32, 64, 128],
            [8],
            8,
            True,
            None,
        ),
        pytest.param(
            [8, 16, 32, 64],
            [16, 32, 64, 128],
            [16, 32, 64, 128],
            [8],
            8,
            False,
            8,
        ),
        pytest.param(
            [4, 8, 16, 32, 64],
            [16, 32, 64, 128, 128],
            [16, 32, 64, 128, 128],
            [4],
            4,
            False,
            4,
        ),
    ],
)
def test_UFPNPE(
    in_strides,
    in_channels,
    out_channels,
    output_strides,
    group_base,
    is_with_relu,
    pe_stride,
):
    ufpn = UFPN(
        in_strides=in_strides,
        in_channels=in_channels,
        out_channels=out_channels,
        output_strides=output_strides,
        group_base=group_base,
        is_with_relu=is_with_relu,
        bn_kwargs=dict(eps=1e-5, momentum=0.1),
        pe_stride=pe_stride,
    )

    features, coordinate_map = generate_fake_inputs(
        input_size=(1920 // 2, 1280 // 2),
        in_strides=in_strides,
        in_channels=in_channels,
        output_strides=output_strides,
    )

    inputs = (features, coordinate_map)
    outputs = ufpn(*inputs)

    out_channels = [out_channels[in_strides.index(i)] for i in output_strides]
    assert isinstance(outputs, list) and len(outputs) == len(output_strides)
    for (i, output) in enumerate(outputs):
        assert isinstance(output, torch.Tensor)
        assert len(output.shape) == 4
        assert output.shape[1] == out_channels[i]

    features = qtensor_test(features)
    coordinate_map = qtensor_test(coordinate_map)
    qat_test_with_multi_inputs(
        ufpn, (features, coordinate_map), with_quantized=True
    )
