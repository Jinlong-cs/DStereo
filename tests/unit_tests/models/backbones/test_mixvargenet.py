from math import ceil, pow

import pytest
import torch

from hat.models.backbones.mixvargenet import (
    IdentityConfig,
    MixVarGENetConfig,
    get_mixvargenet_stride2channels,
)
from hat.registry import build_from_registry
from tests.unit_tests.models.base import profile_test, qat_test

net_config_example = [
    [
        MixVarGENetConfig(
            in_channels=32,
            out_channels=32,
            head_op="mixvarge_f2",
            stack_ops=[],
            stack_factor=1,
            stride=1,
        ),  # noqa
    ],  # stride 2
    [
        MixVarGENetConfig(
            in_channels=32,
            out_channels=32,
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
        ),  # noqa
    ],  # stride 4
    [
        MixVarGENetConfig(
            in_channels=32,
            out_channels=64,
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
        ),  # noqa
    ],  # stride 8
    [
        MixVarGENetConfig(
            in_channels=64,
            out_channels=96,
            head_op="mixvarge_f2_gb16",
            stack_ops=[
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
            ],
            stack_factor=1,
            stride=2,
        ),  # noqa
    ],  # stride 16
    [
        MixVarGENetConfig(
            in_channels=96,
            out_channels=160,
            head_op="mixvarge_f2_gb16",
            stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
            stack_factor=1,
            stride=2,
        ),  # noqa
    ],  # stride 32
]

net_config_tiny_example = [
    [
        IdentityConfig(
            in_channels=32,
            out_channels=32,
            stride=1,
        ),  # Identity
    ],  # stride 2
    [
        MixVarGENetConfig(
            in_channels=32,
            out_channels=32,
            head_op="mixvarge_f2",
            stack_ops=["mixvarge_f2", "mixvarge_f2"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 4
    [
        MixVarGENetConfig(
            in_channels=32,
            out_channels=64,
            head_op="mixvarge_f2",
            stack_ops=["mixvarge_f2", "mixvarge_f2"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 8
    [
        MixVarGENetConfig(
            in_channels=64,
            out_channels=96,
            head_op="mixvarge_f2_gb16",
            stack_ops=[
                "mixvarge_f1_gb16",
                "mixvarge_f1_gb16",
                "mixvarge_f1_gb16",
                "mixvarge_f1_gb16",
                "mixvarge_f1_gb16",
                "mixvarge_f1_gb16",
            ],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 16
    [
        MixVarGENetConfig(
            in_channels=96,
            out_channels=160,
            head_op="mixvarge_f2_gb16",
            stack_ops=["mixvarge_f1_gb16", "mixvarge_f1_gb16"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 32
]

net_config_2x4xfusion_example = [
    [
        MixVarGENetConfig(
            in_channels=32,
            out_channels=32,
            head_op="mixvarge_f2",
            stack_ops=[],
            stack_factor=1,
            stride=1,
            fusion_strides=[],
            extra_downsample_num=2,
        ),  # noqa
    ],  # stride 2
    [
        MixVarGENetConfig(
            in_channels=32,
            out_channels=32,
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[2],
            extra_downsample_num=2,
        ),  # noqa
    ],  # stride 4
    [
        MixVarGENetConfig(
            in_channels=32,
            out_channels=64,
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[2, 4],
            extra_downsample_num=2,
        ),  # noqa
    ],  # stride 8
    [
        MixVarGENetConfig(
            in_channels=64,
            out_channels=96,
            head_op="mixvarge_f2_gb16",
            stack_ops=[
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
            ],
            stack_factor=1,
            stride=2,
            fusion_strides=[4, 8],
            extra_downsample_num=2,
        ),  # noqa
    ],  # stride 16
    [
        MixVarGENetConfig(
            in_channels=96,
            out_channels=160,
            head_op="mixvarge_f2_gb16",
            stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
            stack_factor=1,
            stride=2,
            fusion_strides=[8, 16],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 32
]

net_config_2x4xfusion_example_stackfactor2 = [
    [
        MixVarGENetConfig(
            in_channels=32,
            out_channels=32,
            head_op="mixvarge_f2",
            stack_ops=[],
            stack_factor=2,
            stride=1,
            fusion_strides=[],
            extra_downsample_num=2,
        ),  # noqa
    ],  # stride 2
    [
        MixVarGENetConfig(
            in_channels=32,
            out_channels=32,
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=2,
            stride=2,
            fusion_strides=[2],
            extra_downsample_num=2,
        ),  # noqa
    ],  # stride 4
    [
        MixVarGENetConfig(
            in_channels=32,
            out_channels=64,
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=2,
            stride=2,
            fusion_strides=[2, 4],
            extra_downsample_num=2,
        ),  # noqa
    ],  # stride 8
    [
        MixVarGENetConfig(
            in_channels=64,
            out_channels=96,
            head_op="mixvarge_f2_gb16",
            stack_ops=[
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
            ],
            stack_factor=2,
            stride=2,
            fusion_strides=[4, 8],
            extra_downsample_num=2,
        ),  # noqa
    ],  # stride 16
    [
        MixVarGENetConfig(
            in_channels=96,
            out_channels=160,
            head_op="mixvarge_f2_gb16",
            stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
            stack_factor=2,
            stride=2,
            fusion_strides=[8, 16],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 32
]

net_config_2x4x8xfusion_example = [
    [
        MixVarGENetConfig(
            in_channels=32,
            out_channels=32,
            head_op="mixvarge_f2",
            stack_ops=[],
            stack_factor=1,
            stride=1,
            fusion_strides=[],
            extra_downsample_num=3,
        ),  # noqa
    ],  # stride 2
    [
        MixVarGENetConfig(
            in_channels=32,
            out_channels=32,
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[2],
            extra_downsample_num=3,
        ),  # noqa
    ],  # stride 4
    [
        MixVarGENetConfig(
            in_channels=32,
            out_channels=64,
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[2, 4],
            extra_downsample_num=3,
        ),  # noqa
    ],  # stride 8
    [
        MixVarGENetConfig(
            in_channels=64,
            out_channels=96,
            head_op="mixvarge_f2_gb16",
            stack_ops=[
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
            ],
            stack_factor=1,
            stride=2,
            fusion_strides=[2, 4, 8],
            extra_downsample_num=2,
        ),  # noqa
    ],  # stride 16
    [
        MixVarGENetConfig(
            in_channels=96,
            out_channels=160,
            head_op="mixvarge_f2_gb16",
            stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
            stack_factor=1,
            stride=2,
            fusion_strides=[4, 8, 16],
            extra_downsample_num=1,
        ),  # noqa
    ],  # stride 32
    [
        MixVarGENetConfig(
            in_channels=160,
            out_channels=320,
            head_op="mixvarge_f2_gb16",
            stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
            stack_factor=1,
            stride=2,
            fusion_strides=[8, 16, 32],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 64
]

rep_mixvargenet_config = [
    [
        MixVarGENetConfig(  # stride 2
            in_channels=32,
            out_channels=32,
            head_op="repmixvarge_f2",
            stack_ops=[],
            stack_factor=1,
            stride=1,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],
    [
        MixVarGENetConfig(  # stride 4
            in_channels=32,
            out_channels=32,
            head_op="repmixvarge_k3k3_f4",
            stack_ops=["repmixvarge_k3k3_f4", "repmixvarge_k3k3_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],
    [
        MixVarGENetConfig(  # stride 8
            in_channels=32,
            out_channels=64,
            head_op="repmixvarge_k3k3_f4",
            stack_ops=["repmixvarge_k3k3_f4", "repmixvarge_k3k3_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],
    [
        MixVarGENetConfig(  # stride 16
            in_channels=64,
            out_channels=96,
            head_op="repmixvarge_f2_gb16",
            stack_ops=[
                "repmixvarge_f2_gb16",
                "repmixvarge_f2_gb16",
                "repmixvarge_f2_gb16",
                "repmixvarge_f2_gb16",
                "repmixvarge_f2_gb16",
                "repmixvarge_f2_gb16",
            ],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],
    [
        MixVarGENetConfig(  # stride 32
            in_channels=96,
            out_channels=160,
            head_op="repmixvarge_f2_gb16",
            stack_ops=["repmixvarge_f2_gb16", "repmixvarge_f2_gb16"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],
]


@pytest.mark.parametrize(
    ["input_resize_scale"],
    [
        pytest.param(1),
        pytest.param(2),
        pytest.param(0.5),
    ],
)
def test_mixvargenet_scale_value(input_resize_scale):
    num_classes = 10
    bn_kwargs = dict(eps=1e-5, momentum=0.1)
    include_top = False
    cfg = dict(
        type="MixVarGENet",
        net_config=net_config_example,
        num_classes=num_classes,
        bn_kwargs=bn_kwargs,
        output_list=[],
        include_top=include_top,
        flat_output=False,
        input_resize_scale=input_resize_scale,
    )
    model = build_from_registry(cfg)

    input_size = 480
    x = torch.randn((1, 3, input_size, input_size))

    # x_seq = [x for _ in range(input_sequence_length)]
    y = model(x)
    for stride_idx, _ in enumerate(net_config_example):
        stride = pow(2, stride_idx + 1)
        assert y[stride_idx].shape[-1] == ceil(
            input_size * input_resize_scale / stride
        )


@pytest.mark.parametrize(
    ["net_config", "flat_output", "include_top"],
    [
        pytest.param(net_config_example, False, False),
        pytest.param(net_config_example, True, True),
        pytest.param(net_config_tiny_example, False, False),
        pytest.param(net_config_tiny_example, True, True),
        pytest.param(net_config_2x4xfusion_example, False, False),
        pytest.param(net_config_2x4xfusion_example, True, True),
        pytest.param(net_config_2x4xfusion_example_stackfactor2, False, False),
        pytest.param(net_config_2x4xfusion_example_stackfactor2, True, True),
        pytest.param(net_config_2x4x8xfusion_example, False, False),
        pytest.param(net_config_2x4x8xfusion_example, True, True),
        pytest.param(rep_mixvargenet_config, False, False),
        pytest.param(rep_mixvargenet_config, True, True),
    ],
)
def test_mixvargenet(net_config, flat_output, include_top):
    num_classes = 10
    bn_kwargs = dict(eps=1e-5, momentum=0.1)
    cfg = dict(
        type="MixVarGENet",
        net_config=net_config,
        num_classes=num_classes,
        bn_kwargs=bn_kwargs,
        include_top=include_top,
        flat_output=flat_output,
        output_list=[],
        input_resize_scale=1.0,
    )
    model = build_from_registry(cfg)
    input_size = 480
    x = torch.randn((1, 3, input_size, input_size))
    y = model(x)

    if not include_top:
        assert isinstance(y, list) and len(y) == len(net_config)

        for stride_idx, _ in enumerate(net_config):
            stride = pow(2, stride_idx + 1)
            assert y[stride_idx].shape[-1] == ceil(input_size / stride)
    else:
        assert isinstance(y, torch.Tensor)
        assert y.shape[1] == num_classes

    qat_test(model, x)
    # Dont run profile_test before qat_test, strange bug will occur.
    profile_test(model, x)


@pytest.mark.parametrize(
    ["input_sequence_length"],
    [
        pytest.param(1),
        pytest.param(4),
    ],
)
def test_mixvargenet_sequence_input(input_sequence_length):
    num_classes = 10
    bn_kwargs = dict(eps=1e-5, momentum=0.1)
    include_top = True
    cfg = dict(
        type="MixVarGENet",
        net_config=net_config_example,
        num_classes=num_classes,
        bn_kwargs=bn_kwargs,
        output_list=[],
        input_sequence_length=input_sequence_length,
    )
    model = build_from_registry(cfg)

    input_size = 480
    x = torch.randn((1, 3, input_size, input_size))
    x_seq = [x for _ in range(input_sequence_length)]
    y = model(x_seq)
    print(y.shape)

    if not include_top:
        assert isinstance(y, list) and len(y) == len(net_config_example)

        for stride_idx, _ in enumerate(net_config_example):
            stride = pow(2, stride_idx + 1)
            assert y[stride_idx].shape[-1] == ceil(input_size / stride)
    else:
        assert isinstance(y, torch.Tensor)
        assert y.shape[1] == num_classes

    qat_test(model, x_seq)
    # Dont run profile_test before qat_test, strange bug will occur.
    profile_test(model, x_seq)


def test_get_mixvargenet_stride2channels():
    stride2channels = get_mixvargenet_stride2channels(net_config_example)
    assert isinstance(stride2channels, dict)
