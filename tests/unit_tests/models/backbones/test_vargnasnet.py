import pytest
import torch

from hat.models.backbones.vargnasnet import (
    VargNASBlockConfig,
    get_vargnasnet_stride2channels,
)
from hat.registry import build_from_registry
from tests.unit_tests.models.base import profile_test, qat_test

net_config_example = [
    VargNASBlockConfig(
        in_channels=32,
        out_channels=32,
        head_op="varg_k3f1",
        stack_ops=[],
        stack_ops_num=0,
        stride=1,
    ),  # noqa
    VargNASBlockConfig(
        in_channels=32,
        out_channels=24,
        head_op="vargr_k3",
        stack_ops=["varg_k3"],
        stack_ops_num=1,
        stride=2,
    ),  # noqa
    VargNASBlockConfig(
        in_channels=24,
        out_channels=40,
        head_op="vargr_k5",
        stack_ops=[],
        stack_ops_num=0,
        stride=2,
    ),  # noqa
    VargNASBlockConfig(
        in_channels=40,
        out_channels=56,
        head_op="vargr_k3",
        stack_ops=["varg_k5", "varg_k3"],
        stack_ops_num=2,
        stride=2,
    ),  # noqa
    VargNASBlockConfig(
        in_channels=56,
        out_channels=72,
        head_op="vargr_k3",
        stack_ops=["varg_k3", "varg_k3"],
        stack_ops_num=2,
        stride=1,
    ),  # noqa
    VargNASBlockConfig(
        in_channels=72,
        out_channels=96,
        head_op="vargr_k3",
        stack_ops=["varg_k3"],
        stack_ops_num=1,
        stride=2,
    ),  # noqa
    VargNASBlockConfig(
        in_channels=96,
        out_channels=160,
        head_op="vargr_k5",
        stack_ops=[],
        stack_ops_num=0,
        stride=1,
    ),  # noqa
]


@pytest.mark.parametrize(
    ["net_config", "flat_output", "include_top"],
    [
        pytest.param(net_config_example, False, False),
        pytest.param(net_config_example, False, True),
        pytest.param(net_config_example, True, True),
    ],
)
def test_varg_nasnet(net_config, flat_output, include_top):
    num_classes = 10
    bn_kwargs = dict(eps=1e-5, momentum=0.1)
    cfg = dict(
        type="VargNASNet",
        net_config=net_config,
        num_classes=num_classes,
        bn_kwargs=bn_kwargs,
        include_top=include_top,
        flat_output=flat_output,
    )
    model = build_from_registry(cfg)
    input_size = 480
    x = torch.randn((1, 3, input_size, input_size))
    y = model(x)

    if not include_top:
        assert isinstance(y, list) and len(y) == 5
        assert y[0].shape[-1] == input_size / 2
        assert y[1].shape[-1] == input_size / 4
        assert y[2].shape[-1] == input_size / 8
        assert y[3].shape[-1] == input_size / 16
        assert y[4].shape[-1] == input_size / 32
    else:
        assert isinstance(y, torch.Tensor)
        assert y.shape[1] == num_classes

    qat_test(model, x)
    # Dont run profile_test before qat_test, strange bug will occur.
    profile_test(model, x)


@pytest.mark.parametrize(
    ["input_sequence_length"],
    [
        pytest.param(4),
    ],
)
def test_nasnet_sequence_input(input_sequence_length):
    num_classes = 10
    bn_kwargs = dict(eps=1e-5, momentum=0.1)
    cfg = dict(
        type="VargNASNet",
        net_config=net_config_example,
        num_classes=num_classes,
        bn_kwargs=bn_kwargs,
        include_top=False,
        flat_output=False,
        bias=False,
        input_sequence_length=input_sequence_length,
    )
    model = build_from_registry(cfg)
    input_size = 480
    x = torch.randn((1, 3, input_size, input_size))
    x_seq = [x for _ in range(input_sequence_length)]
    y = model(x_seq)

    assert isinstance(y, list) and len(y) == 5
    qat_test(model, x_seq)
    # Dont run profile_test before qat_test, strange bug will occur.
    profile_test(model, x_seq)


def test_get_vargnasnet_stride2channels():
    stride2channels = get_vargnasnet_stride2channels(net_config_example)
    assert isinstance(stride2channels, dict)
