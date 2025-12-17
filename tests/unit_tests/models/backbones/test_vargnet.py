import os

import pytest
import torch

from hat.models.backbones.vargnetv2 import get_vargnetv2_stride2channels
from hat.registry import build_from_registry
from tests.unit_tests.models.base import profile_test, qat_test


@pytest.mark.parametrize(
    [
        "factor",
        "alpha",
        "group_base",
        "flat_output",
        "include_top",
        "head_factor",
    ],
    [
        pytest.param(1, 0.25, 8, False, False, 1),
        pytest.param(1, 0.25, 8, False, True, 1),
    ],
)
def test_vargnet(
    factor, alpha, group_base, flat_output, include_top, head_factor
):
    num_classes = 10
    cfg = dict(
        type="VargNetV2",
        bn_kwargs={},
        num_classes=num_classes,
        factor=factor,
        alpha=alpha,
        group_base=group_base,
        flat_output=flat_output,
        include_top=include_top,
        head_factor=head_factor,
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
    [
        "factor",
        "alpha",
        "group_base",
        "flat_output",
        "include_top",
        "head_factor",
    ],
    [
        pytest.param(1, 0.5, 8, False, False, 1),
        pytest.param(1, 0.75, 8, False, True, 2),
        pytest.param(1, 1.0, 4, True, True, 1),
    ],
)
def test_tiny_vargnetv2(
    factor, alpha, group_base, flat_output, include_top, head_factor
):
    num_classes = 10
    cfg = dict(
        type="TinyVargNetV2",
        bn_kwargs={},
        num_classes=num_classes,
        factor=factor,
        alpha=alpha,
        group_base=group_base,
        flat_output=flat_output,
        include_top=include_top,
        head_factor=head_factor,
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
def test_vargnet_sequence_input(input_sequence_length):
    num_classes = 10
    cfg = dict(
        type="VargNetV2",
        bn_kwargs={},
        num_classes=num_classes,
        flat_output=False,
        include_top=False,
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


def test_vargnet_sequence_input_pretrain(tmpdir):
    cfg = dict(
        type="VargNetV2",
        bn_kwargs={},
        num_classes=10,
        flat_output=False,
        include_top=False,
    )
    model = build_from_registry(cfg)
    state = {"state_dict": model.state_dict()}
    ckpt_file = os.path.join(tmpdir, "tmp.pth")
    torch.save(state, ckpt_file)

    cfg["input_sequence_length"] = 4
    model = build_from_registry(cfg)

    input_size = 480
    x = torch.randn((1, 3, input_size, input_size))
    x_seq = [x for _ in range(4)]
    y = model(x_seq)
    assert isinstance(y, list) and len(y) == 5


def test_get_vargnetv2_stride2channels():
    stride2channels = get_vargnetv2_stride2channels(0.5)
    assert isinstance(stride2channels, dict)
