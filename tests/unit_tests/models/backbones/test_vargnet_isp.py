import pytest
import torch

from hat.registry import build_from_registry
from tests.unit_tests.models.base import profile_test, qat_test


@pytest.mark.parametrize(
    [
        "num_classes",
        "input_channels",
        "factor",
        "alpha",
        "bn_kwargs",
        "group_base",
        "flat_output",
        "include_top",
        "head_factor",
    ],
    [
        pytest.param(
            10,
            3,
            1,
            0.25,
            dict(eps=1e-5, momentum=0.1),
            8,
            False,
            False,
            1,
        ),
        pytest.param(10, 3, 1, 0.25, dict(), 8, False, True, 2),
        pytest.param(
            10,
            3,
            1,
            0.25,
            dict(eps=1e-5, momentum=0.1),
            8,
            True,
            True,
            1,
        ),
    ],
)
def test_vargnet_isp(
    num_classes,
    input_channels,
    factor,
    alpha,
    bn_kwargs,
    group_base,
    flat_output,
    include_top,
    head_factor,
):
    cfg = dict(
        type="VargNetV2ISP",
        num_classes=num_classes,
        input_channels=input_channels,
        factor=factor,
        alpha=alpha,
        bn_kwargs=bn_kwargs,
        group_base=group_base,
        flat_output=flat_output,
        include_top=include_top,
        head_factor=head_factor,
    )
    model = build_from_registry(cfg)
    input_size = 480
    x = torch.randn((1, input_channels, input_size, input_size))
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
