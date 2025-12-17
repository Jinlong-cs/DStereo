import pytest
import torch

from hat.models.backbones.vargnetv2_2631 import VargNetV2Stage2631
from tests.unit_tests.models.base import profile_test, qat_test


@pytest.mark.parametrize(
    [
        "factor",
        "group_base",
        "flat_output",
        "include_top",
    ],
    [
        pytest.param(1, 8, False, False),
        pytest.param(1, 8, False, True),
        pytest.param(1, 8, True, True),
    ],
)
def test_vargnet(factor, group_base, flat_output, include_top):
    num_classes = 10
    model = VargNetV2Stage2631(
        bn_kwargs={},
        multiplier=factor,
        group_base=group_base,
        num_classes=num_classes,
        flat_output=flat_output,
        include_top=include_top,
    )
    input_size = 480
    x = torch.randn((1, 3, input_size, input_size))
    y = model(x)

    if not include_top:
        assert isinstance(y, list) and len(y) == 5
        assert y[0].shape[-1] == input_size // 2
        assert y[1].shape[-1] == input_size // 4
        assert y[2].shape[-1] == input_size // 8
        assert y[3].shape[-1] == input_size // 16
        assert y[4].shape[-1] == input_size // 32
    else:
        assert isinstance(y, torch.Tensor)
        assert y.shape[1] == num_classes

    qat_test(model, x)
    # Dont run profile_test before qat_test, strange bug will occur.
    profile_test(model, x)
