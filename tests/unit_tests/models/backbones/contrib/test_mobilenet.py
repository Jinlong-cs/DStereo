import pytest
import torch

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test


@pytest.mark.parametrize(
    ["width_mult", "round_nearest"],
    [
        pytest.param(1.0, 8),
    ],
)
def test_mobilenet_v2(width_mult, round_nearest):
    cfg = dict(
        type="mobilenet_v2",
        width_mult=width_mult,
        round_nearest=round_nearest,
        num_classes=10,
    )
    model = build_from_registry(cfg)
    x = torch.randn((1, 3, 224, 224))
    y = model(x)
    assert isinstance(y, torch.Tensor)
    assert y.shape[1] == 10

    qat_test(model, x)
