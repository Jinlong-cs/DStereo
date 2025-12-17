import pytest
import torch

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test


@pytest.mark.parametrize(
    ["model", "zero_init_residual", "replace_stride_with_dilation"],
    [
        pytest.param("resnet18", True, None),
        pytest.param("resnet18", False, [False, False, False]),
        pytest.param("resnet50", True, None),
    ],
)
def test_resnet(model, zero_init_residual, replace_stride_with_dilation):
    cfg = dict(
        type=model,
        pretrained_path=None,
        num_classes=10,
        zero_init_residual=zero_init_residual,
        replace_stride_with_dilation=replace_stride_with_dilation,
    )
    model = build_from_registry(cfg)
    x = torch.randn((1, 3, 224, 224))
    y = model(x)
    assert isinstance(y, torch.Tensor)
    assert y.shape[1] == 10

    qat_test(model, x)
