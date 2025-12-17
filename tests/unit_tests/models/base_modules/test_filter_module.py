import horizon_plugin_pytorch as horizon
import pytest
import torch

from hat.models.base_modules.postprocess import FilterModule
from tests.unit_tests.models.base import qtensor_test


@pytest.mark.parametrize(
    ["threshold", "idx_range"],
    [
        pytest.param(-4.59, None),
        pytest.param(-4.59, [2, 6]),
    ],
)
def test_filter_module(threshold, idx_range):
    model = FilterModule(
        threshold=threshold,
        idx_range=idx_range,
    )
    model.eval()
    x1 = [torch.randn(4, 10, 128, 240)]
    y = model(*x1)
    assert len(y) == 4  # batch_size
    assert len(y[0]) == 4
    assert y[0][2].shape[-1] == 2  # coords

    horizon.march.set_march(horizon.march.March.BAYES)
    qat_model = horizon.quantization.prepare_qat(model, inplace=False)
    x1 = qtensor_test(x1)
    y = qat_model(*x1)

    assert len(y) == 4  # batch_size
    assert len(y[0]) == 4
    assert y[0][2].shape[-1] == 2  # coords
