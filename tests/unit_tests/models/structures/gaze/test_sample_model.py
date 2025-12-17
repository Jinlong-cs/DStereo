import pytest
import torch

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test


@pytest.mark.parametrize(
    [
        "img_w",
        "img_h",
        "grid_w",
        "grid_h",
    ],
    [
        pytest.param(
            512,
            330,
            320,
            150,
        ),
    ],
)
def test_sample_model_task(img_w, img_h, grid_w, grid_h):
    config = dict(
        type="SampleModel",
        output_size=(grid_w, grid_h),
        deploy=True,
    )
    sample_model = build_from_registry(config)
    x = {
        "img": torch.rand(4, 1, img_h, img_w),
        "grid": torch.rand(4, 2, grid_h, grid_w),
    }
    batch_out = sample_model(x)
    assert batch_out.shape == (4, 1, grid_h, grid_w)
    qat_test(sample_model, x, with_quantized=False)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
