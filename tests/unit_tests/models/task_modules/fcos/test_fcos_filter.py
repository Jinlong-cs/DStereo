import horizon_plugin_pytorch as horizon
import pytest
import torch

from hat.models.task_modules.fcos import (
    FCOSMultiStrideCatFilter,
    FCOSMultiStrideFilter,
)
from tests.unit_tests.models.base import qtensor_test


@pytest.mark.parametrize(
    ["strides", "idx_range", "threshold"],
    [
        pytest.param([4, 8], None, -4.59),
        pytest.param([4, 8], [2, 6], -4.59),
    ],
)
def test_fcos_multi_stride_filter(strides, idx_range, threshold):
    model = FCOSMultiStrideFilter(
        strides=strides, idx_range=idx_range, threshold=threshold
    )
    model.eval()
    batch_size = 4
    x1 = [
        [
            torch.randn((batch_size, 10, 128, 240)),
            torch.randn((batch_size, 10, 64, 120)),
        ],
        [
            torch.randn((batch_size, 4, 128, 240)),
            torch.randn((batch_size, 4, 64, 120)),
        ],
        [
            torch.randn((batch_size, 1, 128, 240)),
            torch.randn((batch_size, 1, 64, 120)),
        ],
    ]
    y = model(x1)
    assert len(y) == len(strides)
    assert len(y[0]) == batch_size
    assert len(y[0][0]) == 4  # coords, score, bbox, centerness
    assert y[0][0][0].shape[-1] == 2  # coords

    horizon.march.set_march(horizon.march.March.BAYES)
    qat_model = horizon.quantization.prepare_qat(model, inplace=False)
    x1 = qtensor_test(x1)
    y = qat_model(x1)
    assert len(y) == len(strides)
    assert len(y[0]) == batch_size
    assert len(y[0][0]) == 4  # coords, score, bbox, centerness
    assert y[0][0][0].shape[-1] == 2  # coords


@pytest.mark.parametrize(
    ["strides", "idx_range", "threshold", "task_strides"],
    [
        pytest.param(
            [4, 8],
            None,
            -4.59,
            [[4, 8], [4, 8]],
        ),
        pytest.param(
            [4, 8],
            [2, 6],
            -4.59,
            [[4, 8], [4, 8]],
        ),
    ],
)
def test_fcos_multi_stride_cat_filter(
    strides, idx_range, threshold, task_strides
):
    model = FCOSMultiStrideCatFilter(
        strides=strides,
        idx_range=idx_range,
        threshold=threshold,
        task_strides=task_strides,
    )
    model.eval()
    batch_size = 4
    x1 = [
        [
            [
                torch.randn((batch_size, 10, 128, 240)),
                torch.randn((batch_size, 10, 64, 120)),
            ],
            [
                torch.randn((batch_size, 4, 128, 240)),
                torch.randn((batch_size, 4, 64, 120)),
            ],
            [
                torch.randn((batch_size, 1, 128, 240)),
                torch.randn((batch_size, 1, 64, 120)),
            ],
        ],
        [
            [
                torch.randn((batch_size, 10, 128, 240)),
                torch.randn((batch_size, 10, 64, 120)),
            ],
            [
                torch.randn((batch_size, 4, 128, 240)),
                torch.randn((batch_size, 4, 64, 120)),
            ],
            [
                torch.randn((batch_size, 1, 128, 240)),
                torch.randn((batch_size, 1, 64, 120)),
            ],
        ],
    ]
    y = model(x1)
    assert len(y) == len(strides)
    assert len(y[0]) == batch_size
    assert len(y[0][0]) == 2 + 2 * len(
        task_strides
    )  # coords, score, (bbox, centerness) * task_num
    assert y[0][0][0].shape[-1] == 2  # coords

    horizon.march.set_march(horizon.march.March.BAYES)
    qat_model = horizon.quantization.prepare_qat(model, inplace=False)
    x1 = qtensor_test(x1)
    y = qat_model(x1)
    assert len(y) == len(strides)
    assert len(y[0]) == batch_size
    assert len(y[0][0]) == 2 + 2 * len(
        task_strides
    )  # coords, score, (bbox, centerness) * task_num
    assert y[0][0][0].shape[-1] == 2  # coords
