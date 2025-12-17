import torch

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test, qtensor_test


def test_fastscnn():
    config = dict(
        type="FastSCNNNeck",
        in_channels=[32, 128],
        feat_channels=[128, 128],
        indexes=[2, 4],
        bn_kwargs={},
    )

    fastscnn = build_from_registry(config)
    input_size = 256
    x = [
        torch.randn((2, 16, input_size // 2, input_size // 2)),
        torch.randn((2, 16, input_size // 4, input_size // 4)),
        torch.randn((2, 32, input_size // 8, input_size // 8)),
        torch.randn((2, 64, input_size // 16, input_size // 16)),
        torch.randn((2, 128, input_size // 32, input_size // 32)),
    ]
    y = fastscnn(x)
    assert len(y) == 3
    assert y[0].shape == (2, 32, 32, 32)
    assert y[1].shape == (2, 128, 8, 8)
    assert y[2].shape == (2, 128, 32, 32)

    x = qtensor_test(x)
    qat_test(fastscnn, x, with_quantized=False)
