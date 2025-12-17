import torch

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test, qtensor_test


def test_dw_unet():
    base_channels = 1
    input_size = 256

    config = dict(
        type="DwUnet",
        base_channels=base_channels,
        output_scales=(4, 8, 16, 32, 64),
    )

    dw_unet = build_from_registry(config)

    x = [
        torch.randn(
            (
                1,
                base_channels * 32 // (2 ** i),
                input_size // 32 * 2 ** i,
                input_size // 32 * 2 ** i,
            )
        )
        for i in range(4, -1, -1)
    ]

    y = dw_unet(x)

    assert len(y) == 5
    for i in range(5):
        torch.testing.assert_allclose(
            y[i].shape,
            (torch.as_tensor(x[i].shape) * torch.tensor((1, 2, 0.5, 0.5))).to(
                dtype=torch.long
            ),
        )

    x = qtensor_test(x)
    qat_test(dw_unet, x, with_quantized=False)
