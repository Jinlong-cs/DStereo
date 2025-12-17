import torch

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test, qtensor_test


def test_pafpn():
    feat_channels = 32
    config = dict(
        type="RPAFPN",
        in_channels=[32, 24, 40, 72, 160],
        out_channels=feat_channels,
        start_level=1,
        add_extra_convs="on_output",  # use P5
        num_outs=5,
        relu_before_extra_convs=True,
    )
    rpafpn = build_from_registry(config)

    input_size = 256
    x = [
        torch.randn((1, 32, input_size // 2, input_size // 2)),
        torch.randn((1, 24, input_size // 4, input_size // 4)),
        torch.randn((1, 40, input_size // 8, input_size // 8)),
        torch.randn((1, 72, input_size // 16, input_size // 16)),
        torch.randn((1, 160, input_size // 32, input_size // 32)),
    ]
    y = rpafpn(x)
    assert len(y) == 5
    assert y[0].shape == (1, feat_channels, 64, 64)
    assert y[1].shape == (1, feat_channels, 32, 32)
    assert y[2].shape == (1, feat_channels, 16, 16)
    assert y[3].shape == (1, feat_channels, 8, 8)
    assert y[4].shape == (1, feat_channels, 4, 4)

    x = qtensor_test(x)
    qat_test(rpafpn, x, with_quantized=False)
