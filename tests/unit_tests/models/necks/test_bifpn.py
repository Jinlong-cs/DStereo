import torch

from hat.models.backbones.vargnetv2 import get_vargnetv2_stride2channels
from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test, qtensor_test


def test_same_channel_bifpn():
    alpha = 0.5
    feat_channels = 64
    config = dict(
        type="BiFPN",
        fpn_name="bifpn_sum",
        in_strides=[2, 4, 8, 16, 32],
        out_strides=[4, 8, 16, 32, 64],
        stride2channels=get_vargnetv2_stride2channels(alpha),
        out_channels=feat_channels,
        stack=2,
        start_level=1,
        end_level=-1,
        num_outs=5,
    )
    fpn = build_from_registry(config)

    input_size = 256
    x = [
        torch.randn((1, 16, input_size // 2, input_size // 2)),
        torch.randn((1, 16, input_size // 4, input_size // 4)),
        torch.randn((1, 32, input_size // 8, input_size // 8)),
        torch.randn((1, 64, input_size // 16, input_size // 16)),
        torch.randn((1, 128, input_size // 32, input_size // 32)),
    ]
    y = fpn(x)
    assert len(y) == 5
    assert y[0].shape == (1, feat_channels, 64, 64)
    assert y[1].shape == (1, feat_channels, 32, 32)
    assert y[2].shape == (1, feat_channels, 16, 16)
    assert y[3].shape == (1, feat_channels, 8, 8)
    assert y[4].shape == (1, feat_channels, 4, 4)

    x = qtensor_test(x)
    qat_test(fpn, x, with_quantized=False)


def test_diff_channel_bifpn():
    alpha = 0.5
    feat_channels = {
        4: 32,
        8: 48,
        16: 96,
        32: 192,
        64: 360,
    }
    config = dict(
        type="BiFPN",
        fpn_name="bifpn_sum",
        in_strides=[2, 4, 8, 16, 32],
        out_strides=[4, 8, 16, 32, 64],
        stride2channels=get_vargnetv2_stride2channels(alpha),
        out_channels=feat_channels,
        stack=2,
        start_level=1,
        end_level=-1,
        num_outs=5,
    )
    fpn = build_from_registry(config)

    input_size = 256
    x = [
        torch.randn((1, 16, input_size // 2, input_size // 2)),
        torch.randn((1, 16, input_size // 4, input_size // 4)),
        torch.randn((1, 32, input_size // 8, input_size // 8)),
        torch.randn((1, 64, input_size // 16, input_size // 16)),
        torch.randn((1, 128, input_size // 32, input_size // 32)),
    ]
    y = fpn(x)
    assert len(y) == 5
    assert y[0].shape == (1, feat_channels[4], 64, 64)
    assert y[1].shape == (1, feat_channels[8], 32, 32)
    assert y[2].shape == (1, feat_channels[16], 16, 16)
    assert y[3].shape == (1, feat_channels[32], 8, 8)
    assert y[4].shape == (1, feat_channels[64], 4, 4)

    x = qtensor_test(x)
    qat_test(fpn, x, with_quantized=False)
