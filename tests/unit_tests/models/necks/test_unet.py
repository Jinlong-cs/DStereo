import torch

from hat.models.backbones.vargnetv2 import get_vargnetv2_stride2channels
from hat.models.necks.unet import Unet
from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test, qtensor_test


def test_unet():
    alpha = 0.5
    config = dict(
        type="Unet",
        in_strides=[2, 4, 8, 16, 32],
        out_strides=[4, 8, 16, 32],
        stride2channels=get_vargnetv2_stride2channels(alpha),
        factor=2,
        use_bias=False,
        group_base=8,
    )
    unet = build_from_registry(config)
    input_size = 256
    x = [
        torch.randn((1, 16, input_size // 2, input_size // 2)),
        torch.randn((1, 16, input_size // 4, input_size // 4)),
        torch.randn((1, 32, input_size // 8, input_size // 8)),
        torch.randn((1, 64, input_size // 16, input_size // 16)),
        torch.randn((1, 128, input_size // 32, input_size // 32)),
    ]
    y = unet(x)
    assert len(y) == 4
    assert y[0].shape == (1, 16, 64, 64)
    assert y[1].shape == (1, 32, 32, 32)
    assert y[2].shape == (1, 64, 16, 16)
    assert y[3].shape == (1, 128, 8, 8)

    x = qtensor_test(x)
    qat_test(unet, x, with_quantized=False)


def test_unetv2():
    alpha_neck = 1.5
    alpha_backbone = 2.0
    unet = Unet(
        in_strides=[2, 4, 8, 16, 32],
        out_strides=[4, 8, 16, 32],
        out_stride2channels=get_vargnetv2_stride2channels(alpha_neck),
        stride2channels=get_vargnetv2_stride2channels(alpha_backbone),
        factor=2,
        use_bias=False,
        group_base=8,
    )
    input_size = 256
    x = [
        torch.randn((1, 64, input_size // 2, input_size // 2)),
        torch.randn((1, 64, input_size // 4, input_size // 4)),
        torch.randn((1, 128, input_size // 8, input_size // 8)),
        torch.randn((1, 256, input_size // 16, input_size // 16)),
        torch.randn((1, 512, input_size // 32, input_size // 32)),
    ]
    y = unet(x)
    assert len(y) == 4
    assert y[0].shape == (1, 48, 64, 64)
    assert y[1].shape == (1, 96, 32, 32)
    assert y[2].shape == (1, 192, 16, 16)
    assert y[3].shape == (1, 384, 8, 8)

    x = qtensor_test(x)
    qat_test(unet, x, with_quantized=False)
