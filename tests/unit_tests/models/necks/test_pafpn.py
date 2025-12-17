import pytest
import torch

from hat.models.necks.pafpn import PAFPN, VargPAFPN
from tests.unit_tests.models.base import qat_test, qtensor_test


@pytest.mark.parametrize(
    ["norm_cfg"],
    [
        pytest.param(None),
        pytest.param({"norm_type": "gn", "num_groups": 32, "affine": True}),
        pytest.param({"norm_type": "bn", "affine": True}),
    ],
)
def test_pafpn(norm_cfg):
    feat_channels = {4: 32, 8: 64, 16: 96, 32: 160, 64: 320}
    out_strides = [4, 8, 16, 32, 64]
    pafpn = PAFPN(
        in_channels=[32, 24, 40, 72, 160],
        out_channels=feat_channels,
        out_strides=out_strides,
        start_level=1,
        add_extra_convs="on_output",  # use P5
        num_outs=5,
        relu_before_extra_convs=True,
        norm_cfg=norm_cfg,
    )

    input_size = 256
    x = [
        torch.randn((1, 32, input_size // 2, input_size // 2)),
        torch.randn((1, 24, input_size // 4, input_size // 4)),
        torch.randn((1, 40, input_size // 8, input_size // 8)),
        torch.randn((1, 72, input_size // 16, input_size // 16)),
        torch.randn((1, 160, input_size // 32, input_size // 32)),
    ]
    y = pafpn(x)
    assert len(y) == 5
    assert y[0].shape == (1, feat_channels[out_strides[0]], 64, 64)
    assert y[1].shape == (1, feat_channels[out_strides[1]], 32, 32)
    assert y[2].shape == (1, feat_channels[out_strides[2]], 16, 16)
    assert y[3].shape == (1, feat_channels[out_strides[3]], 8, 8)
    assert y[4].shape == (1, feat_channels[out_strides[4]], 4, 4)

    if norm_cfg and norm_cfg.get("norm_type") == "gn":
        with pytest.raises(ValueError):
            x = qtensor_test(x)
            qat_test(pafpn, x, with_quantized=False)
    else:
        x = qtensor_test(x)
        qat_test(pafpn, x, with_quantized=False)


@pytest.mark.parametrize(
    [
        "varg_block_type",
        "group_base",
    ],
    [
        pytest.param("BasicVarGBlock", 8),
        pytest.param("BasicMixVarGEBlock", 16),
    ],
)
def test_varg_pafpn(varg_block_type, group_base):
    out_strides = [8, 16, 32]
    out_channels = 64
    varg_pafpn = VargPAFPN(
        in_channels=[32, 64, 128],
        out_channels=out_channels,
        out_strides=out_strides,
        start_level=0,
        num_outs=len(out_strides),
        varg_block_type=varg_block_type,
        group_base=group_base,
        bn_kwargs={},
    )

    input_size = 256
    x = [
        torch.randn((1, 32, input_size // 8, input_size // 8)),
        torch.randn((1, 64, input_size // 16, input_size // 16)),
        torch.randn((1, 128, input_size // 32, input_size // 32)),
    ]
    y = varg_pafpn(x)
    assert len(y) == len(out_strides)
    assert y[0].shape == (1, out_channels, 32, 32)
    assert y[1].shape == (1, out_channels, 16, 16)
    assert y[2].shape == (1, out_channels, 8, 8)

    x = qtensor_test(x)
    qat_test(varg_pafpn, x, with_quantized=False)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
