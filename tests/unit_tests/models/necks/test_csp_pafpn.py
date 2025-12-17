import torch

from hat.registry import build_from_registry


def test_csp_pafpn():
    neck = dict(
        type="CSPPAFPN",
        in_channels=[96, 192, 384, 768],
        num_csp_blocks=2,
        out_channels=192,
        num_outs=5,
        # node_name="neck",
    )

    input_size = 224
    csp_pafpn = build_from_registry(neck)
    x = [
        torch.randn((1, 96, input_size // 2, input_size // 2)),
        torch.randn((1, 192, input_size // 4, input_size // 4)),
        torch.randn((1, 384, input_size // 8, input_size // 8)),
        torch.randn((1, 768, input_size // 16, input_size // 16)),
    ]
    y = csp_pafpn(x)
    assert len(y) == 5
    assert y[0].shape == (1, 192, input_size // 2, input_size // 2)
    assert y[1].shape == (1, 192, input_size // 4, input_size // 4)
    assert y[2].shape == (1, 192, input_size // 8, input_size // 8)
    assert y[3].shape == (1, 192, input_size // 16, input_size // 16)
