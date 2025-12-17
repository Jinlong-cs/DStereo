import torch

from hat.registry import build_from_registry


def test_fpn_quant():
    module = dict(
        type="FPNQuant",
        stride_num=4,
    )
    input_size = 512
    fpn_quant_module = build_from_registry(module)
    input = [
        torch.randn((1, 32, input_size // 8, input_size // 8)),
        torch.randn((1, 32, input_size // 16, input_size // 16)),
        torch.randn((1, 32, input_size // 32, input_size // 32)),
        torch.randn((1, 32, input_size // 64, input_size // 64)),
    ]

    output = fpn_quant_module(input)

    assert len(input) == len(output)
