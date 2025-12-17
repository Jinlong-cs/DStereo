import torch

from hat.models.base_modules.quant_module import QuantModule


def test_quant_module():

    quant_module = QuantModule(scale=1.0)
    input_tensor = torch.randn((2, 1, 512, 960))

    output_qtensor = quant_module(input_tensor)

    assert output_qtensor.dtype == torch.float32
