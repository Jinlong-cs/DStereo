import torch

from hat.models.base_modules.basic_cspdarknet_module import (
    Bottleneck,
    CSPLayer,
    Focus,
    SPPBottleneck,
)


def test_Bottleneck():
    data = torch.randn((1, 3, 112, 112))
    base_Bottleneck = Bottleneck(3, 3)
    output = base_Bottleneck(data)
    assert output is not None


def test_CSPLayer():
    data = torch.randn((1, 3, 112, 112))
    base_CSPLayer = CSPLayer(3, 3)
    output = base_CSPLayer(data)
    assert output is not None


def test_Focus():
    data = torch.randn((1, 3, 112, 112))
    base_Focus = Focus(3, 3)
    output = base_Focus(data)
    assert output is not None


def test_SPPBottleneck():
    data = torch.randn((1, 3, 112, 112))
    base_SPPBottleneck = SPPBottleneck(3, 3)
    output = base_SPPBottleneck(data)
    assert output is not None
