import torch

from hat.models.backbones.csp_darknet import CSPDarknet


def test_CSPDarknet():
    backbone = CSPDarknet(dep_mul=1.0, wid_mul=1.0)
    data = torch.randn((1, 3, 224, 224))
    outputs = backbone(data)
    assert len(outputs) == 4
    assert outputs[0].shape == (1, 128, 56, 56)
    assert outputs[1].shape == (1, 256, 28, 28)
    assert outputs[2].shape == (1, 512, 14, 14)
    assert outputs[3].shape == (1, 1024, 7, 7)
