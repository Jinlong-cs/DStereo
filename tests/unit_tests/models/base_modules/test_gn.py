import torch

from hat.models.base_modules.gn_module import GroupNorm2d


def test_gn():
    inputs = torch.randn(16, 3, 256, 256)
    channel = inputs.shape[1]
    group = 3
    assert channel // group
    gn = GroupNorm2d(group, channel)
    outputs = gn(inputs)
    assert outputs.shape == inputs.shape
