import pytest
import torch

from hat.models.base_modules.basic_vargdarknet_module import VargDarkNetBlock


def test_varg_darknet_block():
    h, w = 64, 64
    in_chan = 64
    out_chan = 32
    module = VargDarkNetBlock(
        in_channels=in_chan,
        out_channels=out_chan,
        bn_kwargs={},
    )
    inputs = torch.rand(4, in_chan, h, w)
    outputs = module(inputs)
    assert outputs.shape[1] == out_chan * 2
    assert outputs.shape[-1] == w
    module.fuse_model()
    assert isinstance(module, torch.nn.Module)
    with pytest.raises(AssertionError):
        in_chan = 32
        out_chan = 32
        module = VargDarkNetBlock(
            in_channels=in_chan,
            out_channels=out_chan,
            bn_kwargs={},
        )
        outputs = module(inputs)
