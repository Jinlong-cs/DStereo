import pytest
import torch
import torch.nn as nn
from horizon_plugin_pytorch import qat_mode
from pytest import param

from hat.models.base_modules.reparam import OnlineReParamBlock


@pytest.mark.parametrize(
    ["in_chn", "out_chn", "kernel", "stride", "groups", "act", "norm"],
    [
        param(16, 32, {3: 2, 1: 1}, 1, 1, None, None),
        param(16, 16, {3: 2, 1: 1}, 1, 1, None, None),
        param(32, 16, {3: 2, 1: 1}, 1, 1, None, None),
        param(32, 16, {5: 1, 3: 2, 1: 1}, 1, 8, None, None),
        param(32, 16, {3: 2, 1: 1}, 2, 1, None, None),
        param(32, 16, {3: 2, 1: 1}, 1, 8, None, None),
        param(32, 16, {3: 2, 1: 1}, 1, 8, nn.ReLU(True), None),
        param(32, 16, {3: 2, 1: 1}, 1, 8, None, nn.BatchNorm2d(16)),
        param(32, 16, {3: 2, 1: 1}, 2, 8, nn.ReLU(True), nn.BatchNorm2d(16)),
    ],
)
def test_reparam(in_chn, out_chn, kernel, stride, groups, act, norm):
    qat_mode.set_qat_mode(qat_mode.QATMode.WithBN)

    block = OnlineReParamBlock(
        in_channels=in_chn,
        out_channels=out_chn,
        kernel_args=kernel,
        stride=stride,
        groups=groups,
        act_layer=act,
        norm_layer=norm,
    )

    x = torch.randn([2, in_chn, 16, 32], dtype=torch.float32)
    float_out = block(x)
    assert block.merge_conv is None
    block.fuse_model()
    assert isinstance(block.merge_conv, nn.Module)
    merge_out = block(x)
    assert (float_out - merge_out).abs().sum() == 0.0
    qat_mode.set_qat_mode(qat_mode.QATMode.FuseBN)
