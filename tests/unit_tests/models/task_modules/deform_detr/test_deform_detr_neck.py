import torch
import torch.nn as nn

from hat.models.task_modules.deform_detr import ChannelMapperNeck


def test_ChannelMapperNeck():
    features = [
        torch.rand([1, 4, 60, 40]),
        torch.rand([1, 8, 30, 20]),
        torch.rand([1, 16, 15, 10]),
        torch.rand([1, 32, 8, 5]),
    ]
    module = ChannelMapperNeck(
        in_channels=[8, 16, 32],
        out_indices=[1, 2, 3],
        out_channel=8,
        kernel_size=1,
        extra_convs=1,
        norm_layer=nn.BatchNorm2d(8),
    )
    results = module(features)
    assert len(results) == 4
    for i in range(4):
        assert results[i].shape[1] == 8
