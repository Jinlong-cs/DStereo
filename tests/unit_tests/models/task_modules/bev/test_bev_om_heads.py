import pytest
import torch

from hat.models.backbones.vargnetv2 import get_vargnetv2_stride2channels
from hat.models.task_modules.bev.head import ANCBEVOMHead


@pytest.mark.parametrize(
    [
        "out_stride",
        "out_nums",
        "output_name",
        "select_strides",
        "align_channels",
        "repeat_times",
        "repeat_channels",
    ],
    [
        pytest.param(2, [2], ["cls"], [2], 96, 1, 96),
        pytest.param(4, [2], ["cls"], [2], 96, 1, 96),
        pytest.param(4, [2, 1, 1], ["cls", "sin", "cos"], [2], 96, 1, 96),
        pytest.param(4, [2, 1], ["cls", "sin"], [2, 4], 96, 1, 96),
        pytest.param(4, [2, 1], ["cls", "sin"], [2, 4], 96, 2, 96),
        pytest.param(4, [2, 1], ["cls", "sin"], [2, 4], 96, 2, 48),
        pytest.param(8, [2, 1], ["cls", "sin"], [2, 4], 96, 2, 48),
    ],
)
def test_BEVOMHead(
    out_stride,
    out_nums,
    output_name,
    select_strides,
    align_channels,
    repeat_times,
    repeat_channels,
):
    alpha = 0.5
    feature_name = "feats"
    in_strides = [2, 4, 8, 16, 32]
    stride2channels = get_vargnetv2_stride2channels(alpha)
    bev_om_head = ANCBEVOMHead(
        feature_name=feature_name,
        in_strides=in_strides,
        select_strides=select_strides,
        out_stride=out_stride,
        align_channels=align_channels,
        stride2channels=stride2channels,
        out_nums=out_nums,
        output_name=output_name,
        repeat_times=repeat_times,
        repeat_channels=repeat_channels,
        bn_kwargs={},
    )

    bev_bs = 2
    input_bev_size = (512, 512)

    bev_stage2_feats = [
        torch.randn(
            (
                bev_bs,
                stride2channels[x],
                input_bev_size[0] // x,
                input_bev_size[1] // x,
            )
        )
        for x in [2, 4, 8, 16, 32]
    ]

    data = {}
    data[feature_name] = [bev_stage2_feats]
    out = bev_om_head(data)
    for name, num in zip(out, out_nums):
        assert out[name][0].shape == (
            bev_bs,
            num,
            input_bev_size[0] // out_stride,
            input_bev_size[0] // out_stride,
        )
