# Copyright (c) Horizon Robotics. All rights reserved.

import pytest

from hat.models.task_modules.depth import DepthUnetHead
from tests.unit_tests.models.base import qat_test, qtensor_test
from tests.utils import gen_fake_feats


@pytest.mark.parametrize(
    [
        "in_channels",
        "out_channels",
        "stacked_convs",
    ],
    [
        pytest.param(32, 32, 6),
        pytest.param(64, 32, 4),
        pytest.param(64, 16, 2),
    ],
)
def test_depth_head(in_channels, out_channels, stacked_convs):
    in_strides_list = [
        [2, 4, 8, 16, 32],
    ]
    out_strides_list = [
        [1, 4, 8],
    ]
    start_level_list = [
        0,
    ]
    end_level_list = [
        3,
    ]
    use_auxi_loss_list = [True, True, False, True, True]
    head_out_names = {"depth": "pred_depth"}
    for in_strides, out_strides, start_level, end_level, use_auxi in zip(
        in_strides_list,
        out_strides_list,
        start_level_list,
        end_level_list,
        use_auxi_loss_list,
    ):
        # generate fake feats
        w, h = 896, 896
        feats, _, _ = gen_fake_feats(
            w, h, in_strides, feat_channels=in_channels
        )
        depth_head = DepthUnetHead(
            in_strides=in_strides,
            out_strides=out_strides,
            in_channels=[in_channels] * len(in_strides),
            out_channels=[out_channels] * (end_level - start_level + 1),
            pred_out_channel=2,
            stacked_convs=stacked_convs,
            start_level=start_level,
            end_level=end_level,
            bn_kwargs=dict(eps=2e-5, momentum=0.1),
            group_base=8,
            conv_method="conv2d",
            use_auxi_loss=use_auxi,
            with_refine=False,
            last_with_relu=False,
            dequant_output=True,
            int8_output=True,  # True
            head_out_names=head_out_names,
        )

        assert len(depth_head.conv_depth_layers[0]) == stacked_convs

        outs = depth_head(feats)
        if use_auxi:
            assert len(depth_head.conv_pred_layers) == len(out_strides)
            assert len(outs["pred_depth"]) == len(out_strides)
        for ii, out in enumerate(outs["pred_depth"]):
            assert out.shape[2:] == (
                h // out_strides[ii],
                w // out_strides[ii],
            ), f"{out.shape}, {h}, {w}, {out_strides[ii]}"

        # test qat
        feats_qat = [qtensor_test(feat) for feat in feats]
        qat_test(depth_head, feats_qat, with_quantized=False)
