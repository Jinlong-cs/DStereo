# Copyright (c) Horizon Robotics. All rights reserved.

import pytest

from hat.models.task_modules.ipm_seg import MaskcatFeatHead
from tests.unit_tests.models.base import qat_test, qtensor_test
from tests.utils import gen_fake_feats


@pytest.mark.parametrize(
    [
        "num_classes",
        "in_channels",
        "out_channels",
        "stacked_convs",
        "has_project_layer",
    ],
    [
        pytest.param(4, 32, 32, 6, True),
        pytest.param(6, 64, 32, 4, False),
        pytest.param(2, 64, 16, 2, True),
    ],
)
def test_seg_head(
    num_classes, in_channels, out_channels, stacked_convs, has_project_layer
):
    in_strides_list = [
        [4],
        [4, 8],
        [4, 8, 16, 32, 64],
        [4, 8, 16, 32, 64],
        [4, 8, 16, 32, 64],
    ]
    out_strides_list = [[4], [4], [2, 16, 32], [4, 16, 32], [2]]
    start_level_list = [0, 0, 1, 1, 0]
    end_level_list = [0, 0, 3, 3, 0]
    use_auxi_loss_list = [True, True, False, True, True]
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
        seg_head = MaskcatFeatHead(
            num_classes,
            in_strides,
            out_strides,
            in_channels=in_channels,
            out_channels=out_channels,
            stacked_convs=stacked_convs,
            start_level=start_level,
            end_level=end_level,
            has_project_layer=has_project_layer,
            use_auxi_loss=use_auxi,
            bn_kwargs=dict(eps=2e-5, momentum=0.1),
        )

        assert len(seg_head.conv_seg_layers[0]) == stacked_convs
        if has_project_layer:
            assert len(seg_head.project_layers) == len(out_strides)
        # do forward and check sometings
        outs = seg_head(feats)
        if use_auxi:
            assert len(outs) == len(out_strides)
            assert len(seg_head.conv_cls_layers) == len(out_strides)
        for ii, out in enumerate(outs):
            assert out.shape[2:] == (
                h // out_strides[ii],
                w // out_strides[ii],
            )
            assert out.size(1) == num_classes
        # test qat
        feats_qat = [qtensor_test(feat) for feat in feats]
        qat_test(seg_head, feats_qat, with_quantized=False)


@pytest.mark.parametrize(
    [
        "num_classes",
        "in_channels",
        "out_channels",
        "stacked_convs",
        "has_project_layer",
    ],
    [
        pytest.param(4, 32, 32, 6, True),
        pytest.param(6, 64, 32, 4, False),
        pytest.param(2, 64, 16, 2, True),
    ],
)
def test_seg_head_with_upscale_output(
    num_classes, in_channels, out_channels, stacked_convs, has_project_layer
):
    in_strides_list = [
        [4],
        [4, 8],
        [4, 8, 16, 32, 64],
        [4, 8, 16, 32, 64],
        [4, 8, 16, 32, 64],
    ]
    out_strides_list = [[4], [4], [2, 16, 32], [4, 16, 32], [2]]
    start_level_list = [0, 0, 1, 1, 0]
    end_level_list = [0, 0, 3, 3, 0]
    use_auxi_loss_list = [True, True, False, True, True]
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
        seg_head = MaskcatFeatHead(
            num_classes,
            in_strides,
            out_strides,
            in_channels=in_channels,
            out_channels=out_channels,
            stacked_convs=stacked_convs,
            start_level=start_level,
            end_level=end_level,
            has_project_layer=has_project_layer,
            use_auxi_loss=use_auxi,
            bn_kwargs=dict(eps=2e-5, momentum=0.1),
            upsample_output_scale=True,
        )

        assert len(seg_head.conv_seg_layers[0]) == stacked_convs
        if has_project_layer:
            assert len(seg_head.project_layers) == len(out_strides)
        # do forward and check sometings
        outs = seg_head(feats)
        if use_auxi:
            assert len(outs) == len(out_strides)
            assert len(seg_head.conv_cls_layers) == len(out_strides)
        for _, out in enumerate(outs):
            assert out.shape[2:] == (h, w)
            assert out.size(1) == num_classes
        # test qat
        feats_qat = [qtensor_test(feat) for feat in feats]
        qat_test(seg_head, feats_qat, with_quantized=False)
