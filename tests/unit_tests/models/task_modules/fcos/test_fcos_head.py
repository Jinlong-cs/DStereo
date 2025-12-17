# Copyright (c) Horizon Robotics. All rights reserved.


import pytest

from hat.models.task_modules.fcos import FCOSHead
from tests.unit_tests.models.base import qat_test, qtensor_test
from tests.utils import gen_fake_feats


@pytest.mark.parametrize(
    [
        "num_classes",
        "feat_channels",
        "stacked_convs",
        "use_sigmoid",
        "share_bn",
        "use_plain_conv",
        "use_gn",
        "use_scale",
        "add_stride",
        "set_all_int16_qconfig",
    ],
    [
        pytest.param(
            10, 64, 4, True, False, False, False, False, False, False
        ),
        pytest.param(1, 32, 4, True, False, False, False, False, False, True),
        pytest.param(
            80, 64, 2, False, True, False, False, False, False, False
        ),
        pytest.param(10, 32, 4, True, True, True, True, True, True, True),
        pytest.param(1, 32, 4, True, True, True, False, True, True, False),
        pytest.param(80, 64, 2, False, True, False, False, True, True, False),
        pytest.param(80, 64, 2, False, True, False, True, False, True, True),
    ],
)
def test_fcos_head(
    num_classes,
    feat_channels,
    stacked_convs,
    use_sigmoid,
    share_bn,
    use_plain_conv,
    use_gn,
    use_scale,
    add_stride,
    set_all_int16_qconfig,
):
    in_strides_list = [
        (4, 8, 16, 32, 64),
        (8, 16, 32, 64),
        (8, 16, 32, 64),
        (8, 16, 32),
        (4, 8, 16, 32, 64),
    ]
    out_strides_list = [
        (8, 16, 32, 64),
        (8, 16, 32, 64),
        (8, 16, 32, 64, 128),
        (8, 16, 32, 64),
        (8, 16, 32, 64, 128),
    ]
    for in_strides, out_strides in zip(in_strides_list, out_strides_list):
        w, h = 640, 640
        feats, stride2channels, stride2hw = gen_fake_feats(w, h, in_strides)

        if (set(out_strides).difference(in_strides) and not add_stride) or (
            set(out_strides).issubset(in_strides) and add_stride
        ):
            with pytest.raises(AssertionError):
                FCOSHead(
                    num_classes,
                    in_strides,
                    out_strides,
                    stride2channels,
                    True,
                    feat_channels,
                    stacked_convs,
                    use_sigmoid,
                    share_bn,
                    use_plain_conv=use_plain_conv,
                    use_gn=use_gn,
                    use_scale=use_scale,
                    add_stride=add_stride,
                    set_all_int16_qconfig=set_all_int16_qconfig,
                )
            continue

        # generate fake feats
        fcos_head = FCOSHead(
            num_classes,
            in_strides,
            out_strides,
            stride2channels,
            True,
            feat_channels,
            stacked_convs,
            use_sigmoid,
            share_bn,
            use_plain_conv=use_plain_conv,
            use_gn=use_gn,
            use_scale=use_scale,
            add_stride=add_stride,
            set_all_int16_qconfig=set_all_int16_qconfig,
        )

        assert fcos_head.conv_cls.out_channels == num_classes + (
            1 - int(use_sigmoid)
        )

        if add_stride:
            assert len(fcos_head.extra_strides) == len(
                fcos_head.extra_stride_convs
            )

        if share_bn:
            assert len(fcos_head.cls_convs) == stacked_convs
            assert len(fcos_head.cls_convs) == stacked_convs
        else:
            assert len(fcos_head.cls_convs) == len(out_strides)
            assert len(fcos_head.reg_convs) == len(out_strides)
            assert len(fcos_head.cls_convs[0]) == stacked_convs
            assert len(fcos_head.cls_convs[0]) == stacked_convs

        # do forward and check something
        assert len(feats) == len(in_strides)
        cls_scores, bbox_preds, centernesses = fcos_head(feats)
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        assert len(cls_scores) == len(out_strides)
        for ii, cls_score in enumerate(cls_scores):
            if not fcos_head.add_stride:
                assert cls_score.shape[2:] == stride2hw[out_strides[ii]]

        # test share_bn fuse_model
        if (
            not fcos_head.use_plain_conv
            and not fcos_head.use_gn
            and not fcos_head.use_scale
        ):
            feats_qat = [qtensor_test(feat) for feat in feats]
            qat_test(
                fcos_head,
                feats_qat,
                with_quantized=False,
            )

        # When using ConvModule2D rather than SeparableConvModule2d (in this
        # case, ConvModule2D is broken into Conv2D, Norm, ReLU layers whose
        # fusion functions are not implemented individually), GroupNorm rather
        # than BatchNorm, and Scale function in training a big model, model
        # fusion should not be performed as these modules' fusion functions are
        # not implemented.
        if fcos_head.use_plain_conv or fcos_head.use_gn or fcos_head.use_scale:
            with pytest.raises(NotImplementedError):
                feats_qat = [qtensor_test(feat) for feat in feats]
                qat_test(
                    fcos_head,
                    feats_qat,
                    with_quantized=False,
                )
