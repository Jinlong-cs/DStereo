# Copyright (c) Horizon Robotics. All rights reserved.


import pytest

from hat.models.task_modules.autoassign import AutoAssignHead
from tests.unit_tests.models.base import qat_test, qtensor_test
from tests.utils import gen_fake_feats


@pytest.mark.parametrize(
    [
        "num_classes",
        "feat_channels",
        "stacked_convs",
        "use_sigmoid",
        "share_bn",
    ],
    [
        pytest.param(10, 64, 4, True, False),
        pytest.param(1, 32, 4, True, False),
        pytest.param(80, 64, 2, False, True),
    ],
)
def test_autoassign_head(
    num_classes, feat_channels, stacked_convs, use_sigmoid, share_bn
):
    in_strides_list = [
        (4, 8, 16, 32, 64),
        (8, 16, 32, 64),
        (8, 16, 32, 64, 128),
    ]
    out_strides_list = [
        (8, 16, 32, 64),
        (8, 16, 32, 64),
        (8, 16, 32, 64, 128),
    ]
    for in_strides, out_strides in zip(in_strides_list, out_strides_list):
        # generate fake feats
        w, h = 640, 640
        feats, stride2channels, stride2hw = gen_fake_feats(w, h, in_strides)
        autoassign_head = AutoAssignHead(
            num_classes,
            in_strides,
            out_strides,
            stride2channels,
            True,
            feat_channels,
            stacked_convs,
            use_sigmoid,
            share_bn,
        )
        assert autoassign_head.conv_cls.out_channels == num_classes + (
            1 - int(use_sigmoid)
        )
        if share_bn:
            assert len(autoassign_head.cls_convs) == stacked_convs
            assert len(autoassign_head.cls_convs) == stacked_convs
        else:
            assert len(autoassign_head.cls_convs) == len(out_strides)
            assert len(autoassign_head.reg_convs) == len(out_strides)
            assert len(autoassign_head.cls_convs[0]) == stacked_convs
            assert len(autoassign_head.cls_convs[0]) == stacked_convs
        # do forward and check sometings
        assert len(feats) == len(in_strides)
        cls_scores, bbox_preds, centernesses = autoassign_head(feats)
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        assert len(cls_scores) == len(out_strides)
        for ii, cls_score in enumerate(cls_scores):
            assert cls_score.shape[2:] == stride2hw[out_strides[ii]]

        # test share_bn fuse_model
        feats_qat = [qtensor_test(feat) for feat in feats]
        qat_test(autoassign_head, feats_qat, with_quantized=False)
