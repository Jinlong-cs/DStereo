# Copyright (c) Horizon Robotics. All rights reserved.


import math
from collections import OrderedDict

import pytest

from hat.models.task_modules.mtfcos3d import MTFCOS3DHead
from tests.unit_tests.models.base import qat_test, qtensor_test
from tests.utils import gen_fake_feats

PI = math.pi


@pytest.mark.parametrize(
    [
        "num_classes",
        "feat_channels",
        "stacked_convs",
        "use_sigmoid",
        "share_bn",
        "use_multibin",
        "multibin_centers",
    ],
    [
        pytest.param(2, 64, 4, True, False, True, (0.0, PI / 2, PI, -PI / 2)),
        pytest.param(1, 32, 4, True, False, False, (0.0, PI / 2, PI, -PI / 2)),
        pytest.param(4, 64, 2, True, True, True, (0.0, PI)),
        pytest.param(2, 32, 4, True, True, False, None),
        pytest.param(1, 32, 4, True, True, True, (0.0, PI)),
        pytest.param(4, 64, 2, True, True, False, None),
        pytest.param(4, 64, 2, True, True, True, (0.0, PI)),
    ],
)
def test_mtfcos3d_head(
    num_classes,
    feat_channels,
    stacked_convs,
    use_sigmoid,
    share_bn,
    use_multibin,
    multibin_centers,
):
    in_strides_list = [
        (4, 8, 16, 32, 64),
        (8, 16, 32, 64, 128),
        (8, 16, 32, 64, 128),
        (8, 16, 32, 64),
        (4, 8, 16, 32, 64, 128),
    ]
    out_strides_list = [
        (8, 16, 32, 64),
        (8, 16, 32, 64),
        (8, 16, 32, 64, 128),
        (8, 16, 32),
        (8, 16, 32, 64, 128),
    ]

    multibin_margin = 10.0 / 180.0 * PI

    head_channels = OrderedDict(
        cls=[num_classes],
        offset_2d_reg=[4],
        offset_3d_group_reg=[2],
        depth_3d_group_reg=[1],
        dim_3d_group_reg=[3],
        dir_reg=[2]
        if not use_multibin
        else [len(multibin_centers)],  # hard code should be fix
        ctrness_2d_reg=[1],
        ctrness_3d_reg=[1],
    )
    if use_multibin:
        if multibin_margin > 0.0:
            head_channels.update(rot_3d_group_reg=[len(multibin_centers)])
        else:
            head_channels.update(rot_3d_group_reg=[1])
    else:
        head_channels.update(rotsin_3d_group_reg=[1])  # local yaw/alpha

    for in_strides, out_strides in zip(in_strides_list, out_strides_list):
        w, h = 640, 640
        feats, stride2channels, _ = gen_fake_feats(w, h, in_strides)

        # generate fake feats
        mtfcos3d_head = MTFCOS3DHead(
            num_classes=num_classes,
            in_strides=in_strides,
            out_strides=out_strides,
            head_channels=head_channels,
            stride2channels=stride2channels,
            upscale_bbox_pred=True,
            feat_channels=feat_channels,
            stacked_convs=stacked_convs,
            use_sigmoid=use_sigmoid,
            share_bn=share_bn,
            int8_output=False,
        )

        if share_bn:
            assert len(mtfcos3d_head.cls_convs) == stacked_convs
            assert len(mtfcos3d_head.cls_convs) == stacked_convs
        else:
            assert len(mtfcos3d_head.cls_convs) == len(out_strides)
            assert len(mtfcos3d_head.reg_convs) == len(out_strides)
            assert len(mtfcos3d_head.cls_convs[0]) == stacked_convs
            assert len(mtfcos3d_head.cls_convs[0]) == stacked_convs

        # do forward and check something
        assert len(feats) == len(in_strides)
        pred = mtfcos3d_head(feats)
        assert len(pred) == len(head_channels)

        feats_qat = [qtensor_test(feat) for feat in feats]
        qat_test(mtfcos3d_head, feats_qat, with_quantized=False)
