# Copyright (c) Horizon Robotics. All rights reserved.

from collections import OrderedDict

import numpy as np
import pytest

from hat.models.task_modules.mtfcos3d import MTFCOS3DTarget
from tests.unit_tests.models.task_modules.mtfcos3d.utils import (
    gen_fake_mtf_real3d_label_pred_data,
)

INF = 1e8
default_regress_ranges = (
    (-1, 64),
    (64, 128),
    (128, 256),
    (256, 512),
    (512, INF),
)


@pytest.mark.parametrize(
    ["h", "w", "strides", "num_classes", "use_iou_replace_ctrness"],
    [
        pytest.param(640, 640, (4, 8, 16, 32, 64), 2, False),
        pytest.param(512, 512, (8, 16, 32, 64), 5, False),
        pytest.param(640, 640, (4, 8, 16, 32, 64), 2, True),
        pytest.param(512, 512, (8, 16, 32, 64), 5, True),
    ],
)
def test_mtfcos3d_target(h, w, strides, num_classes, use_iou_replace_ctrness):
    regress_ranges = default_regress_ranges[: len(strides)]
    multibin_centers = (0.0, np.pi / 2, np.pi, -np.pi / 2)
    head_channels = OrderedDict(
        cls=[num_classes],
        offset_2d_reg=[4],
        offset_3d_group_reg=[2],
        depth_3d_group_reg=[1],
        dim_3d_group_reg=[3],
        dir_reg=[len(multibin_centers)],
        ctrness_2d_reg=[1],
        ctrness_3d_reg=[1],
        rotsin_3d_group_reg=[1],
    )

    mtfcos3d_target = MTFCOS3DTarget(
        num_classes=num_classes,
        strides=strides,
        regress_ranges=regress_ranges,
        head_channels=head_channels,
        use_iou_replace_ctrness=use_iou_replace_ctrness,
    )
    label, pred = gen_fake_mtf_real3d_label_pred_data(
        h, w, strides, head_channels
    )
    outputs = mtfcos3d_target(label, pred)
    assert outputs["cls"]["pred"].shape[-1] == num_classes
    assert outputs["offset_2d_reg"]["pred"].shape[-1] == 4
    assert (
        outputs["ctrness_2d_reg"]["pred"].shape[0]
        == outputs["offset_2d_reg"]["pred"].shape[0]
    )
