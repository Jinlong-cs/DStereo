# Copyright (c) Horizon Robotics. All rights reserved.

import pytest
import torch

from hat.models.task_modules.roi_modules.kps_head import RCNNKPSSplitHead
from tests.unit_tests.models.base import qat_test, qtensor_test


@pytest.mark.parametrize(
    ["points_num", "in_channel", "upscale", "int8_output"],
    [
        pytest.param(10, 32, True, False),
        pytest.param(1, 64, False, True),
        pytest.param(8, 128, False, False),
    ],
)
def test_RCNNKPSSplitHead(
    points_num,
    in_channel,
    upscale,
    int8_output,
):
    input_size = 416
    input = torch.randn((1, in_channel, input_size // 8, input_size // 8))

    kps_split_head = RCNNKPSSplitHead(
        points_num=points_num,
        in_channel=in_channel,
        upscale=upscale,
        int8_output=int8_output,
    )
    output = kps_split_head(input)

    assert output["kps_rcnn_cls_pred"].size(1) == points_num
    assert output["kps_rcnn_reg_pred"].size(1) == points_num * 2

    # test share_bn fuse_model
    input_qat = qtensor_test(input)
    qat_test(kps_split_head, input_qat, with_quantized=False)
