# Copyright (c) Horizon Robotics. All rights reserved.

import pytest
import torch

from hat.models.backbones.mixvargenet import MixVarGENetConfig
from hat.models.task_modules.roi_modules.roi_3d_head import (
    RCNNHM3DMixVarGEHead,
    RCNNHM3DVarGNetHead,
)
from tests.unit_tests.models.base import qat_test, qtensor_test


@pytest.mark.parametrize(
    [
        "rot_channel",
        "stop_gradient",
        "with_iou_pre",
        "int8_output",
        "undistort_depth_uv",
    ],
    [
        pytest.param(2, True, True, True, False),
        pytest.param(2, False, True, False, True),
        pytest.param(1, False, False, False, False),
    ],
)
def test_RCNNHM3DVarGNetHead(
    rot_channel,
    stop_gradient,
    with_iou_pre,
    int8_output,
    undistort_depth_uv,
):
    input_size = 64
    roi_out_channel = 16
    input = torch.randn(
        (10, roi_out_channel, input_size // 8, input_size // 8)
    )

    roi_3d_head = RCNNHM3DVarGNetHead(
        bn_kwargs={},
        dw_num_filter=64,
        pw_num_filter=64,
        pw_num_filter2=64,
        roi_out_channel=roi_out_channel,
        group_base=8,
        factor=1,
        rot_channel=rot_channel,
        is_dim_match=False,
        stop_gradient=stop_gradient,
        with_iou_pred=with_iou_pre,
        undistort_depth_uv=undistort_depth_uv,
        int8_output=int8_output,
    )
    output = roi_3d_head(input)

    assert output["cls_pred"].size(1) == 1
    assert output["offset_2d_pred"].size(1) == 2
    assert output["offset_3d_pred"].size(1) == 2
    if undistort_depth_uv:
        assert output["depth_u_pred"].size(1) == 1
        assert output["depth_v_pred"].size(1) == 1
    else:
        assert output["depth_pred"].size(1) == 1
    assert output["dims_pred"].size(1) == 3
    assert output["rot_pred"].size(1) == rot_channel
    if with_iou_pre:
        assert output["iou_pred"].size(1) == 1

    # test share_bn fuse_model
    input_qat = qtensor_test(input)
    qat_test(roi_3d_head, input_qat, with_quantized=False)


@pytest.mark.parametrize(
    [
        "rot_channel",
        "stop_gradient",
        "with_iou_pre",
        "int8_output",
        "undistort_depth_uv",
    ],
    [
        pytest.param(2, True, True, True, False),
        pytest.param(2, False, True, False, True),
        pytest.param(1, False, False, False, False),
    ],
)
def test_RCNNHM3DMixVarGEHead(
    rot_channel,
    stop_gradient,
    with_iou_pre,
    int8_output,
    undistort_depth_uv,
):
    input_size = 64
    roi_out_channel = 16
    input = torch.randn(
        (10, roi_out_channel, input_size // 8, input_size // 8)
    )
    head_config = [
        MixVarGENetConfig(
            in_channels=roi_out_channel,
            out_channels=64,
            head_op="mixvarge_f2",
            stack_ops=[],
            stride=1,
            fusion_strides=(),
            extra_downsample_num=0,
        ),
        MixVarGENetConfig(
            in_channels=64,
            out_channels=64,
            head_op="mixvarge_f4",
            stack_ops=[],
            stride=1,
            fusion_strides=(),
            extra_downsample_num=0,
        ),
        MixVarGENetConfig(
            in_channels=64,
            out_channels=64,
            head_op="mixvarge_f4",
            stack_ops=[],
            stride=1,
            fusion_strides=(),
            extra_downsample_num=0,
        ),
    ]

    roi_3d_head = RCNNHM3DMixVarGEHead(
        bn_kwargs={},
        mid_num_filter=64,
        rot_channel=rot_channel,
        stop_gradient=stop_gradient,
        with_iou_pred=with_iou_pre,
        head_config=head_config,
        undistort_depth_uv=undistort_depth_uv,
        int8_output=int8_output,
    )
    output = roi_3d_head(input)

    assert output["cls_pred"].size(1) == 1
    assert output["offset_2d_pred"].size(1) == 2
    assert output["offset_3d_pred"].size(1) == 2
    if undistort_depth_uv:
        assert output["depth_u_pred"].size(1) == 1
        assert output["depth_v_pred"].size(1) == 1
