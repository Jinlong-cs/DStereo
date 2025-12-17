# Copyright (c) Horizon Robotics. All rights reserved.

import pytest
import torch

from hat.models.backbones.mixvargenet import MixVarGENetConfig
from hat.models.task_modules.roi_modules.rcnn_head import (
    RCNNMixVarGEShareHead,
    RCNNVarGNetHead,
    RCNNVarGNetShareHead,
    RCNNVarGNetSplitHead,
)
from tests.unit_tests.models.base import qat_test, qtensor_test


@pytest.mark.parametrize(
    [
        "bias",
        "upscale",
    ],
    [
        pytest.param(True, False),
        pytest.param(False, True),
        pytest.param(True, True),
    ],
)
def test_RCNNMixVarGEShareHead(bias, upscale):
    data_shape = [16, 16, 16, 32]
    fake_data = torch.randn(size=data_shape)
    output_channels = 8
    head_config = [
        MixVarGENetConfig(
            in_channels=data_shape[1],
            out_channels=data_shape[1],
            head_op="mixvarge_f2",
            stack_ops=[],
            stride=1,
            fusion_strides=(),
            extra_downsample_num=0,
        ),
        MixVarGENetConfig(
            in_channels=data_shape[1],
            out_channels=data_shape[1],
            head_op="mixvarge_f2",
            stack_ops=[],
            stride=1,
            fusion_strides=(),
            extra_downsample_num=0,
        ),
        MixVarGENetConfig(
            in_channels=data_shape[1],
            out_channels=output_channels,
            head_op="mixvarge_f2",
            stack_ops=[],
            stride=1,
            fusion_strides=(),
            extra_downsample_num=0,
        ),
    ]
    share_head = RCNNMixVarGEShareHead(
        bias=bias,
        bn_kwargs={},
        head_config=head_config,
        upscale=upscale,
    )
    output = share_head(fake_data)
    assert output is not None
    assert output.size(1) == output_channels
    assert output.shape[2] == data_shape[2] << int(upscale)
    assert output.shape[3] == data_shape[3] << int(upscale)

    # test share_bn fuse_model
    input_qat = qtensor_test(fake_data)
    qat_test(share_head, input_qat, with_quantized=False)


@pytest.mark.parametrize(
    [
        "upscale",
        "upscale_conv",
    ],
    [
        pytest.param(True, True),
        pytest.param(True, False),
        pytest.param(False, False),
    ],
)
def test_RCNNVarGNetShareHead(upscale, upscale_conv):
    data_shape = [16, 16, 16, 32]
    fake_data = torch.randn(size=data_shape)
    output_channels = 8
    share_head = RCNNVarGNetShareHead(
        bn_kwargs={},
        roi_out_channel=16,
        upscale=upscale,
        upscale_conv=upscale_conv,
        gc_num_filter=16,
        pw_num_filter=16,
        pw_num_filter2=output_channels,
        group_base=4,
        factor=1,
        stride=2,
    )
    output = share_head(fake_data)
    assert output is not None
    assert output.size(1) == output_channels
    assert output.shape[2] == (data_shape[2] // 2) << int(upscale)
    assert output.shape[3] == (data_shape[3] // 2) << int(upscale)

    # test share_bn fuse_model
    input_qat = qtensor_test(fake_data)
    qat_test(share_head, input_qat, with_quantized=False)


@pytest.mark.parametrize(
    [
        "num_fg_classes",
        "in_channel",
        "class_agnostic_reg",
        "with_background",
        "with_box_reg",
        "with_tracking_feat",
        "use_bin",
        "upscale",
        "upscale_conv",
    ],
    [
        pytest.param(10, 32, True, False, False, True, True, False, False),
        pytest.param(1, 64, False, True, False, False, False, True, True),
        pytest.param(1, 64, False, True, False, False, False, True, False),
        pytest.param(80, 128, False, False, True, True, False, False, False),
    ],
)
def test_RCNNVarGNetSplitHead(
    num_fg_classes,
    in_channel,
    class_agnostic_reg,
    with_background,
    with_box_reg,
    with_tracking_feat,
    use_bin,
    upscale,
    upscale_conv,
):
    input_shape = (16, 32)
    reg_channel_base = 4
    x = torch.randn((1, in_channel, *input_shape))
    with_bg = 1 if with_background else 0
    pw_num_filter2 = 128

    split_head = RCNNVarGNetSplitHead(
        num_fg_classes=num_fg_classes,
        bn_kwargs={},
        class_agnostic_reg=class_agnostic_reg,
        with_background=with_background,
        in_channel=in_channel,
        pw_num_filter2=pw_num_filter2,
        with_box_reg=with_box_reg,
        reg_channel_base=reg_channel_base,
        use_bin=use_bin,
        upscale=upscale,
        upscale_conv=upscale_conv,
        with_tracking_feat=with_tracking_feat,
    )
    output = split_head(x)

    factor = 0 if use_bin else 3

    if with_box_reg:
        if class_agnostic_reg:
            assert output["rcnn_reg_pred"].size(1) == reg_channel_base
        else:
            assert (
                output["rcnn_reg_pred"].size(1)
                == num_fg_classes * reg_channel_base
            )
        assert (
            output["rcnn_reg_pred"].shape[2]
            == (input_shape[0] << int(upscale)) - factor
        )
        assert (
            output["rcnn_reg_pred"].shape[3]
            == (input_shape[1] << int(upscale)) - factor
        )

    if with_tracking_feat:
        assert output["rcnn_tracking_feat"].size(1) == pw_num_filter2

    assert output["rcnn_cls_pred"].size(1) == num_fg_classes + with_bg
    assert (
        output["rcnn_cls_pred"].shape[2]
        == (input_shape[0] << int(upscale)) - factor
    )
    assert (
        output["rcnn_cls_pred"].shape[3]
        == (input_shape[1] << int(upscale)) - factor
    )

    # test share_bn fuse_model
    input_qat = qtensor_test(x)
    qat_test(split_head, input_qat, with_quantized=False)


@pytest.mark.parametrize(
    [
        "num_fg_classes",
        "class_agnostic_reg",
        "with_background",
        "roi_out_channel",
    ],
    [
        pytest.param(10, True, True, 32),
        pytest.param(1, True, False, 64),
        pytest.param(80, False, False, 128),
    ],
)
def test_RCNNVarGNetHead(
    num_fg_classes,
    class_agnostic_reg,
    with_background,
    roi_out_channel,
):
    input_size = 416
    x = torch.randn((1, roi_out_channel, input_size // 8, input_size // 8))

    with_bg = 1 if with_background else 0
    with_box_cls = True
    with_box_reg = True
    with_tracking_feat = True

    pw_num_filter2 = 96
    reg_channel_base = 4

    vargnet_head = RCNNVarGNetHead(
        num_fg_classes=num_fg_classes,
        bn_kwargs={},
        class_agnostic_reg=class_agnostic_reg,
        with_background=with_background,
        roi_out_channel=roi_out_channel,
        dw_num_filter=128,
        pw_num_filter=128,
        pw_num_filter2=pw_num_filter2,
        group_base=8,
        factor=1,
        with_box_cls=with_box_cls,
        with_box_reg=with_box_reg,
        reg_channel_base=reg_channel_base,
        with_tracking_feat=with_tracking_feat,
    )
    output = vargnet_head(x)

    if with_box_reg:
        if class_agnostic_reg:
            assert output["rcnn_reg_pred"].size(1) == reg_channel_base
        else:
            assert (
                output["rcnn_reg_pred"].size(1)
                == num_fg_classes * reg_channel_base
            )
    if with_box_cls:
        assert output["rcnn_cls_pred"].size(1) == num_fg_classes + with_bg
    if with_tracking_feat:
        assert output["rcnn_tracking_feat"].size(1) == pw_num_filter2

    # test share_bn fuse_model
    input_qat = qtensor_test(x)
    qat_test(vargnet_head, input_qat, with_quantized=False)
