# Copyright (c) Horizon Robotics. All rights reserved.

import torch

from hat.models.task_modules.bev.bev_psd_heads import (
    ANCBEVPSDGlobalHead,
    ANCBEVPSDLocalHead,
)

# from tests.unit_tests.models.base import qat_test, qtensor_test


def test_bevpsd_head():
    num_slot_type = 3
    in_channels = 32
    out_channels = 32
    in_strides = [2, 4, 8, 16, 32]
    global_downsample_factor = 4
    features = [
        torch.randn(1, in_channels, 192, 128),
        torch.randn(1, in_channels, 96, 64),
        torch.randn(1, in_channels, 48, 32),
        torch.randn(1, in_channels, 24, 16),
        torch.randn(1, in_channels, 12, 8),
    ]
    global_head = ANCBEVPSDGlobalHead(
        num_slot_type=num_slot_type,
        in_channels=in_channels,
        out_channels=out_channels,
        in_strides=in_strides,
        out_stride=global_downsample_factor * 2,
        stack=1,
        group_base=8,
    )
    (
        global_classification,
        global_offset,
        global_occupancy,
        global_slot_type,
        global_direction,
    ) = global_head(features)
    assert (
        len(global_classification)
        == len(global_offset)
        == len(global_occupancy)
        == len(global_slot_type)
        == len(global_direction)
    )
    local_downsample_factor = 1
    local_head = ANCBEVPSDLocalHead(
        in_channels=in_channels,
        out_channels=out_channels,
        in_strides=in_strides,
        out_stride=local_downsample_factor * 2,
        stack=1,
        group_base=8,
    )
    (
        local_classification,
        local_offset,
        local_sline_angle,
        local_point_type,
    ) = local_head(features)
    assert (
        len(local_classification)
        == len(local_offset)
        == len(local_sline_angle)
        == len(local_point_type)
    )
