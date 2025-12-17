# Copyright (c) Horizon Robotics. All rights reserved.

import torch

from hat.models.task_modules.ipm_psd.super_psd_heads import (
    SuperPSDGlobalHead,
    SuperPSDLocalHead,
)
from tests.unit_tests.models.base import qat_test, qtensor_test


def test_superpsd_head():

    in_channels = 32
    local_stride_idx = 0
    global_stride_idx = 3
    num_slot_type = 3
    features = [
        torch.randn(1, in_channels, 224, 224),
        torch.randn(1, in_channels, 112, 112),
        torch.randn(1, in_channels, 56, 56),
        torch.randn(1, in_channels, 28, 28),
    ]
    global_head = SuperPSDGlobalHead(
        num_slot_type, in_channels, global_stride_idx
    )
    local_head = SuperPSDLocalHead(in_channels, local_stride_idx)

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
    qat_features = qtensor_test(features)
    qat_test(local_head, qat_features, with_quantized=False)
