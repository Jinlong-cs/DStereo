# Copyright (c) Horizon Robotics. All rights reserved.

import torch

from hat.models.task_modules.ipm_psd.super_psd_target import (
    SuperPSDGlobalTarget,
    SuperPSDLocalTarget,
)


def test_superpsd_global_target():

    input_size = 896
    feature_size = 28
    global_labels = [
        [
            [
                126.93574999999998,
                708.6539999999999,
                -1,
                -0.9999964833259583,
                0.002632478717714548,
                0,
                253.972,
                645.179,
                253.64,
                771.461,
                0.131,
                770.644,
                0.0,
                647.332,
            ]
        ]
    ]
    target_keys = [
        "global_classification_obj",
        "global_offset_obj",
        "global_occupancy_obj",
        "global_slot_type_obj",
        "global_direction_obj",
    ]
    grad_keys = [
        "global_classification_grad",
        "global_offset_grad",
        "global_occupancy_grad",
        "global_slot_type_grad",
        "global_direction_grad",
    ]
    global_preds = [
        torch.randn(1, 1, 28, 28),
        torch.randn(1, 8, 28, 28),
        torch.randn(1, 1, 28, 28),
        torch.randn(1, 3, 28, 28),
        torch.randn(1, 2, 28, 28),
    ]
    target = SuperPSDGlobalTarget(
        input_size=input_size, feature_size=feature_size
    )
    global_target_dict, global_grad_dict = target(global_preds, global_labels)

    for tk in target_keys:
        assert tk in global_target_dict
    for gk in grad_keys:
        assert gk in global_grad_dict


def test_superpsd_local_target():

    input_size = 896
    feature_size = 224
    radius = 6
    local_labels = [
        [
            [
                253.972,
                645.179,
                253.64,
                771.461,
                0.131,
                770.644,
                0.0,
                647.332,
                -0.9999640583992004,
                0.008476827293634415,
                -0.9999948143959045,
                -0.0032228140626102686,
                0.9999948143959045,
                0.0032228140626102686,
                0.9999640583992004,
                -0.008476827293634415,
                1,
                1,
                0,
                0,
                -1,
            ]
        ]
    ]
    target_keys = [
        "local_classification_obj",
        "local_offset_obj",
        "local_sline_angle_obj",
        "local_point_type_obj",
    ]
    grad_keys = [
        "local_classification_grad",
        "local_offset_grad",
        "local_sline_angle_grad",
        "local_point_type_grad",
    ]
    local_preds = [
        torch.randn(1, 4, 224, 224),
        torch.randn(1, 8, 224, 224),
        torch.randn(1, 8, 224, 224),
        torch.randn(1, 4, 224, 224),
    ]
    target = SuperPSDLocalTarget(
        radius=radius, input_size=input_size, feature_size=feature_size
    )
    local_target_dict, local_grad_dict = target(local_preds, local_labels)

    for tk in target_keys:
        assert tk in local_target_dict
    for gk in grad_keys:
        assert gk in local_grad_dict
