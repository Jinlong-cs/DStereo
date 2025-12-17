# Copyright (c) Horizon Robotics. All rights reserved.

from hat.data.transforms.bev_psd_target import (
    ANCBEVPSDGlobalTargetGenerator,
    ANCBEVPSDLocalTargetGenerator,
)


def test_bevpsd_global_target():
    input_size = (192, 128)
    grid_stride = 4
    out_size = (input_size[0] // grid_stride, input_size[1] // grid_stride)
    global_slot_weight_dict = {  # noqa
        0: 1.0,
        1: 1.0,
        2: 1.0,
        -1: 0.0,
    }
    data = {
        "annos_bev_psd_obj": {},
        "gt_bev_psd_obj": {},
    }
    data["annos_bev_psd_obj"]["global"] = [
        [
            22.69707,
            192.0,
            22.86434,
            183.97923,
            49.281822,
            184.5914,
            49.034035,
            192.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            8.0,
            0.0,
            0.0,
        ],
        [
            7.583123,
            143.73433,
            28.338324,
            168.80269,
            9.603111,
            184.6475,
            0.0,
            173.14828,
            0.0,
            149.97392,
            0.0,
            0.0,
            0.0,
            10.0,
            0.0,
            0.0,
        ],
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
    target = ANCBEVPSDGlobalTargetGenerator(
        input_size=input_size,
        out_size=out_size,
        slot_weight_dict=global_slot_weight_dict,
    )
    data = target(data)

    for tk in target_keys:
        assert tk in data["gt_bev_psd_obj"].keys()
    for gk in grad_keys:
        assert gk in data["gt_bev_psd_obj"].keys()


def test_bevpsd_local_target():
    input_size = (192, 128)
    grid_stride = 1
    out_size = (input_size[0] // grid_stride, input_size[1] // grid_stride)
    local_slot_weight_dict = {  # noqa
        0: 1.0,
        1: 1.0,
        2: 1.0,
        -1: 0.0,
    }
    radius = 3
    data = {
        "annos_bev_psd_obj": {},
        "gt_bev_psd_obj": {},
    }
    data["annos_bev_psd_obj"]["local"] = [
        [
            22.69706916809082,
            192.0,
            22.86433982849121,
            183.97923278808594,
            49.281822204589844,
            184.59140014648438,
            49.034034729003906,
            192.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            8.0,
            0.0,
            0.0,
        ],
        [
            74.97314453125,
            192.0,
            75.85369873046875,
            166.78338623046875,
            86.59459686279297,
            167.24154663085938,
            86.10718536376953,
            192.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            8.0,
            1.0,
            0.0,
        ],
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
    target = ANCBEVPSDLocalTargetGenerator(
        input_size=input_size,
        out_size=out_size,
        radius=radius,
        slot_weight_dict=local_slot_weight_dict,
    )
    data = target(data)

    for tk in target_keys:
        assert tk in data["gt_bev_psd_obj"].keys()
    for gk in grad_keys:
        assert gk in data["gt_bev_psd_obj"].keys()
