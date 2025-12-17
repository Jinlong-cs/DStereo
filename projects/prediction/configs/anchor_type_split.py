# Copyright (c) Horizon Robotics. All rights reserved.

import numpy as np

from hat.core.traj_pred_utils import normalize_yaw

ANCHORSET_124_TYPE_DICT = {
    "poor": {
        "slow_straight": [3],
        "medium_straight": [11],
        "fast_straight": [20],
        "slow_left": [26],
        "fast_left": [39],
        "slow_right": [47],
        "fast_right": [50],
        "right_lc": [69],
        "right_lc_merge": [77],
        "left_lc": [92],
        "left_lc_merge": [101],
        "ped_right": [112],
        "ped_straight": [113],
        "ped_left": [114],
    },
    "classical": {
        "slow_straight": [0, 1, 2, 3, 4, 5, 6, 7],
        "medium_straight": [8, 9, 10, 11, 12, 13, 14, 15],
        "fast_straight": [16, 17, 18, 19, 20, 21, 22, 23],
        "slow_left": [24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
        "fast_left": [34, 35, 36, 37, 38, 39, 40],
        "slow_right": [41, 42, 43, 44, 45, 46, 47, 48, 49],
        "fast_right": [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
        "right_lc": [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
        "right_lc_merge": [71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82],
        "left_lc": [83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93],
        "left_lc_merge": [
            94,
            95,
            96,
            97,
            98,
            99,
            100,
            101,
            102,
            103,
            104,
            105,
        ],
        "ped_right": [106, 109, 112, 115, 118, 121],
        "ped_straight": [107, 110, 113, 116, 119, 122],
        "ped_left": [108, 111, 114, 117, 120, 123],
    },
    "expand": {
        # 1. vehicle straight
        "slow_straight": [0, 2, 3],
        "medium_straight": [1, 4, 5, 6, 7, 12, 15],
        "fast_straight": [8, 9, 11, 13, 14],
        "very_fast_straight": [10, 17, 20, 22, 23],
        "flying_on_road": [16, 18, 19, 21],
        # 2. vehicle turn left
        "slow_gentle_left": [],
        "slow_sharp_left": [36, 37],
        "medium_gentle_left": [25, 33],
        "medium_sharp_left": [34, 35, 38, 39, 40],
        "fast_gentle_left": [24, 26, 27, 28, 30, 31, 32],
        "fast_sharp_left": [29],
        # 3. vehicle turn right
        "slow_gentle_right": [48],
        "slow_sharp_right": [50, 51, 53, 56],
        "medium_gentle_right": [42, 49],
        "medium_sharp_right": [52, 54, 55, 57, 58, 59],
        "fast_gentle_right": [41, 43, 44, 46, 47],
        "fast_sharp_right": [45],
        # 4. vehicle left lane change
        "slow_gentle_left_lc": [83, 84, 88],
        "slow_sharp_left_lc": [85, 86],
        "fast_gentle_left_lc": [89, 90, 91, 92, 93],
        "fast_sharp_left_lc": [87],
        # 5. vehicle left lane change: merge to straight lane
        "slow_gentle_left_lc_merge": [94, 95, 96, 100],
        "slow_sharp_left_lc_merge": [97, 98],
        "fast_gentle_left_lc_merge": [101, 102, 103, 104, 105],
        "fast_sharp_left_lc_merge": [99],
        # 6. vehicle right lane change
        "slow_gentle_right_lc": [60, 61, 65],
        "slow_sharp_right_lc": [62, 63],
        "fast_gentle_right_lc": [66, 67, 68, 69, 70],
        "fast_sharp_right_lc": [64],
        # 7. vehicle right lane change: merge to straight lane
        "slow_gentle_right_lc_merge": [71, 72, 73, 77],
        "slow_sharp_right_lc_merge": [74, 75],
        "fast_gentle_right_lc_merge": [78, 79, 80, 81, 82],
        "fast_sharp_right_lc_merge": [76],
        # 8. pedestrain
        "ped_right": [106, 109, 112, 115, 118, 121],
        "ped_straight": [107, 110, 113, 116, 119, 122],
        "ped_left": [108, 111, 114, 117, 120, 123],
    },
}

ANCHORSET_139_TYPE_DICT = {
    "classical": {
        "slow_straight": [0, 1, 2, 3, 4, 5, 6, 7],
        "medium_straight": [8, 9, 10, 11, 12, 13, 14, 15],
        "fast_straight": [16, 17, 18, 19, 20, 21, 22, 23, 126, 129, 132, 135],
        "slow_left": [24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
        "fast_left": [34, 35, 36, 37, 38, 39, 40],
        "slow_right": [41, 42, 43, 44, 45, 46, 47, 48, 49],
        "fast_right": [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
        "right_lc": [
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            124,
            127,
            130,
            133,
        ],
        "right_lc_merge": [71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82],
        "left_lc": [
            83,
            84,
            85,
            86,
            87,
            88,
            89,
            90,
            91,
            92,
            93,
            125,
            128,
            131,
            134,
        ],
        "left_lc_merge": [
            94,
            95,
            96,
            97,
            98,
            99,
            100,
            101,
            102,
            103,
            104,
            105,
        ],
        "ped_right": [106, 109, 112, 115, 118, 121],
        "ped_straight": [107, 110, 113, 116, 119, 122],
        "ped_left": [108, 111, 114, 117, 120, 123],
    },
}


def classical_anc_type_traj_classify(
    traj, track_cls, delta_t=0.5
):  # noqa: D205,D400,D401
    """A simple anchor type classification method for the `classical`
    anchor type set

    Args:
        traj (np.array, [traj_len, 2]): the trajectory.
        track_cls (int): the obstacle class. -1: others, 0: vehicle,
            1: pedestrain, 2: cyclist.
        delta_t (float): the delta time between two adjacent trajectory
            points.

    Return:
        type_key (str): the anchor type key. It should be one of the
            values in ANCHOR_TYPE["classical"].
    """
    cur_traj_len = len(traj)
    assert cur_traj_len > 1

    delta_x_y = np.diff(traj, axis=0)
    yaw = np.arctan2(delta_x_y[:, 1], delta_x_y[:, 0])
    delta_yaw = np.diff(yaw, axis=-1)
    delta_s = np.sqrt(np.sum(delta_x_y ** 2, axis=-1))
    mean_velo = np.mean(delta_s, axis=-1) / delta_t  # m/s

    track_is_ped = track_cls == 1
    if track_is_ped:
        type_key = "ped_straight"
        delta_yaw = normalize_yaw(yaw[-1] - yaw[0])
        if delta_yaw > np.pi / 8:
            type_key = "ped_right"
        elif delta_yaw < -np.pi / 8:
            type_key = "ped_left"
    else:
        track_speed_level = ""
        if mean_velo <= 4:  # ~= 15 km/h
            track_speed_level = "slow"
        elif mean_velo <= 8.3:  # ~= 30 km/h
            track_speed_level = "medium"
        else:
            track_speed_level = "fast"

        turn_type = None
        delta_yaw = normalize_yaw(yaw[-1] - yaw[0])
        mid_idx = int(cur_traj_len / 2)
        part_1_mean_yaw = normalize_yaw(np.mean(yaw[:mid_idx]))
        part_2_mean_yaw = normalize_yaw(np.mean(yaw[mid_idx:]))
        if delta_yaw < -np.pi / 4:
            turn_type = "right"
            if track_speed_level == "medium":
                track_speed_level = "slow"
        elif delta_yaw > np.pi / 4:
            turn_type = "left"
            if track_speed_level == "medium":
                track_speed_level = "slow"
        elif delta_yaw < -np.pi / 16:
            if np.abs(part_1_mean_yaw) > np.abs(part_2_mean_yaw):
                turn_type = "right_lc_merge"
                track_speed_level = ""
            else:
                turn_type = "right_lc"
                track_speed_level = ""
        elif delta_yaw > np.pi / 16:
            if np.abs(part_1_mean_yaw) > np.abs(part_2_mean_yaw):
                turn_type = "left_lc_merge"
                track_speed_level = ""
            else:
                turn_type = "left_lc"
                track_speed_level = ""
        else:
            turn_type = "straight"
        type_key_list = []
        if track_speed_level != "":
            type_key_list.append(track_speed_level)
        if turn_type is not None:
            type_key_list.append(turn_type)
        type_key = "_".join(type_key_list)
    return type_key


ANCHOR_TYPE_CLASSIFY_FUNC = {
    "poor": classical_anc_type_traj_classify,
    "classical": classical_anc_type_traj_classify,
    # TODO (shengzhe): implement the classification function for
    # the expand anchor type set.
}
