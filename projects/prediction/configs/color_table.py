# Copyright (c) Horizon Robotics. All rights reserved.

import numpy as np


def gen_color_table(label_colors):
    all_seg_labels = list(label_colors.keys())
    color_table = {}
    rand_table_before = (np.random.randint(-128, 127, size=128) / 128).tolist()
    rand_table_after = (
        np.random.randint(-128, 127, size=128 - len(all_seg_labels)) / 128
    ).tolist()

    color_table["r_val"] = tuple(
        rand_table_before
        + [(label_colors[label][0] - 128) / 128 for label in all_seg_labels]
        + rand_table_after
    )
    color_table["g_val"] = tuple(
        rand_table_before
        + [(label_colors[label][1] - 128) / 128 for label in all_seg_labels]
        + rand_table_after
    )
    color_table["b_val"] = tuple(
        rand_table_before
        + [(label_colors[label][2] - 128) / 128 for label in all_seg_labels]
        + rand_table_after
    )
    return color_table


def gen_drivable_map_color_table():
    label_colors = {
        "unknown": [0, 127, 0],
        "straight_only": [0, 255, 0],
        "turnleft_only": [255, 0, 0],
        "turnleft_and_straight": [255, 255, 0],
        "turnright_only": [0, 0, 255],
        "turnright_and_straight": [0, 255, 255],
        "turnleft_and_right": [255, 0, 255],
        "curve": [127, 127, 127],
        "free": [255, 255, 255],
        "offroad": [0, 0, 0],
    }
    return gen_color_table(label_colors)


def gen_roadmap_with_vl_color_table():
    # Get color table.
    label_colors = {
        "others": [0, 0, 0],
        "roadedge": [0, 0, 255],
        "roadarrow": [47, 79, 79],
        "solid_lanes": [200, 200, 200],
        "stopline": [192, 0, 64],
        "crosswalk": [255, 127, 80],
        "sections": [0, 0, 0],
        "junctions": [0, 0, 0],
        "virtuallanes": [127, 127, 127],
    }
    return gen_color_table(label_colors)
