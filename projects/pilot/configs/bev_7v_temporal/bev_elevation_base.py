import os

import numpy as np
import torch

from projects.pilot.configs.bev_7v_temporal.base import (
    get_roi_resize_cfg,
    reformat_compile_vcs_range,
)
from projects.pilot.configs.bev_7v_temporal.common import (
    bevfusion_output_size,
    save_prefix,
    spatial_resolution,
    vcs_origin_coord,
    vcs_plane_heights,
    vcs_range,
)

# -------------------------- TASK Common Cfg ---------------------------
task_in_stride = 2
roi_vcs_range = (-30.4, -22.4, 100.8, 22.4)
task_out_size = (656, 224)  # (h, w)
task_out_resolution = (
    abs(roi_vcs_range[2] - roi_vcs_range[0]) / task_out_size[0],
    abs(roi_vcs_range[3] - roi_vcs_range[1]) / task_out_size[1],
)  # (h, w)
roi_vcs_origin_coord = [
    int(
        roi_vcs_range[2]
        / (roi_vcs_range[2] - roi_vcs_range[0])
        * task_out_size[0]
    ),
    int(
        roi_vcs_range[3]
        / (roi_vcs_range[3] - roi_vcs_range[1])
        * task_out_size[1]
    ),
]

roi_resize_cfg = get_roi_resize_cfg(
    input_size=bevfusion_output_size,
    in_stride=task_in_stride,
    output_size=task_out_size,
    ori_vcs_range=vcs_range,
    roi_vcs_range=roi_vcs_range,
)
roi_resize = dict(
    type="RoiCropResize",
    in_strides=[2, 4, 8, 16, 32],
    target_stride=roi_resize_cfg["in_stride"],
    output_size=roi_resize_cfg["output_size"],
    roi_box=roi_resize_cfg["roi_box"],
    node_name="bev_evevation_roi_resize",
)

stage2_stride2channels = {
    2: 48,
    4: 48,
    8: 96,
    16: 192,
    32: 192,
    64: 192,
    128: 192,
    256: 192,
}

common_desc = dict(
    vcs_origin_coord=vcs_origin_coord,
    spatial_resolution=[0.2, 0.2],
    bev_stage2_input_resolution=spatial_resolution,
    bev_stage2_output_resolution=task_out_resolution,
    visible_range=reformat_compile_vcs_range(roi_vcs_range),
    warp_offset_range=reformat_compile_vcs_range(vcs_range),
    vcs_plane_heights=vcs_plane_heights,
)

# -------------- bev elevation remap dict ---------------------------------
# bev elevation class setting
# 45 mean class num before remap.
# 7 and 9 mean class num after remap.
# REG means use regression model(output channel is 1).
height_to_bin_config = {
    "min_height_value": -0.4,
    "max_height_value": 4,
    "bins_num": 45,
    "scale": 10,  # If not multiplied by 10, there will be precision loss # noqa
}

valid_bev_elevation_types = [
    "HDE_45_7",
    "HDE_45_9",
    "HDE_REG",
]

stage2_elevation_class_dict = {
    "HDE_45_7": 7,
    "HDE_45_9": 9,
    "HDE_REG": 1,
}

stage2_elevation_remap_dict = {
    "HDE_45_7": {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 1,
        7: 2,
        8: 3,
        9: 3,
        10: 4,
        11: 4,
        12: 4,
        13: 4,
        14: 4,
        15: 5,
        16: 5,
        17: 5,
        18: 5,
        19: 5,
        20: 5,
        21: 5,
        22: 5,
        23: 5,
        24: 5,
        25: 6,
        26: 6,
        27: 6,
        28: 6,
        29: 6,
        30: 6,
        31: 6,
        32: 6,
        33: 6,
        34: 6,
        35: 6,
        36: 6,
        37: 6,
        38: 6,
        39: 6,
        40: 6,
        41: 6,
        42: 6,
        43: 6,
        44: 6,
        45: 255,
        255: 255,
    },
    "HDE_45_9": {
        0: 0,
        1: 0,
        2: 1,
        3: 1,
        4: 2,
        5: 2,
        6: 3,
        7: 4,
        8: 5,
        9: 5,
        10: 6,
        11: 6,
        12: 6,
        13: 6,
        14: 6,
        15: 7,
        16: 7,
        17: 7,
        18: 7,
        19: 7,
        20: 7,
        21: 7,
        22: 7,
        23: 7,
        24: 7,
        25: 8,
        26: 8,
        27: 8,
        28: 8,
        29: 8,
        30: 8,
        31: 8,
        32: 8,
        33: 8,
        34: 8,
        35: 8,
        36: 8,
        37: 8,
        38: 8,
        39: 8,
        40: 8,
        41: 8,
        42: 8,
        43: 8,
        44: 8,
        45: 255,
        255: 255,
    },
    "HDE_REG": {
        -5: 0,
        -4: 0,
        -3: 1,
        -2: 1,
        -1: 2,
        0: 2,
        1: 3,
        2: 4,
        3: 5,
        4: 5,
        5: 6,
        6: 6,
        7: 6,
        8: 6,
        9: 6,
        10: 7,
        11: 7,
        12: 7,
        13: 7,
        14: 7,
        15: 7,
        16: 7,
        17: 7,
        18: 7,
        19: 7,
        20: 8,
        21: 8,
        22: 8,
        23: 8,
        24: 8,
        25: 8,
        26: 8,
        27: 8,
        28: 8,
        29: 8,
        30: 8,
        31: 8,
        32: 8,
        33: 8,
        34: 8,
        35: 8,
        36: 8,
        37: 8,
        38: 8,
        39: 8,
        40: 8,
        255: 255,
    },
}

stage2_elevation_names_dict = {
    "HDE_45_7": [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
    ],
    "HDE_45_9": [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
    ],
    "HDE_REG": [],
}

# remap dict to label dynamic object from lidar seg
lidar_seg_remap_dict = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
    10: 0,
    11: 0,
    12: 0,
    13: 0,
    14: 0,
    15: 0,
    16: 0,
    17: 0,
    18: 0,
    19: 0,
    20: 1,
    21: 1,
    22: 1,
    23: 1,
    24: 1,
    25: 1,
    26: 1,
    27: 1,
    28: 1,
    29: 1,
    30: 1,
    31: 1,
    32: 1,
    33: 1,
    34: 1,
    35: 1,
    36: 1,
    37: 1,
    38: 1,
    39: 1,
    40: 1,
    41: 1,  # 锥桶
    42: 1,  # 地锁
    255: 1,
}

# remap mannual freespace annotation to label freespace_yes and freespace_no
freespace_mannual_remap_dict = {
    0: 0,
    1: 1,
    2: 1,
    3: 1,
    255: 255,
}

# remap to label freespace_yes and freespace_no from auto
# label data according to lidar seg, for lidar seg label and category
# name details, please refer to
# http://wiki.hobot.cc/pages/viewpage.action?pageId=248420446
freespace_auto_remap_dict = {
    0: 1,
    1: 0,  # 路面
    2: 1,
    3: 0,  #
    4: 0,  #
    5: 0,  #
    6: 0,  #
    7: 1,
    8: 0,  # 减速坎
    9: 1,
    10: 1,
    11: 1,
    12: 1,
    13: 1,
    14: 1,
    15: 1,
    16: 1,
    17: 1,
    18: 1,
    19: 1,
    20: 1,
    21: 1,
    22: 1,
    23: 1,
    24: 1,
    25: 1,
    26: 1,
    27: 1,
    28: 1,
    29: 1,
    30: 1,
    31: 1,
    32: 1,
    33: 1,
    34: 1,
    35: 1,
    36: 1,
    37: 1,
    38: 1,
    39: 1,
    40: 1,
    41: 1,  # 锥桶
    42: 1,  # 地锁
    255: 255,
}

# remap freespace mannual anno for elevation
ele_freespace_mannual_remap_dict = {
    0: 1,
    1: 0,
    2: 1,
    3: 1,
    255: 0,
}

# remap freespace auto label from lidar seg anno to get
# freespace region and obstacles in freespace region
# for autolabel elevation
ele_freespace_auto_remap_dict = {
    0: 0,
    1: 1,  # 路面
    2: 0,  # 人行道
    3: 1,  # 车道线
    4: 1,  # 停车线
    5: 1,  # 人行横道线
    6: 1,  # 车道箭头
    7: 1,  # 路沿
    8: 1,  # 减速坎
    9: 0,
    10: 0,
    11: 0,
    12: 0,
    13: 1,  # 隔离带
    14: 1,  # 围栏
    15: 0,
    16: 0,
    17: 0,
    18: 0,
    19: 1,  # 其他障碍物
    20: 1,  # >=20主要各种车
    21: 1,
    22: 1,
    23: 1,
    24: 1,
    25: 1,
    26: 1,
    27: 1,
    28: 1,
    29: 1,
    30: 1,
    31: 1,
    32: 1,
    33: 0,  # 花坛
    34: 0,  # 桥
    35: 1,
    36: 1,
    37: 1,
    38: 0,
    39: 0,
    40: 0,
    41: 1,  # 锥桶
    42: 1,  # 地锁
    255: 0,
}

# Note all raw gt saved with 0.1m/pixel spatial resolution
# config fot task raw resolution-vcs_range pairs,
# resolution as key, corresponding vcs range as values.
# resolution in (h, w) order, vcs range in (bottom, right, top, left)
# different saved resolution corresponding to data used
# in different period.
raw_gt_res_vcsrange_cfg = {
    (656, 384): (-30.4, -38.4, 100.8, 38.4),
    (512, 512): (-30.0, -51.2, 72.4, 51.2),  # (bottom, right, top, left)
    (1024, 1024): (-30.0, -51.2, 72.4, 51.2),
    (2048, 1536): (-51.2, -76.8, 153.6, 76.8),
    (2048, 1600): (-51.2, -80.0, 153.6, 80.0),
}


def get_rle_postprocess(data_name, use_rle_pad, padding, dtype=torch.int8):
    """Get RLE(running length encoding) postprocess.

    Args:
        data_name (str): input name to apply RLE postprocess
        use_rle_pad (bool): whether use pad before RLE.
        padding (tuple): (left, right, top, bottom) order.
        dtype: Defaults to torch.int8.
    """
    rle_postprocess = []
    rle_postprocess.append(
        dict(
            type="RLEPostprocess",
            data_name=data_name,
            dtype=dtype,
        )
    )
    # pad need place before RLE
    if use_rle_pad:
        pad_postprocess = dict(
            type="PadPostprocess",
            data_name=data_name,
            padding=padding,
        )
        rle_postprocess.insert(0, pad_postprocess)
    return rle_postprocess


def get_rle_padding(size, multiplier_base=256):
    """Get padding size for width align to least multiple multiplier_base.

    Args:
        size (tuple): ordered (h, w).
        multiplier_base (int, optional): align width to the least mutiple
            of multipiler_base.
    """
    w = size[1]
    align_w = w
    if w % multiplier_base != 0:
        align_w = (w // multiplier_base + 1) * multiplier_base
    padding = [0, align_w - w, 0, 0]  # (left, right, top, bottom)
    use_rle_pad = align_w > w
    return use_rle_pad, padding, align_w


# -------------------------- PACK_INFER_SAVE --------------------------
save_freespace_pack_infer = dict(
    type="SaveSegmentationConsistencyResult",
    output_dir=os.path.join(save_prefix, "inference", "bev_freespace"),
    task_list=["bev_freespace"],
    prefix="OutputModule_predict",
)

save_vismask_pack_infer = dict(
    type="SaveSegmentationConsistencyResult",
    output_dir=os.path.join(save_prefix, "inference", "bev_vismask"),
    task_list=["bev_vismask"],
    prefix="OutputModule_predict",
)

# ------------------------- PACK INFER VIS ------------------------------
VIS_COLORS = np.zeros((2, 3), dtype="uint8")
VIS_COLORS[0, :] = [255, 144, 30]  # freespace_yes
VIS_COLORS[1, :] = [0, 0, 255]  # freespace_no
