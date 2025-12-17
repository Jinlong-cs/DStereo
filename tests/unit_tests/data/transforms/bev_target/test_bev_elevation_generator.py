# -*- coding: utf-8 -*-
# Copyright (c) Horizon Robotics. All rights reserved.

import numpy as np
import torch

from hat.data.transforms.bev_elevation import (
    ANCBEVFreespaceGenerator,
    ANCBEVVisMaskGenerator,
)


def generate_fake_data():
    data = {}
    bev_elevation_chanel1 = np.random.randint(
        low=-25, high=200, size=(512, 512, 1)
    )
    bev_elevation_chanel2 = np.random.randint(
        low=0, high=19, size=(512, 512, 1)
    )
    bev_elevation_chanel3 = np.random.randint(
        low=0, high=38, size=(512, 512, 1)
    )
    data["gt_bev_elevation_raw"] = np.concatenate(
        (bev_elevation_chanel1, bev_elevation_chanel2, bev_elevation_chanel3),
        axis=2,
    ).astype(np.uint8)
    data["gt_bev_elevation_vismask_raw"] = np.random.randint(
        low=0, high=2, size=(512, 512, 3)
    ).astype(np.float)
    bev_freespace_chanel1 = np.random.randint(
        low=0, high=4, size=(512, 512, 1)
    )
    bev_freespace_chanel2 = np.random.randint(
        low=0, high=3, size=(512, 512, 1)
    )
    bev_freespace_chanel3 = np.random.randint(
        low=0, high=10, size=(512, 512, 1)
    )
    data["gt_bev_freespace_raw"] = np.concatenate(
        (bev_freespace_chanel1, bev_freespace_chanel2, bev_freespace_chanel3),
        axis=2,
    )
    return data


def test_bev_elevation_vismask_generator():
    torch.manual_seed(0)
    data = generate_fake_data()
    global_dilate_cfg = {
        "use_dilate": True,
        "kernel": 15,
        "iterations": 1,
    }
    raw_gt_res_vcsrange_cfg = {
        (512, 512): (-30.0, -51.2, 72.4, 51.2),  # (bottom, right, top, left)
    }
    bev_elevation_mask_generator = ANCBEVVisMaskGenerator(
        target_size=(512, 512),
        ignore_index=255,
        vcs_range=(-30.0, -51.2, 72.4, 51.2),
        raw_gt_res_vcsrange_cfg=raw_gt_res_vcsrange_cfg,
        global_dilate_cfg=global_dilate_cfg,
    )
    data = bev_elevation_mask_generator(data)
    assert data["gt_bev_elevation_vismask"]["vismask"].shape == (512, 512)
    assert data["gt_bev_elevation_vismask"]["vismask"].min() >= 0
    assert data["gt_bev_elevation_vismask"]["vismask"].max() <= 1


def test_bev_freespace_generator():
    torch.manual_seed(0)
    data = generate_fake_data()
    vcs_range = (-12.80, -12.80, 25.60, 12.80)
    smallobj_dilate_cfg = {
        "use_dilate": True,
        "kernel": 5,
        "autolabel_area_range": (4, 100),  # measued in pixel num
        "manuallabel_area_range": (1, 100),
    }
    raw_gt_res_vcsrange_cfg = {
        (512, 512): (-30.0, -51.2, 72.4, 51.2),  # (bottom, right, top, left)
    }
    lidar_seg_remap_dict = {
        0: 0,
        1: 1,
        2: 1,
        255: 255,
    }
    freespace_mannual_remap_dict = {
        0: 0,
        1: 1,
        2: 1,
        3: 1,
        255: 255,
    }
    freespace_auto_remap_dict = {
        0: 1,
        1: 0,
        2: 1,
        255: 255,
    }
    bev_freespace_generator = ANCBEVFreespaceGenerator(
        vcs_range=vcs_range,
        raw_gt_res_vcsrange_cfg=raw_gt_res_vcsrange_cfg,
        target_size=(512, 512),
        lidar_seg_remap_dict=lidar_seg_remap_dict,
        freespace_mannual_remap_dict=freespace_mannual_remap_dict,
        freespace_auto_remap_dict=freespace_auto_remap_dict,
        ele_freespace_mannual_remap_dict=None,
        ele_freespace_auto_remap_dict=None,
        occupancy_output_freespace=True,
        occupancy_output_elevation=False,
        smallobj_dilate_cfg=smallobj_dilate_cfg,
        mannualabel_use_lidar_seg=False,
        ego_region_range=(-1.0, -1.0, 3.7, 1.0),
        median_blur_seg_height=True,
    )
    data = bev_freespace_generator(data)
    assert data["gt_bev_freespace"]["gt_bev_freespace"].shape == (512, 512)
