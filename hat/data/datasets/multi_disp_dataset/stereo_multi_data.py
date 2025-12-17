#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import logging

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

# Copyright (c) Horizon Robotics. All rights reserved.
import copy
import re
from typing import List, Optional
from torch.utils.data.dataset import Dataset, ConcatDataset
from hat.registry import OBJECT_REGISTRY

from .augment_dataset import *
from .cat_random_datast import *
from .list_dataset import TartanAirDataset, FallingThingsDataset, SIDODDataset, DrivingStereoDataset, ETH3DDataset, KITTIDataset, SintelDataset, SIRSDataset, MiddleburyDataset
logger = logging.getLogger(__name__)
CUR_DIR = osp.abspath(osp.dirname(__file__))

__all__ = [
    "StereoMultiData",
]


@OBJECT_REGISTRY.register
def StereoMultiData(
        dataset_list,
        max_disp,
        test_mode,
        aug_args=None,
        res_args=None,
        norm_args=None,
        crop_args=None,
        debug=False,
        img_open_mode='bgr',
):
    train_sets = []
    for dataset in dataset_list:
        if 'InStereo2K' == dataset: # 1080 * 860
            train_sets.append(
                AugDataset(
                    base_dataset='public/stereo_data/InStereo2K_train.list',
                    img_open_mode=img_open_mode,
                    debug=debug,
                    test_mode=test_mode,
                    max_disp=max_disp,
                    aug_args=aug_args,
                    res_args=res_args if test_mode else [-1, -1, True, 1.0, crop_args[2] / 1080., 1.2],
                    norm_args=norm_args,
                    crop_args=crop_args
                ))
        elif 'Sceneflow' == dataset:    # 960 * 540
            train_sets.append(
                AugDataset(
                    base_dataset='public/stereo_data/sceneflow_train.list',
                    debug=debug,
                    test_mode=test_mode,
                    max_disp=max_disp,
                    aug_args=aug_args,
                    res_args=res_args if test_mode else [-1, -1, True, 1.0, crop_args[2] / 960., 1.4],
                    norm_args=norm_args,
                    crop_args=crop_args
                ))
        elif 'DrivingStereo' == dataset:    # 879 * 400
            train_sets.append(AugDataset(
                    base_dataset=DrivingStereoDataset("public/Public_Datasets/DrivingStereo/driving_stereo_train.txt", img_open_mode=img_open_mode),
                    test_mode=test_mode,
                    max_disp=max_disp,
                    aug_args=aug_args,
                    res_args=res_args,
                    norm_args=norm_args,
                    crop_args=crop_args))
        elif 'ETH3D' == dataset:
            train_sets.append(AugDataset(
                    base_dataset=ETH3DDataset("public/Public_Datasets/ETH3D/ETH3D_train.txt", img_open_mode=img_open_mode),
                    test_mode=test_mode,
                    max_disp=max_disp,
                    aug_args=aug_args,
                    res_args=res_args,
                    norm_args=norm_args,
                    crop_args=crop_args))
        elif 'Middlebury' == dataset:   # 718 * 496
            train_sets.append(AugDataset(
                    base_dataset=MiddleburyDataset("public/Public_Datasets/Middlebury/MiddEval3_train_all.txt", debug=debug, img_open_mode=img_open_mode),
                    test_mode=test_mode,
                    max_disp=max_disp,
                    aug_args=aug_args,
                    res_args=res_args if test_mode else [-1, -1, True, 1.0, crop_args[2] / 718., 1.2],
                    norm_args=norm_args,
                    crop_args=crop_args))
        elif 'KITTI12' == dataset:
            train_sets.append(AugDataset(
                    base_dataset=KITTIDataset("public/Public_Datasets/KITTI2012/kitti12_train194.txt", img_open_mode=img_open_mode),
                    test_mode=test_mode,
                    max_disp=max_disp,
                    aug_args=aug_args,
                    res_args=res_args,
                    norm_args=norm_args,
                    crop_args=crop_args))
        elif 'KITTI15' == dataset:
            train_sets.append(AugDataset(
                    base_dataset=KITTIDataset("public/Public_Datasets/KITTI2015/kitti15_train200.txt", img_open_mode=img_open_mode),
                    test_mode=test_mode,
                    max_disp=max_disp,
                    aug_args=aug_args,
                    res_args=res_args,
                    norm_args=norm_args,
                    crop_args=crop_args))
        elif 'Sintel' == dataset:
            train_sets.append(AugDataset(
                    base_dataset=SintelDataset("public/Public_Datasets/Sintel/training", debug=debug, img_open_mode=img_open_mode),
                    test_mode=test_mode,
                    max_disp=max_disp,
                    aug_args=aug_args,
                    res_args=res_args,
                    norm_args=norm_args,
                    crop_args=crop_args))
        elif 'IRS' == dataset:  # 960 * 540
            train_sets.append(AugDataset(
                    base_dataset=SIRSDataset("public/Public_Datasets/IRS/IRSDataset_TRAIN.list", debug=debug, img_open_mode=img_open_mode),
                    test_mode=test_mode,
                    max_disp=max_disp,
                    aug_args=aug_args,
                    res_args=res_args if test_mode else [-1, -1, True, 1.0, crop_args[2] / 960., 1.2],
                    norm_args=norm_args,
                    crop_args=crop_args))
        elif 'FallingThings' == dataset:        # 960 * 540
            data_root = "NVIDIA/FallingThings/fat/mixed/"
            for scene in os.listdir(data_root):
                dis_path = os.path.join(data_root, scene)
                if os.path.isfile(dis_path):
                    continue
                if "_disp_gt" not in scene:
                    continue
                if len(os.listdir(dis_path)) == 0:
                    continue
                train_sets.append(AugDataset(
                        base_dataset=FallingThingsDataset(dis_path, debug=debug, img_open_mode=img_open_mode),
                        test_mode=test_mode,
                        max_disp=max_disp,
                        aug_args=aug_args,
                        res_args=res_args if test_mode else [-1, -1, True, 1.0, crop_args[2] / 960., 1.2],
                        norm_args=norm_args,
                        crop_args=crop_args))
        elif 'SIDODDataset' == dataset: # 960 * 540
            data_root = "NVIDIA/SIDOD/mixed_distractor/"
            for scene in os.listdir(data_root):
                dis_path = os.path.join(data_root, scene)
                if os.path.isfile(dis_path):
                    continue
                if "_disp_gt" not in scene:
                    continue
                if len(os.listdir(dis_path)) == 0:
                    continue
                # data_set_list.append(scene_root)
                train_sets.append(AugDataset(
                        base_dataset=SIDODDataset(dis_path, debug=debug, img_open_mode=img_open_mode),
                        test_mode=test_mode,
                        max_disp=max_disp,
                        aug_args=aug_args,
                        res_args=res_args if test_mode else [-1, -1, True, 1.0, crop_args[2] / 960., 1.2],
                        norm_args=norm_args,
                        crop_args=crop_args))
        elif 'TartanAir' == dataset:        # 640, 480
            data_root = "TartanAir/"
            data_info = 0
            for scene in os.listdir(data_root):
                tmp_scene_root = os.path.join(data_root, scene)
                if os.path.isfile(tmp_scene_root):
                    continue
                for level in os.listdir(tmp_scene_root):
                    tmp_scene_level_root = os.path.join(tmp_scene_root, level)
                    if os.path.isfile(tmp_scene_level_root):
                        continue
                    for target in os.listdir(tmp_scene_level_root):
                        target_path = os.path.join(tmp_scene_level_root, target)
                        if os.path.isfile(target_path):
                            continue
                        disp_gt_path = os.path.join(target_path, "disp_gt")
                        if not os.path.exists(disp_gt_path):
                            continue
                        if len(os.listdir(disp_gt_path)) == 0:
                            continue
                        dataset = TartanAirDataset(target_path, debug=debug, img_open_mode=img_open_mode)
                        data_info += len(dataset)
                        train_sets.append(AugDataset(
                                base_dataset=dataset,
                                test_mode=test_mode,
                                max_disp=max_disp,
                                aug_args=aug_args,
                                res_args=res_args if test_mode else [-1, -1, True, 1.0, crop_args[2] / 640., 1.2],
                                norm_args=norm_args,
                                crop_args=crop_args))
            logger.info("TartanAir total sample: %d" % data_info)
        else:
            raise NotImplementedError
    return CatRandomDataset(train_sets)