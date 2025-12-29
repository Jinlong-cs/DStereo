#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import logging
import json
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

# Copyright (c) Horizon Robotics. All rights reserved.
import copy
import re
from typing import List, Optional
from torch.utils.data.dataset import Dataset, ConcatDataset
from hat.registry import OBJECT_REGISTRY
from PIL import Image

from .augment_dataset import *
from .cat_random_datast import *
from .list_dataset import (
    TartanAirDataset,
    FallingThingsDataset,
    SIDODDataset,
    DrivingStereoDataset,
    ETH3DDataset,
    KITTIDataset,
    SintelDataset,
    SIRSDataset,
    MiddleburyDataset,
    DStereoDataset
)
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
    img_open_mode="bgr",
):
    train_sets = []
    for dataset in dataset_list:
        print("=> add dataset: ", dataset)
        if "InStereo2K" == dataset: 
            train_sets.append(
                AugDataset(
                    base_dataset="public/stereo_data/InStereo2K_train.list",
                    img_open_mode=img_open_mode,
                    debug=debug,
                    test_mode=test_mode,
                    max_disp=max_disp,
                    aug_args=aug_args,
                    res_args=(
                        res_args
                        if test_mode
                        else [-1, -1, True, 1.0, crop_args[2] / 1080.0, 1.2]
                    ),
                    norm_args=norm_args,
                    crop_args=crop_args,
                )
            )
        elif "Sceneflow" == dataset:  
            train_sets.append(
                AugDataset(
                    base_dataset="/mnt/sznas/yzf/stereo_data/SceneFlow/FlyingThings3D/train_list.txt",
                    debug=debug,
                    test_mode=test_mode,
                    max_disp=max_disp,
                    aug_args=aug_args,
                    res_args=(
                        res_args
                        if test_mode
                        else [-1, -1, True, 1.0, crop_args[2] / 960.0, 1.4]
                    ),
                    norm_args=norm_args,
                    crop_args=crop_args,
                )
            )
        elif "DrivingStereo" == dataset:  
            train_sets.append(
                AugDataset(
                    base_dataset=DrivingStereoDataset(
                        "public/Public_Datasets/DrivingStereo/driving_stereo_train.txt",
                        img_open_mode=img_open_mode,
                    ),
                    test_mode=test_mode,
                    max_disp=max_disp,
                    aug_args=aug_args,
                    res_args=res_args,
                    norm_args=norm_args,
                    crop_args=crop_args,
                )
            )
        elif "ETH3D" == dataset:
            train_sets.append(
                AugDataset(
                    base_dataset=ETH3DDataset(
                        "public/Public_Datasets/ETH3D/ETH3D_train.txt",
                        img_open_mode=img_open_mode,
                    ),
                    test_mode=test_mode,
                    max_disp=max_disp,
                    aug_args=aug_args,
                    res_args=res_args,
                    norm_args=norm_args,
                    crop_args=crop_args,
                )
            )
        elif "Middlebury" == dataset:  
            train_sets.append(
                AugDataset(
                    base_dataset=MiddleburyDataset(
                        "public/Public_Datasets/Middlebury/MiddEval3_train_all.txt",
                        debug=debug,
                        img_open_mode=img_open_mode,
                    ),
                    test_mode=test_mode,
                    max_disp=max_disp,
                    aug_args=aug_args,
                    res_args=(
                        res_args
                        if test_mode
                        else [-1, -1, True, 1.0, crop_args[2] / 718.0, 1.2]
                    ),
                    norm_args=norm_args,
                    crop_args=crop_args,
                )
            )
        elif "KITTI12" == dataset:
            train_sets.append(
                AugDataset(
                    base_dataset=KITTIDataset(
                        "public/Public_Datasets/KITTI2012/kitti12_train194.txt",
                        img_open_mode=img_open_mode,
                    ),
                    test_mode=test_mode,
                    max_disp=max_disp,
                    aug_args=aug_args,
                    res_args=res_args,
                    norm_args=norm_args,
                    crop_args=crop_args,
                )
            )
        elif "KITTI15" == dataset:
            train_sets.append(
                AugDataset(
                    base_dataset=KITTIDataset(
                        "public/Public_Datasets/KITTI2015/kitti15_train200.txt",
                        img_open_mode=img_open_mode,
                    ),
                    test_mode=test_mode,
                    max_disp=max_disp,
                    aug_args=aug_args,
                    res_args=res_args,
                    norm_args=norm_args,
                    crop_args=crop_args,
                )
            )
        elif "Sintel" == dataset:
            train_sets.append(
                AugDataset(
                    base_dataset=SintelDataset(
                        "public/Public_Datasets/Sintel/training",
                        debug=debug,
                        img_open_mode=img_open_mode,
                    ),
                    test_mode=test_mode,
                    max_disp=max_disp,
                    aug_args=aug_args,
                    res_args=res_args,
                    norm_args=norm_args,
                    crop_args=crop_args,
                )
            )
        elif "IRS" == dataset:  
            train_sets.append(
                AugDataset(
                    base_dataset=SIRSDataset(
                        "public/Public_Datasets/IRS/IRSDataset_TRAIN.list",
                        debug=debug,
                        img_open_mode=img_open_mode,
                    ),
                    test_mode=test_mode,
                    max_disp=max_disp,
                    aug_args=aug_args,
                    res_args=(
                        res_args
                        if test_mode
                        else [-1, -1, True, 1.0, crop_args[2] / 960.0, 1.2]
                    ),
                    norm_args=norm_args,
                    crop_args=crop_args,
                )
            )
        elif "FallingThings" == dataset:  
            data_root = "NVIDIA/FallingThings/fat/mixed/"
            for scene in os.listdir(data_root):
                dis_path = os.path.join(data_root, scene)
                if os.path.isfile(dis_path):
                    continue
                if "_disp_gt" not in scene:
                    continue
                if len(os.listdir(dis_path)) == 0:
                    continue
                train_sets.append(
                    AugDataset(
                        base_dataset=FallingThingsDataset(
                            dis_path, debug=debug, img_open_mode=img_open_mode
                        ),
                        test_mode=test_mode,
                        max_disp=max_disp,
                        aug_args=aug_args,
                        res_args=(
                            res_args
                            if test_mode
                            else [-1, -1, True, 1.0, crop_args[2] / 960.0, 1.2]
                        ),
                        norm_args=norm_args,
                        crop_args=crop_args,
                    )
                )
        elif "SIDODDataset" == dataset:  
            data_root = "NVIDIA/SIDOD/mixed_distractor/"
            for scene in os.listdir(data_root):
                dis_path = os.path.join(data_root, scene)
                if os.path.isfile(dis_path):
                    continue
                if "_disp_gt" not in scene:
                    continue
                if len(os.listdir(dis_path)) == 0:
                    continue
                train_sets.append(
                    AugDataset(
                        base_dataset=SIDODDataset(
                            dis_path, debug=debug, img_open_mode=img_open_mode
                        ),
                        test_mode=test_mode,
                        max_disp=max_disp,
                        aug_args=aug_args,
                        res_args=(
                            res_args
                            if test_mode
                            else [-1, -1, True, 1.0, crop_args[2] / 960.0, 1.2]
                        ),
                        norm_args=norm_args,
                        crop_args=crop_args,
                    )
                )
        elif "TartanAir" == dataset:  
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
                        dataset = TartanAirDataset(
                            target_path, debug=debug, img_open_mode=img_open_mode
                        )
                        data_info += len(dataset)
                        train_sets.append(
                            AugDataset(
                                base_dataset=dataset,
                                test_mode=test_mode,
                                max_disp=max_disp,
                                aug_args=aug_args,
                                res_args=(
                                    res_args
                                    if test_mode
                                    else [-1, -1, True, 1.0, crop_args[2] / 640.0, 1.2]
                                ),
                                norm_args=norm_args,
                                crop_args=crop_args,
                            )
                        )
            logger.info("TartanAir total sample: %d" % data_info)
        elif 'DStereoDataset' == dataset:
            # 读取input.json配置文件
            input_path = "/workspace/input/task.json"
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            with open(input_path, "r") as f:
                task_config = json.load(f)
                
            train_config = task_config["train"]
            input_path = train_config["data"]["root"]
            datasets_path = train_config["data"]["datasets_path"] #数据集目录
            datasets_list_dir = input_path
            datasets_list = train_config["data"]["val_datasets_list"] if test_mode else train_config["data"]["train_datasets_list"]
            
            # 遍历dataset中的数据集
            for dataset_list_path in datasets_list:
                
                # 获取数据集列表txt路径
                datasets_list_path = os.path.join(datasets_list_dir, dataset_list_path)
                
                #获取数据集左目、右目、视差的全部绝对路径
                with open(datasets_list_path, 'r') as f:
                    file_list = f.readlines()
                    # print(self.file_list)
                    file_list = [[os.path.join(datasets_path, path) for path in line.strip().split(' ')] for line in file_list if line.strip()]
                
                # 加载数据集首个图像，获取数据集的宽高
                width, height = 960,540
                with Image.open(file_list[0][0]) as img:
                    width, height = img.size  # 获取宽高
                    print(f"数据集图像宽度: {width}, 数据集图像高度: {height}")
                
                train_sets.append(AugDataset(
                    base_dataset=DStereoDataset(
                        file_list=file_list,
                        dataset_name=dataset_list_path.split("_")[0],
                        debug=debug,
                        img_open_mode=img_open_mode
                    ),
                    test_mode=test_mode, 
                    max_disp=max_disp,
                    aug_args=aug_args,
                    res_args=res_args if test_mode else [-1, -1, True, 1.0, crop_args[2] / width, 1.2],
                    norm_args=norm_args,
                    crop_args=crop_args
                ))
        else:
            raise NotImplementedError
    return CatRandomDataset(train_sets)
