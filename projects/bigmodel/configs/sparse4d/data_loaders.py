import math
import os
from functools import partial

import torch
from common import (
    do_experiment,
    enable_temporal_fusion,
    get_dataset_list,
    get_lmdb_dataset_list,
    local_or_remote_debug,
    train_batch_size,
)
from mmcv.parallel import collate

from hat.utils import Config

train_dataset_list = []


try:
    x3c_ds_path = os.path.join(
        os.path.dirname(__file__),
        "../../../pilot/configs/datasets/x3c_multiview_lmdb_datasets.py",
    )
    x3c_datapaths = Config.fromfile(x3c_ds_path).datapaths
    x3c_train_datapaths = (
        x3c_datapaths.multiview_dynamic_3d_detection.train_data_paths
    )
except BaseException:
    x3c_train_datapaths = []

for train_data_path in x3c_train_datapaths:
    lmdb_path = train_data_path["lmdb_path"]
    if do_experiment:
        len_exp = math.ceil(len(lmdb_path) / 4.0)
        lmdb_path = lmdb_path[:len_exp]
    train_dataset_list.extend(
        get_lmdb_dataset_list(lmdb_path, model_setting="x3c", mode="train")
    )

if not do_experiment:
    try:
        x3c_rec_ds_path = os.path.join(
            os.path.dirname(__file__),
            "../../../pilot/configs/datasets/galaxy_x3c_multiview_datasets.py",
        )
        x3c_rec_datapaths = Config.fromfile(x3c_rec_ds_path).datapaths
        x3c_rec_train_datapaths = (
            x3c_rec_datapaths.multiview_dynamic_3d_detection.train_data_paths
        )
    except BaseException:
        x3c_rec_train_datapaths = []

    for train_data_path in x3c_rec_train_datapaths:
        train_dataset_list.extend(
            get_dataset_list(
                train_data_path["rec_path"], model_setting="x3c", mode="train"
            )
        )

    if not enable_temporal_fusion:
        try:
            x3c_rec_ds_path = os.path.join(
                os.path.dirname(__file__),
                "../../../pilot/configs/datasets/x3c_multiview_datasets.py",
            )
            x3c_rec_datapaths = Config.fromfile(x3c_rec_ds_path).datapaths
            x3c_rec_train_datapaths = (
                x3c_rec_datapaths.multiview_dynamic_3d_detection.train_data_paths
            )
        except BaseException:
            x3c_rec_train_datapaths = []

        for train_data_path in x3c_rec_train_datapaths:
            train_dataset_list.extend(
                get_dataset_list(
                    train_data_path["rec_path"],
                    model_setting="x3c",
                    mode="train",
                )
            )

        # as33_ds_path = os.path.join(
        #     os.path.dirname(__file__),
        #     f"../../../pilot/configs/datasets/as33_multiview_datasets.py",
        # )
        # ass33_datapaths = Config.fromfile(as33_ds_path).datapaths
        # ass33_train_datapaths = ass33_datapaths.multiview_dynamic_3d_detection.train_data_paths
        # for train_data_path in ass33_train_datapaths:
        #     train_dataset_list.extend(
        #         get_dataset_list(
        #             train_data_path["rec_path"],
        #             model_setting = "as33",
        #             mode = "train")
        #     )


if local_or_remote_debug:
    train_dataset_list = [train_dataset_list[0], train_dataset_list[-1]]


if enable_temporal_fusion:
    train_dataloader = dict(
        type=torch.utils.data.DataLoader,
        dataset=dict(
            type="ConcatDataset",
            datasets=train_dataset_list,
            with_flag=True,
            accumulate_flag=True,
        ),
        batch_size=1,
        batch_sampler=dict(
            type="DistributedGroupInBatchSampler",
            dataset=dict(
                type="ConcatDataset",
                datasets=train_dataset_list,
                with_flag=True,
                accumulate_flag=True,
            ),
            batch_size=train_batch_size,
            skip_prob=0.8,
            max_skip_num=10,
        ),
        sampler=None,
        collate_fn=partial(collate, samples_per_gpu=train_batch_size),
        # shuffle=True, # shuffle in custom_sampler
        num_workers=4,
        pin_memory=False,
    )
else:
    train_dataloader = dict(
        type=torch.utils.data.DataLoader,
        dataset=dict(type="ConcatDataset", datasets=train_dataset_list),
        batch_size=train_batch_size,
        sampler=dict(type=torch.utils.data.DistributedSampler),
        shuffle=True,
        num_workers=4,
        pin_memory=False,
    )
