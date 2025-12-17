import os

import torch
from common import val_transforms

from hat.core.task_sampler import TaskSampler
from hat.utils.bucket import url_to_local_path

pilot_2d_list = [
    # 6026506,
    # 6026166,
    # 6026181,
    # 6026233,
    # 6026482,
    # 6026187,
    # 6026179,
    # 6026175,
    # 6026174,
    # 6026493,
    6028887,
]
task_name_list = [
    "vehicle",
    # "vehicle_rear",
    # "vehicle_full_traffic_sign",
    # "vehicle_full_traffic_sign",
    # "traffic_light",
    # "person_cyclist",
    # "parsing",
    # "lane",
    # "person",
]

local_datapath = url_to_local_path(
    "dmpv2://auto_eval/adas_eval/eval_platform/fs/"
)
full_path_list = [
    os.path.join(local_datapath, str(dataset_id), "datasets/")
    for dataset_id in pilot_2d_list
]

datapaths = dict(zip(task_name_list, full_path_list))

val_task_config = dict()
val_dataloaders = dict()
for task_name, data_path in datapaths.items():
    val_task_config[task_name] = dict(sampling_factor=1)
    val_data_loader = dict(
        type=torch.utils.data.DataLoader,
        dataset=dict(
            type="ModelEvalRawDataset",
            data_path=data_path,
            to_rgb=True,
            buf_only=True,
            return_orig_img=True,
            transforms=val_transforms,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    val_dataloaders[task_name] = val_data_loader

val_task_sampler = TaskSampler(
    task_config=val_task_config, method="sample_all"
)

val_data_loader = dict(
    type="MultitaskLoader",
    loaders=val_dataloaders,
    task_sampler=val_task_sampler,
    mode="validation",
    return_task=True,
    custom_length=None,
)
