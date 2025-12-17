import copy
import os

import torch

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.data.collates.collates import collate_e2e_dynamic
from hat.data.samplers.dist_clip_group_sampler import (
    ANCDistributedClipValSampler,
    ANCDistributedGroupClipSampler,
)
from hat.utils.seed import worker_reset_seed
from projects.pilot.configs.bev_7v_temporal.base import (
    convert_to_temporal_split_dataloader,
    get_data_dict,
    remove_none,
)
from projects.pilot.configs.bev_7v_temporal.common import (
    common_transforms,
    log_freq,
    train_batch_size_per_gpu,
    train_num_frames_per_iter,
    train_num_workers,
    use_split_dataloader,
    val_num_frames_per_iter,
    val_num_workers,
    views_domain2nums,
)

VALDATA_CLIP_LEN = 20

# --------------------------BASE --------------------------
cfg_dir = os.path.dirname(__file__)
task_name = "e2e_dynamic"
# e2e_frames_setting是端到端任务单独控制时序配置和bev_base中的temporal_frames_setting不共用
num_frames_per_clip = 12
num_frames_for_pred = 60

feature_cache_keys = []
batch_length_for_feature_cache = None
e2e_out_trajectory = True
val_batch_size_per_gpu = 1  # Only support bs=1 for val
# --------------------------BEV BASE --------------------------
use_multi_head = True
load_origin_imgs = False
dataloader_seed = 0
# --------------------------CCONFIG SETTING ---------------------
sequence_data_idxs = {
    "train": list(range(train_num_frames_per_iter)),
    "val": list(range(val_num_frames_per_iter)),
}

url = os.path.join(cfg_dir, "../datasets/e2e/e2e_dynamic_train_version.yaml")
train_data_version_dict = get_data_dict(url)

url = os.path.join(cfg_dir, "../datasets/e2e/e2e_dynamic_val_version.yaml")
val_data_version_dict = get_data_dict(url)

e2e_dynamic_common_transforms = copy.deepcopy(common_transforms)
e2e_dynamic_common_transforms["ANCCollect3DV"].update(
    {"num_frames_per_iter": train_num_frames_per_iter}
)

load_data_types = [
    "timestamp",
    "pack_dir",
    "e2e_dynamic_anno",
    "origin_imgs" if load_origin_imgs else None,
]

train_collect_3dv = e2e_dynamic_common_transforms["ANCCollect3DV"]
train_collect_3dv["img_idxs"] = sequence_data_idxs["train"]
train_collect_3dv["pose_idxs"] = sequence_data_idxs["train"]
train_collect_3dv["load_data_types"] = load_data_types

val_collect_3dv = copy.deepcopy(train_collect_3dv)
val_collect_3dv["img_idxs"] = sequence_data_idxs["val"]
val_collect_3dv["pose_idxs"] = sequence_data_idxs["val"]
val_collect_3dv["num_frames_per_iter"] = val_num_frames_per_iter

prepare_temporal_data = dict(
    # 打开时序的时候对输出图像进行反序。
    type="ANCPrepareTempoDataE2EDynamic",
    length_of_clip=num_frames_per_clip,
    num_frames_per_iter=train_num_frames_per_iter,
    reverse_imgs_order=False,
    views_domain2nums=views_domain2nums,
)
set_temporal_clear_flag = dict(
    type="ANCSetTemporalClearFlag",
    clr_mode="clip",
)
set_temporal_return_latest_flag = dict(
    type="AddKeys",
    kv={"return_latest_flag": False},
)
set_fake_dataset_flag = dict(type="AddKeys", kv={"fake_dataset_flag": False})
val_set_temporal_clear_flag = copy.deepcopy(set_temporal_clear_flag)
val_set_temporal_clear_flag["clr_mode"] = "pack"
val_prepare_temporal_data = copy.deepcopy(prepare_temporal_data)
val_prepare_temporal_data["length_of_clip"] = VALDATA_CLIP_LEN
val_prepare_temporal_data["num_frames_per_iter"] = val_num_frames_per_iter


def get_e2e_transforms(
    e2e_train_target, e2e_val_target
):  # train and val's collect_3dv is different
    train_bev_3d_transforms = [
        train_collect_3dv,
        e2e_dynamic_common_transforms["ANCResize3DV"],
        e2e_dynamic_common_transforms["ANCCrop3DV"],
        None,
        e2e_dynamic_common_transforms["ANCToTensor3DV"],
        e2e_train_target,
        prepare_temporal_data,
        set_temporal_clear_flag,
        set_temporal_return_latest_flag,
        dict(type="AddKeys", kv={"task_name": task_name}),
    ]

    val_bev_3d_transforms = [
        val_collect_3dv,
        e2e_dynamic_common_transforms["ANCResize3DV"],
        e2e_dynamic_common_transforms["ANCCrop3DV"],
        None,
        e2e_dynamic_common_transforms["ANCToTensor3DV"],
        e2e_val_target,
        val_prepare_temporal_data,
        val_set_temporal_clear_flag,
        set_temporal_return_latest_flag,
        set_fake_dataset_flag,
        dict(type="AddKeys", kv={"task_name": task_name}),
    ]
    # calibaion 需要依赖真实数据，不需要set_fake_dataset_flag，
    # inputs里面没有定义fake_dataset_flag.
    calib_bev_3d_transforms = copy.deepcopy(val_bev_3d_transforms)
    calib_bev_3d_transforms.pop(
        calib_bev_3d_transforms.index(set_fake_dataset_flag)
    )
    return (
        remove_none(train_bev_3d_transforms),
        remove_none(val_bev_3d_transforms),
        remove_none(calib_bev_3d_transforms),
    )


def get_dataloader(
    train_datasets,
    val_datasets,
    calib_datasets,
):
    data_loader = dict(
        type=torch.utils.data.DataLoader,
        collate_fn=collate_e2e_dynamic,
        dataset=dict(
            type="ConcatDataset", datasets=train_datasets, with_flag=True
        ),
        batch_size=train_batch_size_per_gpu,
        shuffle=False,
        num_workers=train_num_workers,
        pin_memory=False,
        worker_init_fn=worker_reset_seed,
    )
    data_loader["shuffle"] = False
    data_loader["persistent_workers"] = train_num_workers > 0
    data_loader["sampler"] = dict(
        type=ANCDistributedGroupClipSampler,
        samples_per_gpu=train_batch_size_per_gpu,
        sub_clip_num=num_frames_per_clip // train_num_frames_per_iter,
    )

    if use_split_dataloader:
        data_loader = convert_to_temporal_split_dataloader(
            data_loader, dataloader_seed
        )

    val_data_loader = dict(
        type=torch.utils.data.DataLoader,
        collate_fn=collate_e2e_dynamic,
        dataset=dict(
            type="ConcatDataset",
            datasets=val_datasets,
            with_flag=True,
        ),
        batch_size=val_batch_size_per_gpu,
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=False,
        worker_init_fn=worker_reset_seed,
    )
    val_data_loader["sampler"] = dict(type=ANCDistributedClipValSampler)
    val_data_loader["batch_size"] = val_batch_size_per_gpu

    calib_data_loader = copy.deepcopy(val_data_loader)
    calib_data_loader["dataset"] = dict(
        type="ConcatDataset",
        datasets=calib_datasets,
        with_flag=True,
    )
    return data_loader, val_data_loader, calib_data_loader


def prepare_e2e_fake_dataset(datasets):

    for dataset in datasets:
        transforms_list = dataset["dataset"]["transforms"]
        fake_dataset_transform_index = transforms_list.index(
            set_fake_dataset_flag
        )
        transforms_list[fake_dataset_transform_index]["kv"][
            "fake_dataset_flag"
        ] = True
    return datasets


# -------------------------- MODEL --------------------------


def get_metric_updater(metrics, per_metric_patterns, task_name):

    metric_updater = dict(
        type="MetricUpdater",
        metrics=metrics,
        metric_update_func=update_metric_using_regex(
            per_metric_patterns=per_metric_patterns
        ),
        step_log_freq=log_freq,
        epoch_log_freq=1,
        log_prefix=task_name,
        reset_metrics_by="epoch",
    )
    return metric_updater


def val_metric_update_func(metrics, batch, output):
    for m in metrics:
        m.update(batch, output)
