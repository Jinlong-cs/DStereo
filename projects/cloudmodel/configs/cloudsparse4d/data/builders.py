# Copyright (c) Horizon Robotics. All rights reserved.
import os
from functools import partial

import torch

from hat.utils import Config
from ..evaluation import cloudsparse4d_eval_collate, proj_fn_dict


def get_view_shape(model_setting="x3c"):
    """view shape for different model setting, i.e., x3c, as33"""
    if "as33" in model_setting:
        raw_image_hw = (1280, 2048)
    elif model_setting == "x3c":
        raw_image_hw = (1280, 1920)
    else:
        raise NotImplementedError(model_setting)
    per_view_shape = {
        "camera_front_left": raw_image_hw,
        "camera_front_right": raw_image_hw,
        "camera_rear_left": raw_image_hw,
        "camera_rear_right": raw_image_hw,
        "camera_rear": raw_image_hw,
        "camera_front": (2160, 3840),
        "camera_front_30fov": (2160, 3840),
    }
    return per_view_shape


def get_multiview_dataset_list(
    rec_path_list,
    transforms,
    model_setting,
    cfg,
):
    return [
        dict(
            type="MultiViewRecDataset",
            rec_path=rec_path,
            rec_idx_file=rec_path + ".idx",
            camera_view_names=cfg.camera_view_names,
            view_shapes=get_view_shape(model_setting),
            to_rgb=True,
            decode_img=True,
            transforms=transforms,
            homo_cfg=None,
        )
        for rec_path in rec_path_list
    ]


def get_temporal_lmdb_dataset_list(
    lmdb_path_list,
    transforms,
    cfg,
):
    return [
        dict(
            type="TemporalLmdbDataset",
            idx_path=os.path.join(lmdb_path, "idx"),
            anno_path=os.path.join(lmdb_path, "anno"),
            transforms=[
                dict(
                    type="MVImgLmdbReader",
                    img_path=os.path.join(lmdb_path, "img"),
                    to_rgb=True,
                )
            ]
            + transforms,
            max_interval=cfg.max_interval,
            max_len_in_clip=cfg.max_len_in_clip,
        )
        for lmdb_path in lmdb_path_list
    ]


def get_leaderboard_dataloader_and_callback(
    ds_dict,
    cfg,
    transforms,
    project,
    eval_class,
):
    """get dataloader and aidi-eval callback for one leadboard.

    Args:
        ds_dict: Dict, dataset info(id, eval_class, ...)
        cfg: Config, base config
        transforms: eval data transforms
        project: project name, i.e., pilot, sd, mono
        eval_class: classname to evaluate, i.e., vehicle, person, cyclist
    """

    img_transform = dict(
        type="MVImgJsonReader",
        img_dir=ds_dict["img_dir"],
        to_rgb=True,
    )

    transforms = [img_transform] + transforms
    if project == "sd":
        dataset = dict(
            type="TemporalSDJsonDataset",
            eval_class=eval_class,
            json_file=ds_dict["json_file"],
            transforms=transforms,
            max_interval=0,
            max_len_in_clip=-1,
            category_name2id=cfg.category_name2id,
            sub_category_name2id=cfg.sub_category_name2id,
        )
    elif project == "mono":
        transforms = [t for t in transforms if t["type"] != "GetCalibParams"]
        dataset = dict(
            type="TemporalMonoJsonDataset",
            eval_class=eval_class,
            json_file=ds_dict["json_file"],
            transforms=transforms,
            max_interval=0,
            max_len_in_clip=-1,
            category_name2id=cfg.category_name2id,
            sub_category_name2id=cfg.sub_category_name2id,
        )
    elif project == "pilot":
        dataset = dict(
            type="TemporalJsonDataset",
            json_file=ds_dict["json_file"],
            transforms=transforms,
            max_interval=0,
            max_len_in_clip=-1,
        )
    else:
        raise NotImplementedError
    if cfg.temporal_eval:
        val_dataloader = dict(
            type=torch.utils.data.DataLoader,
            dataset=dict(
                type="ConcatDataset",
                datasets=[dataset],
                with_flag=True,
                accumulate_flag=True,
            ),
            batch_size=1,
            num_workers=4,
            pin_memory=False,
            sampler=None,
            batch_sampler=dict(
                type="DistributedGroupInBatchSampler",
                dataset=dict(
                    type="ConcatDataset",
                    datasets=[dataset],
                    with_flag=True,
                    accumulate_flag=True,
                ),
                batch_size=1,
                stop_by_epoch=True,
            ),
            collate_fn=partial(cloudsparse4d_eval_collate, samples_per_gpu=1),
        )
    else:
        val_batch_size = cfg.val_batch_size
        val_dataloader = dict(
            type=torch.utils.data.DataLoader,
            collate_fn=partial(
                cloudsparse4d_eval_collate, samples_per_gpu=val_batch_size
            ),
            dataset=dict(
                type="ConcatDataset",
                datasets=[dataset],
                with_flag=True,
                accumulate_flag=True,
            ),
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=cfg.val_num_workers,
            pin_memory=False,
            sampler=dict(type=torch.utils.data.DistributedSampler),
        )

    reformat_output_fn = proj_fn_dict[project]
    reformat_out_fn_kwargs = dict(
        obj_key=eval_class,
        dump_obj_key=ds_dict.get("dump_key", eval_class),
        idx2cam={
            0: "front_right",
            1: "rear_right",
            2: "front_left",
            3: "rear_left",
            4: "rear",
            5: "front",
            6: "front_30fov",
        },
        camera_view_names=cfg.sub_dirs,
        class_key_id_map=dict(
            person=0,
            vehicle=1,
            cyclist=2,
            pedestrian=0,
        ),
        score_thresh=ds_dict.get("score_th", 0.1),
        center_type="cube_center",
    )
    prediction_name = (
        cfg.task_name
        if not cfg.temporal_eval
        else cfg.task_name + "_temp_eval"
    )
    ds_id = ds_dict["id"]
    eval_callback = dict(
        type="AIDIEval",
        aidi_eval_dataset_id=ds_id,
        output_root=f"./eval_res/{ds_id}",
        prediction_name=prediction_name,
        prediction_tags=cfg.tags,
        project_id=cfg.project_id,
        reformat_output_fn=reformat_output_fn,
        reformat_out_fn_kwargs=reformat_out_fn_kwargs,
        cpu=8,
        cpu_mem_ratio=4,
    )

    callbacks = [
        eval_callback,
        dict(
            type="StatsMonitor",
            log_freq=5,
            batch_size=1 if cfg.temporal_eval else None,
        ),
    ]
    return val_dataloader, callbacks


def build_train_datasets(train_datasets, cfg):
    """build train datasets based on the train datasets dict and the base config.
    Args:
        train_datasets: Dict, contain all train dataset infos.
        cfg: Config, base cfg.
    """
    all_train_datasets = []
    for ds_dict in train_datasets:
        if ds_dict["type"] == "multiview":
            ds_path = os.path.join(
                os.path.dirname(__file__), "datasets", ds_dict["file"]
            )
            data_paths = Config.fromfile(
                ds_path
            ).datapaths.multiview_dynamic_3d_detection.train_data_paths  # noqa
            for dpath in data_paths:
                all_train_datasets.extend(
                    get_multiview_dataset_list(
                        dpath["rec_path"],
                        ds_dict["transforms"],
                        ds_dict["model_setting"],
                        cfg,
                    )
                )
        elif ds_dict["type"] == "temporal":
            ds_path = os.path.join(
                os.path.dirname(__file__), "datasets", ds_dict["file"]
            )
            data_paths = Config.fromfile(
                ds_path
            ).datapaths.multiview_dynamic_3d_detection.train_data_paths  # noqa
            for dpath in data_paths:
                all_train_datasets.extend(
                    get_temporal_lmdb_dataset_list(
                        dpath["lmdb_path"], ds_dict["transforms"], cfg
                    )
                )
        elif ds_dict["type"] == "cloudbev":
            # TODO: add sd datasets
            pass
        else:
            raise NotImplementedError(ds_dict["type"])
    return all_train_datasets


def build_train_dataloader(
    train_dataset_list,
    batch_size,
    num_workers,
    collate_fn,
    dataset_type="ConcatTemporalMixedDataset",
    batch_sampler="DistributedGroupInBatchSampler",
):
    """build train data loader

    Args:
        train_dataset_list: List, all train datasets.
        batch_size: int, train batch-size.
        num_workers: int, workers per gpu.
        collate_fn: collate function.
        dataset_type: type of the dataset, default as "ConcatTemporalMixedDataset".
        batch_sampler: sampler, default as "DistributedGroupInBatchSampler".
    """
    train_dataloader = dict(
        type=torch.utils.data.DataLoader,
        dataset=dict(
            type=dataset_type,
            datasets=train_dataset_list,
            with_flag=True,
            accumulate_flag=True,
        ),
        batch_size=1,
        batch_sampler=dict(
            type=batch_sampler,
            dataset=dict(
                type=dataset_type,
                datasets=train_dataset_list,
                with_flag=True,
                accumulate_flag=True,
            ),
            batch_size=batch_size,
            skip_prob=-1,
        ),
        sampler=None,
        collate_fn=partial(collate_fn, samples_per_gpu=batch_size),
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_dataloader


def build_eval_dataloaders_and_callbacks(eval_datasets, cfg, transforms):
    """build eval dataloaders and callbacks

    Args:
        eval_datasets: eval datasets, could be List(["6042110"]), "all", "sd", "sd_vehicle", "vehicle".
        cfg: Config, base config.
        transforms: List, data transforms for evaluation.
    """
    leaderboard_path = os.path.join(
        os.path.dirname(__file__), "datasets/leaderboards.py"
    )
    leaderboards = Config.fromfile(leaderboard_path).leaderboards
    dataloaders, callbacks = [], []
    for project, proj_datasets in leaderboards.items():
        for eval_class, datasets in proj_datasets.items():
            if isinstance(eval_datasets, str):
                if eval_datasets == "all":
                    proposals = datasets
                elif (
                    eval_datasets in leaderboards.keys()
                    and eval_datasets == project
                ):
                    proposals = datasets
                elif (
                    eval_datasets in proj_datasets.keys()
                    and eval_datasets == eval_class
                ):
                    proposals = datasets
                else:
                    p, e = eval_datasets.split("_")
                    if p == project and e == eval_class:
                        proposals = datasets
                    else:
                        proposals = []
            elif isinstance(eval_datasets, list):
                # get leaderboards by id
                proposals = [
                    ds for ds in datasets if ds["id"] in eval_datasets
                ]
            else:
                raise NotImplementedError(eval_datasets)
            for p in proposals:
                dataloader, callback = get_leaderboard_dataloader_and_callback(
                    p,
                    cfg,
                    transforms,
                    project,
                    eval_class if "dump_key" not in p else p["dump_key"],
                )
                dataloaders.append(dataloader)
                callbacks.append(callback)
    return dataloaders, callbacks
