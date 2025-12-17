import os
from copy import deepcopy

import torch
from common import (
    model_checkpoint,
    model_name,
    model_version,
    pred_batch_size,
    tasks,
    training_step,
    val_transforms,
)
from models import val_model
from schedule import get_fuse_patterns_by_stage, train_stages
from utils import reformat_wk_cls_to_aidi_eval

from hat.data.collates.collates import collate_2d_2pe
from hat.utils import Config

is_local_train = not os.path.exists("/running_package")
bucket_root = "/horizon-bucket" if is_local_train else "/bucket/input"

suffix = os.getenv("HAT_WK_PREDICTION_NAME_SUFFIX", "")
prediction_name = f"{model_name}_{suffix}" if suffix else model_name
device_ids = [0, 1]
log_rank_zero_only = True
# If eval_part_graph_model_only == True, forward only part graph model
# for diff task datasets, it usually speeds eval up
eval_part_graph_model_only = False

eval_tags = ["hat", "sd", "wk"]

march = "bayes"

all_callbacks = []
all_data_loaders = []

# data
ds_path = os.path.join(
    os.path.dirname(__file__),
    "../datasets/eval_datasets.py",
)
dataset_ids = Config.fromfile(ds_path).dataset_ids

local_datapath = os.path.join(
    bucket_root, "auto_eval/adas_eval/eval_platform/fs/"
)

task_names = [task["name"] for task in tasks]


for task_type, all_datasets in dataset_ids.items():
    for dataset, task_cfgs in all_datasets.items():

        data_path = os.path.join(local_datapath, str(dataset), "datasets")
        data_loader = dict(
            type=torch.utils.data.DataLoader,
            sampler=dict(
                type=torch.utils.data.DistributedSampler,
                shuffle=False,
            ),
            batch_size=pred_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=False,
            drop_last=False,
            collate_fn=collate_2d_2pe,
            dataset=dict(
                type="MtlEvalRaw2PEDataset",
                data_path=data_path,
                task_type=task_type,
                to_rgb=False,
                buf_only=True,
                return_orig_img=False,
                transforms=val_transforms,
            ),
        )
        all_data_loaders.append(data_loader)

        reformat_output_fn = reformat_wk_cls_to_aidi_eval
        reformat_out_fn_kwargs = dict(
            task_cfgs=task_cfgs,
        )
        prediction_tags = {}
        for cur_task in task_cfgs:
            prediction_tags.update({cur_task: cur_task.split("_")[-2]})

        eval_callback = dict(
            type="AIDIEval",
            aidi_eval_dataset_id=dataset,
            output_root=f"./eval_res/{dataset}",  # noqa
            prediction_name=prediction_name,
            prediction_tags=prediction_tags,
            project_id="PDT20220004",
            reformat_output_fn=reformat_output_fn,
            reformat_out_fn_kwargs=reformat_out_fn_kwargs,
            task_cfgs=task_cfgs,
        )

        callbacks = [
            eval_callback,
            dict(
                type="StatsMonitor",
                log_freq=5,
            ),
        ]

        all_callbacks.append(callbacks)


base_predictor = dict(
    type="Predictor",
    model=val_model,
    data_loader=all_data_loaders,
    batch_processor=dict(
        type="MultiBatchProcessor"
        if eval_part_graph_model_only
        else "BasicBatchProcessor",
        batch_transforms=[
            dict(type="BgrToYuv444V2", rgb_input=False),
            dict(
                type="TorchVisionAdapter",
                interface="Normalize",
                mean=128.0,
                std=128.0,
            ),
        ],
        need_grad_update=False,
    ),
    callbacks=all_callbacks,
    num_epochs=1,
    share_callbacks=False,
)


for stage in train_stages:
    pre, cur = get_fuse_patterns_by_stage(stage)

    predictor = deepcopy(base_predictor)
    predictor.update(
        model_convert_pipeline=dict(
            type="QATFuseBNConvertPipeline",
            qat_mode="with_bn_reverse_fold",
            pre_stage_fuse_patterns=pre,
            cur_stage_fuse_patterns=cur,
            fuse_part_configs=dict(
                fuse_method="fuse_norm",
                regex=True,
                strict=False,
            ),
            checkpoint_mode="resume",
            checkpoint_configs=dict(
                checkpoint_path=(
                    model_checkpoint
                    if model_checkpoint
                    else f"aidi://{model_name}/{model_version}/{training_step}"
                ),
                allow_miss=False,
                ignore_extra=True,
                verbose=1,
            ),
            qconfig_params=None,
        )
    )
    globals()[f"{stage}_predictor"] = predictor
