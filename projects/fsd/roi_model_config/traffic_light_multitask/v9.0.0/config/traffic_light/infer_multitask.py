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
from traffic_light_color_cls import traffic_light_color_sub_classnames
from traffic_light_fp_cls import traffic_light_fp_sub_classnames
from traffic_light_time_cls import traffic_light_time_sub_classnames
from traffic_light_type_cls import traffic_light_category_sub_classnames

is_local_train = not os.path.exists("/running_package")
bucket_root = "/horizon-bucket" if is_local_train else "/bucket/input"

suffix = os.getenv("HAT_TL_PREDICTION_NAME_SUFFIX", "")
prediction_name = f"{model_name}_{suffix}" if suffix else model_name
device_ids = [0, 1]
log_rank_zero_only = True
# If eval_part_graph_model_only == True, forward only part graph model
# for diff task datasets, it usually speeds eval up
eval_part_graph_model_only = False

eval_tags = ["hat", "sd", "tl"]

march = "bayes"

all_callbacks = []
all_data_loaders = []

# data
ds_path = os.path.join(
    os.path.dirname(__file__),
    "../datasets/infer_data.lst",
)
with open(ds_path, "r") as fp:
    data_list = []
    for data_path in fp.readlines():
        data_list.append(data_path.split("\n")[0])


task_names = [task["name"] for task in tasks]

for data_path in data_list:
    data_loader = dict(
        type=torch.utils.data.DataLoader,
        drop_last=False,
        persistent_workers=True,
        batch_size=pred_batch_size,
        shuffle=False,
        num_workers=4,
        dataset=dict(
            type="DistributedComposeRandomDataset",
            sample_weights=[1],
            multi_sample_output=True,
            datasets=[
                dict(
                    type="TLInferDataset",
                    data_path=data_path,
                    transforms=val_transforms,
                )
            ],
        ),
    )
    all_data_loaders.append(data_loader)

    visualize_callback = dict(
        type="TLMultitaskVisualize",
        output_dir=data_path,
        out_keys={
            "traffic_light_color_classification": traffic_light_color_sub_classnames,  # noqa
            "traffic_light_category_classification": traffic_light_category_sub_classnames,  # noqa
            "traffic_light_fp_classification": traffic_light_fp_sub_classnames,
            "traffic_light_time_classification": traffic_light_time_sub_classnames,  # noqa
        },
        save_viz_imgs=True,
        model_version="v0.0.1",
        vis_configs=dict(
            color=(0, 255, 0),
            thickness=2,
        ),
        overwrite=True,
    )
    callbacks = [
        visualize_callback,
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
            dict(type="BgrToYuv444", rgb_input=True),
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
