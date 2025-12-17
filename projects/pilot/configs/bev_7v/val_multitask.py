import os
from copy import deepcopy
from importlib import import_module

from hat.core.task_sampler import TaskSampler
from projects.pilot.configs.bev_7v.common import (
    do_val_visualize,
    model_checkpoint,
    model_name,
    model_version,
    save_prefix,
    tasks,
)
from projects.pilot.configs.bev_7v.models import val_model
from projects.pilot.configs.bev_7v.schedule import train_stages

device_ids = [0]
log_rank_zero_only = True

# -------------------------- task --------------------------
task_names = [t["name"] for t in tasks]
TASK_CONFIGS = [import_module(t) for t in task_names]

# -------------------------- val dataloader --------------------------
val_data_loader_list = []
for task in TASK_CONFIGS:
    if hasattr(task, "get_val_dataloader"):
        task_val_data_loader = task.get_val_dataloader()
    else:
        task_val_data_loader = task.val_data_loader_list

    for loader in task_val_data_loader:
        task_data_loader = {task.task_name: loader}
        task_sampler = TaskSampler(
            task_config={task.task_name: {"sampling_factor": 1}},
            method="sample_all",
        )
        val_data_loader_list.append(
            dict(
                type="MultitaskInfLoader",
                loaders=task_data_loader,
                task_sampler=task_sampler,
                return_task=True,
                iter_once=True,
                __build_recursive=False,
            )
        )

# -------------------------- metric --------------------------
metric_callbacks = []
for config in TASK_CONFIGS:
    if hasattr(config, "val_metric_updater_list"):
        val_metric_updater_list = config.val_metric_updater_list
    else:
        val_metric_updater_list = [config.val_metric_updater]
    for val_metric_updater in val_metric_updater_list:
        if do_val_visualize and hasattr(config, "visualize"):
            vis_callback = dict(
                type="ANCSaveVisualize",
                task_list=[config.task_name],
                visualize_list=[config.visualize],
                save_dir=os.path.join(save_prefix, "val_visualize"),
                video=None,
                # video=dict(
                #     type="GenerateVideo",
                #     fps=10,
                #     save_dir=os.path.join(save_prefix, "video"),
                # )
                # if not upload_image2aidi
                # else None,
            )
            metric_callbacks.append([vis_callback, val_metric_updater])
        else:
            metric_callbacks.append(val_metric_updater)

assert len(val_data_loader_list) == len(
    metric_callbacks
), "Length of loader should equal to metric callback!"

# -------------------------- batch processor --------------------------
val_batch_transforms = [
    dict(type="ANCConvertToYuv"),
    dict(type="ANCNormalize3DV", mean=128, std=128),
]
batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
    batch_transforms=val_batch_transforms,
)

# -------------------------- predictor --------------------------
base_predictor = dict(
    type="Predictor",
    model=val_model,
    data_loader=val_data_loader_list,
    batch_processor=batch_processor,
    callbacks=metric_callbacks,
    log_interval=50,
    num_epochs=1,
    share_callbacks=False,
)

for stage in train_stages:
    predictor = deepcopy(base_predictor)
    predictor.update(
        model_convert_pipeline=dict(
            type="FloatQatConvertPipeline",
            qat_mode="fuse_bn",
            enable_qat=stage == "qat",
            checkpoint_mode="resume",
            checkpoint_configs=dict(
                checkpoint_path=model_checkpoint
                if model_checkpoint
                else f"aidi_artifact://{model_name}/{stage}/{model_version}/{stage}-checkpoint-last.pth.tar",
                state_dict_update_func=None,
                allow_miss=False,
                ignore_extra=True,
                verbose=1,
            ),
            qconfig_params=None,
        )
    )
    globals()[f"{stage}_predictor"] = predictor
