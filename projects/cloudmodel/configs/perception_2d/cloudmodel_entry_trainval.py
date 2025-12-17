from importlib import import_module

import torch
from tasks.all_tasks import ModelType

from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.utils.config import ConfigVersion
from projects.cloudmodel.configs.perception_2d.build_callback import (
    build_callback,
)
from projects.cloudmodel.configs.perception_2d.build_dataloader import (
    build_multitask_dataloader as build_dataloader,
)
from projects.cloudmodel.configs.perception_2d.build_model import build_model
from projects.cloudmodel.configs.perception_2d.build_optimizer import (
    build_optimizer,
)
from projects.cloudmodel.configs.perception_2d.build_validation import (
    build_validation,
)
from projects.cloudmodel.configs.perception_2d.build_visualization import (
    build_visualization,
)
from projects.cloudmodel.configs.perception_2d.common import (
    checkpoint_path,
    common_inputs,
    do_ema,
    do_freeze_bn,
    do_val,
    optimizer_cfg,
    pipeline_test,
    resume,
    save_root,
)
from projects.cloudmodel.configs.perception_2d.multitasks import TASKS

# ------------------------------- HAT settings ------------------------------ #
log_rank_zero_only = True
cudnn_benchmark = True
seed = None
march = "bayes"

# -------------------------- build multitask model -------------------------- #
TASK_CONFIGS = [import_module(t) for t in TASKS]
model_types = [i.task.model_type.value for i in TASK_CONFIGS]
assert (
    len(list(set(model_types))) == 1
), "all tasks should use the same model type"
model_type = model_types[0]
train_model = build_model(TASK_CONFIGS, common_inputs, mode="train")
# ------------------------ build multitask dataloader ----------------------- #
train_dataloader = build_dataloader(
    TASK_CONFIGS,
    mode="train",
    inf_loader=True if model_type == ModelType.roi_model.value else False,
)  # TODO model_type == ModelType.roi_model.value

# -------------------------- build batch processor -------------------------- #
train_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    batch_transforms=None,
    loss_collector=collect_loss_by_regex("^.*loss.*"),
    enable_amp=True,
    grad_scaler=torch.cuda.amp.GradScaler(init_scale=512.0),
    delay_sync=True,
)

# ----------------------------- build optimizer ----------------------------- #
optimizer = build_optimizer(
    name=optimizer_cfg["name"],
    lr=optimizer_cfg["lr"],
    weight_decay=optimizer_cfg["weight_decay"],
)

# ----------------------------- build callbacks ----------------------------- #
callbacks = build_callback(
    TASK_CONFIGS,
    num_steps=optimizer_cfg["num_steps"],
    checkpoint_save_root=save_root,
    checkpoint_interval_by="step",
    checkpoint_save_interval=max(optimizer_cfg["num_steps"] // 10, 1000),
    log_freq=50,
    do_ema=do_ema,
    do_freeze_bn=do_freeze_bn,
)

# ------------------ build callbacks for online validation ------------------ #
if do_val:
    # build validation model
    val_model = build_model(TASK_CONFIGS, common_inputs, mode="val")

    # build validation batch processor
    val_batch_processor = dict(
        type="MultiBatchProcessor",
        need_grad_update=False,
        enable_amp=False,
    )

    # build visualization callback in validation
    val_vis_callback = build_visualization(
        TASK_CONFIGS, save_root, det_threshold=0.3, mode="val"
    )

    val_callbacks = build_validation(
        val_model,
        val_batch_processor,
        TASK_CONFIGS,
        vis_callback=val_vis_callback,  # set to None to disable visualization
        interval_by="step",
        val_interval=max(optimizer_cfg["num_steps"] // 10, 5000),
        custom_length=40 if pipeline_test else None,
        inf_loader=True if model_type == ModelType.roi_model.value else False,
    )

    callbacks.extend(val_callbacks)

# ---------------------------- build the trainer ---------------------------- #
# If you want to load a pre-trained checkpoint, you can do it here using the
# checkpoint_path set in common.py.
if checkpoint_path is None:
    model_convert_pipeline = None
else:
    model_convert_pipeline = dict(
        type="LoadCheckpoint",
        checkpoint_path=checkpoint_path,
        state_dict_update_func=None,
        check_hash=False,
        allow_miss=True,
        ignore_extra=True,
        verbose=True,
    )
# ---------------------------- external ------------------------------------- #
# If you want to do some training speed analysis, uncomment these two lines.
# There are several profilers to choose. See hat/profiler.
# from hat.profiler.profilers import SimpleProfiler
# profiler = SimpleProfiler(dirpath="./", filename="simple_profiler")

# specify config version. Currently, V2 is OK
VERSION = ConfigVersion.v2

# Assemble all components into this trainer which is required by tools/train.py
# to start the training and validation.
float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=train_model,
    model_convert_pipeline=model_convert_pipeline,
    data_loader=train_dataloader,
    optimizer=optimizer,
    batch_processor=train_batch_processor,
    callbacks=callbacks,
    stop_by=optimizer_cfg["stop_by"],
    num_steps=optimizer_cfg["num_steps"],
    num_epochs=optimizer_cfg["num_epochs"],
    sync_bn=optimizer_cfg["sync_bn"],
    device=None,
    profiler=None,
    find_unused_parameters=True,
    sync_bn_by_host=False if optimizer_cfg["sync_bn"] else True,
    resume_optimizer=resume,
    resume_epoch_or_step=resume,
)
