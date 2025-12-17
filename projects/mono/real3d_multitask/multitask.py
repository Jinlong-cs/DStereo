import os
from importlib import import_module

import torch
from common import (
    ckpt_dir,
    compile_dir,
    docker_image,
    float_lr,
    float_steps,
    folder_list,
    framework,
    input_bucket,
    interval_by,
    job_list,
    job_name,
    job_password,
    launcher,
    log_freq,
    march,
    max_jobtime,
    multitask,
    num_gpus_per_machine,
    num_machines,
    pretrain_checkpoint,
    priority,
    project_id,
    qat_lr,
    qat_steps,
    save_interval,
    task_label,
    training_step,
    upload_folder_name,
    warmup_begin_lr,
    warmup_steps,
    weight_decay,
)
from dataloader import dataloader
from models import deploy_inputs, deploy_model, model  # noqa

from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
# ------------------------ multitask ---------------------
sync_bn = True
enable_amp = False
save_hash = False
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
device_ids = list(range(num_gpus_per_machine))
# -------------------------- task --------------------------
task_names = [t["name"] for t in multitask]
TASK_CONFIGS = [import_module(t) for t in task_names]
metric_updaters = [T.metric_updater for T in TASK_CONFIGS]

batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    batch_transforms=[
        dict(type="BgrToYuv444", rgb_input=False),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
    loss_collector=collect_loss_by_regex("^.*loss.*"),
    enable_amp=enable_amp,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=log_freq,
)


lr_callback = dict(
    type="CosLrUpdater",
    max_steps=float_steps - warmup_steps,
    warmup_by="step",
    warmup_len=warmup_steps,
    warmup_begin_lr=warmup_begin_lr,
    step_log_interval=log_freq,
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,
)


qat_lr_callback = dict(
    type="CosLrUpdater",
    max_steps=qat_steps - warmup_steps,
    warmup_by="step",
    warmup_len=warmup_steps,
    warmup_begin_lr=warmup_begin_lr,
    step_log_interval=log_freq,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=False,
    best_refer_metric=None,
    save_interval=save_interval,
    interval_by=interval_by,
    save_on_train_end=True,
    save_hash=save_hash,
)

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=pretrain_checkpoint,  # noqa
                allow_miss=True,
                ignore_extra=True,
            ),
        ],
    ),
    data_loader=dataloader,
    optimizer=dict(
        type=torch.optim.AdamW,
        lr=float_lr,
        weight_decay=weight_decay,
    ),
    batch_processor=batch_processor,
    stop_by="step",
    num_steps=float_steps,
    device=None,  # set when building
    sync_bn=sync_bn,
    callbacks=[stat_callback, lr_callback, ckpt_callback] + metric_updaters,
)


qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-last.pth.tar"
                ),
            ),
            dict(type="Float2QAT"),
        ],
    ),
    data_loader=dataloader,
    optimizer=dict(
        type=torch.optim.Adam,
        lr=qat_lr,
        weight_decay=weight_decay,
    ),
    batch_processor=batch_processor,
    stop_by="step",
    num_steps=qat_steps,
    device=None,  # set when building
    sync_bn=sync_bn,
    callbacks=[stat_callback, qat_lr_callback, ckpt_callback]
    + metric_updaters,
)


int_infer_trainer = dict(
    type="Trainer",
    model=deploy_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-last.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
            ),
            dict(type="QAT2Quantize"),
        ],
    ),
    data_loader=None,
    optimizer=None,
    batch_processor=None,
    num_epochs=0,
    device=None,
    callbacks=[
        ckpt_callback,
        trace_callback,
    ],
)

compile_cfg = dict(
    march=march,
    name="real3d_with_desensitization",
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "real3d_with_desensitization.hbm"),
    layer_details=False,
    input_source=["pyramid"],
    input_layout="NHWC",
    output_layout="NHWC",
)

k8s_config = dict(
    job_name=job_name,
    job_password=job_password,
    num_machines=num_machines,
    num_gpus_per_machine=num_gpus_per_machine,
    framework=framework,
    task_label=task_label,
    project_id=project_id,
    input_bucket=input_bucket,
    priority=priority,
    docker_image=docker_image,
    max_jobtime=max_jobtime,
    launcher=launcher,
    upload_folder_name=upload_folder_name,
    folder_list=folder_list,
    job_list=job_list,
)
