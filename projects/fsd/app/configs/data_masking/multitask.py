import os
from importlib import import_module

import torch
from common import (
    ckpt_dir,
    compile_dir,
    docker_image,
    folder_list,
    framework,
    fuse_bn_lr,
    fuse_bn_steps,
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
    save_interval,
    task_label,
    task_name,
    training_step,
    upload_folder_name,
    warmup_begin_lr,
    warmup_steps,
    weight_decay,
    with_bn_lr,
    with_bn_steps,
)
from dataloader import data_loader
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
    max_steps=with_bn_steps - warmup_steps,
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


fuse_bn_lr_callback = dict(
    type="CosLrUpdater",
    max_steps=fuse_bn_steps - warmup_steps,
    warmup_by="step",
    warmup_len=warmup_steps,
    warmup_begin_lr=warmup_begin_lr,
    step_log_interval=log_freq,
)
aidi_expmodel_callback = dict(
    type="AIDIExpModel",
    model_name=task_name,
    task_type="multitask",
    save_model="last",
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

with_bn_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="QATFuseBNConvertPipeline",
        qat_mode="with_bn_reverse_fold",
        pre_stage_fuse_patterns=[],
        cur_stage_fuse_patterns=[],
        fuse_part_configs=dict(
            fuse_method="fuse_norm",
            regex=True,
            strict=False,
        ),
        checkpoint_mode="pre_stage",
        checkpoint_configs=dict(
            checkpoint_path=pretrain_checkpoint,
            allow_miss=True,
            ignore_extra=True,
            verbose=True,
        ),
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        lr=with_bn_lr,
        weight_decay=weight_decay,
    ),
    batch_processor=batch_processor,
    stop_by="step",
    num_steps=with_bn_steps,
    device=None,  # set when building
    sync_bn=sync_bn,
    callbacks=[
        stat_callback,
        lr_callback,
        ckpt_callback,
        aidi_expmodel_callback,
    ]
    + metric_updaters,
)


fuse_bn_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="QATFuseBNConvertPipeline",
        qat_mode="with_bn_reverse_fold",
        pre_stage_fuse_patterns=[],
        cur_stage_fuse_patterns=[
            "^.*backbone",
            "^.*fix_channel_neck",
            "^.*head",
        ],
        fuse_part_configs=dict(
            fuse_method="fuse_norm",
            regex=True,
            strict=False,
        ),
        checkpoint_mode="pre_stage",
        checkpoint_configs=dict(
            checkpoint_path=os.path.join(
                ckpt_dir, "with_bn-checkpoint-last.pth.tar"
            ),
            allow_miss=True,
            ignore_extra=True,
            verbose=True,
        ),
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        lr=fuse_bn_lr,
        weight_decay=weight_decay,
    ),
    batch_processor=batch_processor,
    stop_by="step",
    num_steps=fuse_bn_steps,
    device=None,  # set when building
    sync_bn=sync_bn,
    callbacks=[
        stat_callback,
        fuse_bn_lr_callback,
        ckpt_callback,
        aidi_expmodel_callback,
    ]
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
                    ckpt_dir, "fuse_bn-checkpoint-last.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
                verbose=True,
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
    name="datamasking",
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "datamasking.hbm"),
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
