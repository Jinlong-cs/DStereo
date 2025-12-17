# -*- coding:utf-8 -*-
# Copyright (c) Horizon Robotics, All rights reserved.
# flake8: noqa

import os
import pathlib

import torch
from aidisdk.utils import running_in_cluster
from config.tasks.mmasr.examples.fyz_qirui_1k.data.test import (
    test_dataloaders,
    test_db_levels,
)
from config.tasks.mmasr.examples.fyz_qirui_1k.data.train import (
    batch_size,
    train_dataloader,
)
from config.tasks.mmasr.examples.fyz_qirui_1k.data.val import (
    val_dataloaders,
    val_db_levels,
)
from config.tasks.mmasr.examples.fyz_qirui_1k.model.ao12_vo6_resnet18 import (
    model,
)
from config.tasks.mmasr.examples.k8s_job.k8s_job import k8s_config
from horizon_plugin_pytorch.march import March

from hat.callbacks.checkpoint import FlexibleStateDict
from hat.callbacks.lr_updater import NoamLrUpdater
from hat.engine.processors.loss_collector import collect_loss_by_index
from hat.metrics.avspeech_metric import WER
from hat.metrics.loss_show import LossShow
from hat.utils.root_helper import RootHelper

join = os.path.join

training_step = os.environ.get("HAT_TRAINING_STEP", "float")

# ! HAT ENGINE 配置
# 使用配置文件的名字作为 task_name
# 可以手动设置名字
task_name = pathlib.Path(__file__).stem
# 保存模型的前缀名称
model_prefix = "MMASR-" + task_name.split("_")[0]

cudnn_benchmark = False
seed = None
log_rank_zero_only = True
march = March.BAYES
qat_mode = "fuse_bn"
num_epochs = 1000

# ! 训练相关的配置
CARP_ROOT = RootHelper.get_project_root("halo/avspeech")
data_root = os.path.join(CARP_ROOT, "data")


cluster_device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
# cluster_device_ids = [0, 1]
local_device_ids = [0]

if running_in_cluster():
    # 集群训练的一些配置
    device_ids = cluster_device_ids
    ckpt_dir = f"/job_data/models/{task_name}/"
    epoch_log_freq = 1
    step_log_freq = 100
else:
    # 本地训练的一些配置
    device_ids = local_device_ids
    ckpt_dir = f"./job_data/models/{task_name}/"
    epoch_log_freq = 1
    step_log_freq = 100


ao_checkpoint_path = (
    "/horizon-bucket/J2MM/speech/hdf5/mmasr_pretrain_model/avg_30_xingchen.pt"
)
vo_checkpoint_path = "/horizon-bucket/J2MM/speech/hdf5/mmasr_pretrain_model/exp0202_strcture_sc_vo6_resnet18_float-checkpoint-average-min_wer-20-6c1b15e7.pth.tar"


# 任务相关的配置
batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    batch_transforms=None,
    loss_collector=collect_loss_by_index(0),
    enable_amp=False,
)

val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    batch_transforms=None,
    loss_collector=None,
    enable_amp=False,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=step_log_freq,
    batch_size=batch_size,
)

grad_scale_callback = dict(
    type="GradScale",
    module_and_scale=[
        # ("encoder.vfea_module", 0.01),
        # ("encoder.video_encoders", 0.01),
        # ("CTC", 1.0),
        # ("decoder", 0.01),
    ],
    clip_grad_norm=5,
)

freeze_modules = [
    "encoder.embed",
    "encoder.audio_encoders",
    "encoder.vfea_extractor",
    "encoder.video_encoders",
    # "encoder.vfea_module",
]
freeze_model_callback = dict(
    type="FreezeModule",
    modules=[freeze_modules for _ in range(num_epochs)],
    step_or_epoch=list(range(num_epochs)),
    update_by="epoch",
)


def update_metric(metrics, batch, model_outs):
    loss, loss_att, loss_ctc, att_acc, ctc_out = model_outs
    if att_acc is not None:
        att_acc = torch.Tensor([att_acc]).to(loss.device)
    for metric in metrics:
        if isinstance(metric, LossShow):
            metric.update(
                {
                    "loss": loss,
                    "loss_att": loss_att,
                    "loss_ctc": loss_ctc,
                    "att_acc": att_acc,
                }
            )
        if isinstance(metric, WER):
            tokens = [" ".join(_tokens) for _tokens in batch["tokens"]]
            metric.update(ctc_out, tokens)


metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=step_log_freq,
    reset_metrics_by="log",
    epoch_log_freq=epoch_log_freq,
    log_prefix=model_prefix,
)


def update_tb_train(writer, model_outs, global_step_id, **kwargs):
    mode = "train"
    metrics = kwargs["train_metrics"]
    for metric in metrics:
        if isinstance(metric, LossShow):
            head = f"{mode}/loss"
        elif isinstance(metric, WER):
            head = f"{mode}/wer"
        else:
            continue
        names, values = metric.get()
        if not isinstance(names, (list, tuple)):
            names = (names,)
            values = (values,)
        state = {k: v for k, v in zip(names, values)}
        writer.add_scalars(head, state, global_step_id)


def update_tb_val(writer, epoch_id, **kwargs):
    mode = "val"
    metrics = kwargs["train_metrics"]
    for metric in metrics:
        if isinstance(metric, LossShow):
            head = f"{mode}/loss"
        elif isinstance(metric, WER):
            head = f"{mode}/wer"
        else:
            continue
        names, values = metric.get()
        if not isinstance(names, (list, tuple)):
            names = (names,)
            values = (values,)
        state = {k: v for k, v in zip(names, values)}
        writer.add_scalars(head, state, epoch_id)


val_callbacks = []

for db_level in val_db_levels:
    val_metric_updater = dict(
        type="MetricUpdater",
        metric_update_func=update_metric,
        step_log_freq=step_log_freq,
        reset_metrics_by="epoch",
        epoch_log_freq=epoch_log_freq,
        log_prefix=model_prefix + f" {db_level}",
    )
    each_callback = [val_metric_updater]
    val_callbacks.append(each_callback)


val_callback = dict(
    type="Validation",
    data_loader=val_dataloaders,
    batch_processor=val_batch_processor,
    callbacks=val_callbacks,
    log_interval=0,
    interval_by="epoch",
    share_callbacks=False,
    val_on_train_begin=True,
)

val_callback.update(data_loader=val_dataloaders)


ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,
    interval_by="epoch",
    mode=None,
    monitor_metric_key="att_acc",
)

fckpt_callback = dict(
    type=FlexibleStateDict,
    loaded_state_dict=None,
    load_strict=True,
    save_root=ckpt_dir,
    name_prefix=f"{training_step}-{model_prefix}",
    save_interval=epoch_log_freq,
    interval_by="epoch",
    save_on_train_end=True,
)


def audio_state_dict_update_func(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("encoder"):
            if "encoder.encoders" in key:
                key = key.replace("encoders", "audio_encoders")
            new_state_dict[key] = value
        # else:
        #     new_state_dict[key] = value

    return new_state_dict


def video_state_dict_update_func(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("encoder"):
            new_state_dict[key] = value

    return new_state_dict


# train_dataloader.update(batch_size=)
float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=ao_checkpoint_path,
                allow_miss=True,
                ignore_extra=True,
                state_dict_update_func=audio_state_dict_update_func,
                verbose=True,
            ),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=vo_checkpoint_path,
                allow_miss=True,
                ignore_extra=True,
                state_dict_update_func=video_state_dict_update_func,
                verbose=True,
            ),
        ],
    ),
    data_loader=train_dataloader,
    optimizer=dict(
        type=torch.optim.Adam,
        params={},
        lr=0.001,
    ),
    batch_processor=batch_processor,
    num_epochs=num_epochs,
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type=NoamLrUpdater,
            d_model=25000,
            warmup_step=25000,
            step_log_interval=100,
        ),
        metric_updater,
        val_callback,
        ckpt_callback,
        fckpt_callback,
        # train_tb_callback,
        grad_scale_callback,
        freeze_model_callback,
    ],
    train_metrics=[
        dict(type=LossShow),
        dict(type=WER, name="wer"),
    ],
    val_metrics=[
        dict(type=LossShow),
        dict(type=WER, name="wer"),
    ],
)

test_callbacks = []

for db_level in test_db_levels:
    test_metric_updater = dict(
        type="MetricUpdater",
        metric_update_func=update_metric,
        step_log_freq=step_log_freq,
        reset_metrics_by="epoch",
        epoch_log_freq=epoch_log_freq,
        log_prefix=model_prefix + f" {db_level}",
    )
    each_callback = [test_metric_updater]
    test_callbacks.append(each_callback)

float_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path="/mnt/mnt-data-4/shujie.luo/Projects/HAT/job_data/models/exp0305_strcture_asr_model_ao12_vo6_resnet18_pretrain_both_float-checkpoint-average-min_loss-30-fed55c81.pth.tar",
                verbose=True,
                allow_miss=True,
                ignore_extra=True,
            ),
        ],
    ),
    data_loader=test_dataloaders,
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(type=LossShow),
        dict(type=WER, name="wer"),
    ],
    callbacks=test_callbacks,
    share_callbacks=False,
    log_interval=100,
)

# ! 提交任务的实验配置
k8s_config.update(
    job_name=task_name,
    num_machines=1,
    num_gpus_per_machine=max(cluster_device_ids) + 1,
    project_id="LTCS2020101",
    upload_folder_name=task_name,
    job_list=[
        f"python3 tools/trainv2.py \
            --config {os.path.relpath(__file__, RootHelper.HAT_ROOT)} \
            --stage float",
    ],
)
