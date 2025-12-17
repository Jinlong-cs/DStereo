# Copyright (c) Horizon Robotics. All rights reserved.

import copy
import json
import os
import time
from collections import OrderedDict

import torch
from horizon_plugin_pytorch.quantization import March
from processed_dataset import FPV_PREFIX

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.data.collates.traj_pred_collates import collate_SGNet
from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2

cudnn_benchmark = False
seed = None
log_rank_zero_only = True
march = March.BAYES

task_name = "sgnet"
project = None
local_train = True

# 0.1. Paths.
if local_train:
    assert (
        task_name is not None
    ), "You are training in local env, task name must be assigned."
    device_ids = [0, 1]
    # Set your local path here.
    ckpt_dir = ""
    pretrain_dir = f"{FPV_PREFIX}/train_result/{task_name}/pretrain"
    viz_dir = ""
    tb_dir = ""
else:
    device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    pretrain_dir = f"{FPV_PREFIX}/train_result/{task_name}/pretrain"
    ckpt_dir = "/job_data/models/checkpoint"
    viz_dir = "/job_data/models/viz"
    tb_dir = os.getenv("TENSORBOARD_LOG_PATH") or os.path.join(
        "/job_data/models/tensorboard", ".aidi"
    )
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(viz_dir, exist_ok=True)
os.makedirs(tb_dir, exist_ok=True)
tm = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

# 0.2. Parameters.
batch_size_per_gpu = 64
train_num_workers, val_num_workers = 4, 2
float_lr, quanti_lr = 0.0005, 0.0001
float_epoch, quanti_epoch = 5, 30
weight_decay = 0.00001
warmup_epoch = 1
log_freq = 100
hidden_size = 512
pred_dim = 4
input_dim = 4
latent_dim = 32
k_value = 1
enc_steps = 3
dec_steps = 8
mu = 0.0
sigma = 1.0
metric_name = "fpvped"
monitor_metric_key_name = f"{metric_name}_min_rmse15_{k_value}"
monitor_metric_key_stat = "avg"
monitor_metric_key = f"{monitor_metric_key_name}_{monitor_metric_key_stat}"

# 1. Setup data loader.
train_dataloader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="TrajPredFPVPedDataset",
        dataset_root_paths=[
            os.path.join(FPV_PREFIX, "train_data/FPV_Ped_Train_2022_04"),
            os.path.join(FPV_PREFIX, "train_data/FPV_Ped_Train_2022_05"),
            os.path.join(FPV_PREFIX, "train_data/FPV_Ped_Train_2022_06"),
            os.path.join(FPV_PREFIX, "train_data/FPV_Ped_Train_2022_07"),
            os.path.join(
                FPV_PREFIX, "train_data/FPV_Ped_Train_2023_02_AidiClips"
            ),
        ],
        regen_data_cache=False,
        feature_type=["bbox"],
        obs_type={"ped": 2},
        enable_relative_tar_coord=True,
        enable_cvae=True,
        stage="train",
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    collate_fn=collate_SGNet,
    shuffle=False,
    num_workers=train_num_workers,
    drop_last=True,
)

val_dataloader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="TrajPredFPVPedDataset",
        dataset_root_paths=[
            os.path.join(FPV_PREFIX, "train_data/FPV_Ped_Val_2022_12_ICAmode"),
        ],
        regen_data_cache=False,
        feature_type=["bbox"],
        obs_type={"ped": 2},
        enable_relative_tar_coord=True,
        enable_cvae=True,
        stage="val",
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    collate_fn=collate_SGNet,
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=val_num_workers,
    drop_last=True,
)

# 2. Setup Model.
model = OrderedDict(
    type="SGNet",
    backbone=dict(
        type="SGNetEncoder",
        in_channels=pred_dim,
        enc_steps=enc_steps,
        dec_steps=dec_steps,
        hidden_size=hidden_size,
        is_calibration_step=False,
    ),
    head=dict(
        type="SGNetCvaeDecoder",
        in_channels=pred_dim,
        out_channels=pred_dim,
        hidden_size=hidden_size,
        enc_steps=enc_steps,
        dec_steps=dec_steps,
        latent_dim=latent_dim,
        k_value=k_value,
        is_calibration_step=False,
    ),
    neck=None,
    post_process=dict(
        type="SGNetPostProcessor",
        bbox_type="cxcywh",
        normalize_type="zero-one",
        save_res=True,
        save_res_path=viz_dir,
        dir_name=tm,
    ),
    losses=dict(
        type="SGNetLoss",
        KLD_weight=2000,
        goal_weight=0.02,
        pred_weight=0.01,
        prob_weight=1,
        bbox_loss_weight=2,
        iou_loss_weight=200,
        mpb_loss_weight=1,
        # iou_loss_type="ciou_loss",
    ),
    k_value=k_value,
    is_int_infer_model=False,
)

# 3. Setup metrics.
# --3.1. Training metric (loss) updater.
loss_metrics = [
    dict(type="LossShow", name="total_loss"),
    dict(type="LossShow", name="KLD_loss"),
    dict(type="LossShow", name="goal_loss"),
    dict(type="LossShow", name="pred_loss"),
    dict(type="LossShow", name="goal_bbox_loss"),
    dict(type="LossShow", name="goal_iou_loss"),
    dict(type="LossShow", name="goal_mpb_loss"),
    dict(type="LossShow", name="pred_bbox_loss"),
    dict(type="LossShow", name="pred_iou_loss"),
    dict(type="LossShow", name="pred_mpb_loss"),
]
per_metric_patterns = [
    dict(label_pattern=None, pred_pattern="total_loss"),
    dict(label_pattern=None, pred_pattern="KLD_loss"),
    dict(label_pattern=None, pred_pattern="goal_loss"),
    dict(label_pattern=None, pred_pattern="pred_loss"),
    dict(label_pattern=None, pred_pattern="goal_bbox_loss"),
    dict(label_pattern=None, pred_pattern="goal_iou_loss"),
    dict(label_pattern=None, pred_pattern="goal_mpb_loss"),
    dict(label_pattern=None, pred_pattern="pred_bbox_loss"),
    dict(label_pattern=None, pred_pattern="pred_iou_loss"),
    dict(label_pattern=None, pred_pattern="pred_mpb_loss"),
]
if k_value > 1:
    loss_metrics.append(dict(type="LossShow", name="prob_loss"))
    per_metric_patterns.append(
        dict(label_pattern=None, pred_pattern="prob_loss")
    )

metric_updater = dict(
    type="MetricUpdater",
    metrics=loss_metrics,
    metric_update_func=update_metric_using_regex(
        per_metric_patterns=per_metric_patterns
    ),
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
)


# --3.2. Validation metric updater.
def update_metric(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


# -- 3.2.1. Setup single or multiple metrics. Multiple metrics is used
#         for evaluate multiple heads or multiple classes respectively.
metrics = [
    dict(
        type="TrajPredFPVPedMetric",
        name=metric_name,
        top_k_values=(1,),
        metrics_type={
            "05": 3,
            "10": 5,
            "15": 8,
        },
    )
]

# -- 3.2.2. a validation metric updater, if you setup multiple metrics,
#         split their respective model outputs in "update_metric" func.
val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=1000,
    epoch_log_freq=1,
    log_prefix=task_name,
)
val_metric_updater["log_prefix"] = "Validation " + task_name

# 4. Setup callbacks.
batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    loss_collector=collect_loss_by_regex("^.*loss.*"),
)

val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
)

val_callback = dict(
    type="Validation",
    data_loader=val_dataloader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=None,
    val_on_train_end=True,
)

stats_callback = dict(
    type="StatsMonitor",
    log_freq=log_freq,
)

lr_update_callback = dict(
    type="CosLrUpdater",
    warmup_by="epoch",
    warmup_len=warmup_epoch,
    step_log_interval=log_freq,
)

# job on aidi platform will have TENSORBOARD_LOG_PATH env,
# where the saved tb can be shown through aidi web UI
tb_loss_callback = dict(
    type="TensorBoard",
    save_dir=os.path.join(tb_dir, "float", "loss"),
    loss_name_reg="^.*loss.*",
    update_freq=10,
    update_by="step",
)

# 5. Trainer and solver.
# -- 5.1. float.
float_model = copy.deepcopy(model)
float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=float_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    pretrain_dir,
                    "exp_sgnet-20221120-122919-float-checkpoint-best.pth.tar",
                ),
                allow_miss=True,
                ignore_extra=True,
            ),
        ],
    ),
    data_loader=train_dataloader,
    optimizer=dict(
        type=torch.optim.Adam,
        lr=float_lr,
        weight_decay=weight_decay,
    ),
    batch_processor=batch_processor,
    num_epochs=float_epoch,
    callbacks=[
        metric_updater,
        lr_update_callback,
        stats_callback,
        val_callback,
        dict(
            type="Checkpoint",
            save_dir=ckpt_dir,
            name_prefix="float-",
            strict_match=True,
            mode="min",
            monitor_metric_key=monitor_metric_key,
        ),
        tb_loss_callback,
    ],
    sync_bn=True,
    train_metrics=metrics,
    val_metrics=metrics,
)

float_predictor = dict(
    type="Predictor",
    model=copy.deepcopy(model),
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=[val_dataloader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=metrics,
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

# -- 5.2. calibration.
# Note: The transforms of the dataset during calibration can be
# consistent with that during training or validation, or customized.
# Default used `val_batch_processor`.
calibration_model = copy.deepcopy(model)
calibration_data_loader = copy.deepcopy(train_dataloader)
calibration_data_loader.pop("sampler")  # Calibration do not support DDP or DP
calibration_batch_processor = copy.deepcopy(val_batch_processor)
calibration_val_callback = copy.deepcopy(val_callback)
calibration_stats_callback = copy.deepcopy(stats_callback)
# calibration_stats_callback["log_freq"] = 1
calibration_trainer = dict(
    type="Calibrator",
    model=calibration_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
            ),
            dict(type="Float2Calibration"),
        ],
    ),
    data_loader=calibration_data_loader,
    batch_processor=calibration_batch_processor,
    # 2. 设置 calibration 迭代的 batch 数目
    num_steps=100,
    callbacks=[
        stats_callback,
        calibration_val_callback,
        dict(
            type="Checkpoint",
            save_dir=ckpt_dir,
            name_prefix="calibration-",
            strict_match=True,
            mode="min",
            monitor_metric_key=monitor_metric_key,
        ),
    ],
    train_metrics=metrics,
    val_metrics=metrics,
)

# -- 5.3. qat.
qat_model = copy.deepcopy(model)
qat_val_callback = copy.deepcopy(val_callback)
qat_val_callback["callbacks"] = [val_metric_updater]
qat_tb_loss_dir = os.path.join(tb_dir, "qat", "loss")
qat_tb_loss_callback = copy.deepcopy(tb_loss_callback)
qat_tb_loss_callback["save_dir"] = qat_tb_loss_dir
qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=qat_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        qconfig_params=dict(
            activation_qat_qkwargs=dict(
                averaging_constant=0.0,  # 0.0 ~ 1.0 之间的浮点数
            ),
            weight_qat_qkwargs=dict(
                averaging_constant=1.0,  # 0.0 ~ 1.0 之间的浮点数
            ),
        ),
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=train_dataloader,
    optimizer=dict(
        type=torch.optim.Adam,
        lr=quanti_lr,
        weight_decay=weight_decay,
    ),
    batch_processor=batch_processor,
    num_epochs=quanti_epoch,
    device=None,
    callbacks=[
        metric_updater,
        lr_update_callback,
        stats_callback,
        qat_val_callback,
        dict(
            type="Checkpoint",
            save_dir=ckpt_dir,
            name_prefix="qat-",
            strict_match=True,
            mode="min",
            monitor_metric_key=monitor_metric_key,
        ),
        qat_tb_loss_callback,
    ],
    train_metrics=metrics,
    val_metrics=metrics,
)

qat_predictor = dict(
    type="Predictor",
    model=copy.deepcopy(model),
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=[val_dataloader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=metrics,
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

# -- 5.4. int.
params_desc = dict(
    k_value=k_value,
    enc_steps=enc_steps,
    dec_steps=dec_steps,
    pred_dim=pred_dim,
    input_dim=input_dim,
)

add_desc_pp = dict(
    type="AddDesc",
    per_tensor_desc=[
        # num of desc == num of pred output tensors(one tensor per out stride)
        json.dumps(
            dict(
                task=f"{task_name}_pred_trajs",
                **params_desc,
            )
        ),
    ],
)

deploy_model = copy.deepcopy(model)
deploy_model["post_process"] = add_desc_pp
deploy_model["losses"] = None
deploy_model["is_int_infer_model"] = True

max_num_obs = 8
deploy_inputs = dict(
    input_x=torch.randn((max_num_obs, input_dim, 1, enc_steps)),
    rand_cvae_seed=torch.normal(
        mu, sigma, size=(max_num_obs, latent_dim, 1, k_value)
    ),
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
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
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
        dict(
            type="Checkpoint",
            save_dir=ckpt_dir,
            name_prefix="int-",
            strict_match=True,
            mode="min",
            monitor_metric_key=monitor_metric_key,
        ),
        dict(
            type="SaveTraced",
            save_dir=ckpt_dir,
            trace_inputs=deploy_inputs,
        ),
    ],
)

int_infer_predictor = dict(
    type="Predictor",
    model=copy.deepcopy(model),
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
            ),
            dict(type="QAT2Quantize"),
        ],
    ),
    data_loader=[val_dataloader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=metrics,
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)
