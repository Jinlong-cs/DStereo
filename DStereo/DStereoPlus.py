import copy
import os
import cv2
import numpy as np
import torch
from horizon_plugin_pytorch.march import March
from PIL import Image
from hat.data.collates.collates import collate_disp_cat
from hat.models.backbones.mixvargenet import MixVarGENetConfig
from hat.utils.config import ConfigVersion
from hat.metrics.loss_show import LossShow
from DStereo.common import disp2rgb, depth2rgb, disp2depth, uncert2rgb

VERSION = ConfigVersion.v2

training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "DStereoV23"
data_num_workers = 4
march = March.BAYES_E
# ckpt_dir = "work_dirs/ckpt_models/%s" % task_name
ckpt_dir = "work_dirs/tmp_models_szp1/%s" % task_name
checkpoint_path = (
    "tmp_pretrained_models/mixvargenet_imagenet/float-checkpoint-last.pth.tar"
)
local_train = not os.path.exists("/running_package")
train_batch_size_per_gpu = 8
test_batch_size_per_gpu = 1
log_freq = 10
sync_bn = True
# device_ids = [0,1,2,3,4,5,6,7] # 4卡 【4，5，6，7】
device_ids = [2,]

cudnn_benchmark = True
seed = 666
log_rank_zero_only = True
convert_mode = "fx"

loss_weights = [0.0, 0.0, 0.0, 1.0]

# maxdisp = 192
maxdisp = 96
bias = False
bn_kwargs = {}
refine_levels = 3
# base_lr = 0.0001
base_lr = 0.001
# num_steps = 100000
num_steps = 200000
model = dict(
    type="DStereoPlus",
    maxdisp=maxdisp,
    gru_iters=2,
    backbone=dict(
        type="MixVarGENet",
        net_config=[
            [
                MixVarGENetConfig(
                    in_channels=32,
                    out_channels=32,
                    head_op="mixvarge_f2",
                    stack_ops=[],
                    stride=1,
                    stack_factor=1,
                    fusion_strides=[],
                    extra_downsample_num=0,
                )
            ],
            [
                MixVarGENetConfig(
                    in_channels=32,
                    out_channels=32,
                    head_op="mixvarge_f4",
                    stack_ops=["mixvarge_f4", "mixvarge_f4"],
                    stride=2,
                    stack_factor=1,
                    fusion_strides=[],
                    extra_downsample_num=0,
                )
            ],
            [
                MixVarGENetConfig(
                    in_channels=32,
                    out_channels=64,
                    head_op="mixvarge_f4",
                    stack_ops=["mixvarge_f4", "mixvarge_f4"],
                    stride=2,
                    stack_factor=1,
                    fusion_strides=[],
                    extra_downsample_num=0,
                )
            ],
            [
                MixVarGENetConfig(
                    in_channels=64,
                    out_channels=96,
                    head_op="mixvarge_f2_gb16",
                    stack_ops=[
                        "mixvarge_f2_gb16",
                        "mixvarge_f2_gb16",
                        "mixvarge_f2_gb16",
                        "mixvarge_f2_gb16",
                        "mixvarge_f2_gb16",
                        "mixvarge_f2_gb16",
                    ],
                    stride=2,
                    stack_factor=1,
                    fusion_strides=[],
                    extra_downsample_num=0,
                )
            ],
            [
                MixVarGENetConfig(
                    in_channels=96,
                    out_channels=160,
                    head_op="mixvarge_f2_gb16",
                    stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
                    stride=2,
                    stack_factor=1,
                    fusion_strides=[],
                    extra_downsample_num=0,
                )
            ],
        ],
        disable_quanti_input=False,
        input_channels=3,
        input_sequence_length=1,
        num_classes=1000,
        bn_kwargs=bn_kwargs,
        include_top=False,
        bias=True,
        output_list=[0, 1, 2, 3, 4],
    ),
)

deploy_model = model
deploy_inputs = dict(
    data=dict(
        infra1=torch.randn((1, 3, 352, 640)),
        infra2=torch.randn((1, 3, 352, 640)),
    )
)


data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="StereoMultiData",
        test_mode=False,
        dataset_list=[
            "Sceneflow",
            # "TartanAir",             
            # "IRS",
            # "FallingThings",
            # "SIDODDataset",
        ],
        aug_args=[0.3, 0.5, 0.0, 0.0],
        res_args=[-1, -1, True],
        norm_args=["MixVarGENet"],
        crop_args=["random", 320, 640],
        debug=False,
        max_disp=maxdisp,
        img_open_mode="bgr",
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=train_batch_size_per_gpu,
    pin_memory=True,
    prefetch_factor=4,
    shuffle=True,
    num_workers=data_num_workers,
    collate_fn=collate_disp_cat,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="StereoMultiData",
        test_mode=True,
        dataset_list=[
            "Sceneflow", 
            # "TartanAir",             
            # "IRS",
            # "FallingThings",
            # "SIDODDataset",
        ],
        aug_args=None,
        res_args=[352, 640, False],
        norm_args=["MixVarGENet"],
        crop_args=["center", 352, 640],
        debug=True,
        max_disp=maxdisp,
        img_open_mode="bgr",
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=test_batch_size_per_gpu,
    pin_memory=True,
    shuffle=False,
    num_workers=data_num_workers,
    collate_fn=collate_disp_cat,
)


stat_callback = dict(
    type="StatsMonitor",
    log_freq=1000,
)


def loss_collector(outputs: dict):
    return outputs["losses"]


train_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    enable_amp=False,
    loss_collector=loss_collector,
)
val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
    loss_collector=None,
    enable_amp=False,
)

tensorboard_callback = dict(
    type="TensorBoard",
    save_dir=(
        os.path.join(ckpt_dir, training_step) if local_train else "/job_tboard/"
    ),  
    update_freq=log_freq,
    update_by="step",
    tb_update_funcs=None,
)


def update_loss_metric(metrics, batch, model_outs):
    loss = sum(model_outs["losses"])
    metrics[0].update(loss)
    labels = batch["gt_disp"]
    preds = model_outs["pred_disps"]
    masks = (labels > 0) & (labels < maxdisp)
    metrics[1].update(labels, preds, masks)


def update_metric(metrics, batch, model_outs):
    labels = batch["gt_disp"]
    preds, disp_4x, init_disp = model_outs
    masks = (labels > 0) & (labels < maxdisp)
    metrics[0].update(labels, preds, masks)


loss_show_callback = dict(
    type="MetricUpdater",
    metric_update_func=update_loss_metric,
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix="train_" + task_name,
)

val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=log_freq,
    epoch_log_freq=log_freq,
    log_prefix="Validation_" + task_name,
)

val_callback = dict(
    type="Validation",
    val_interval=10000,
    interval_by="step",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=None,
    val_on_train_end=False,
)
ckpt_callback = dict(
    type="Checkpoint",
    interval_by="step",
    save_interval=10000,
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,
    monitor_metric_key="EPE",
)

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        params={"weight": dict(weight_decay=4e-5)},
        lr=base_lr,
    ),
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=checkpoint_path,
                allow_miss=True,
                ignore_extra=True,
                ignore_tensor_shape=False,
                verbose=True,
            ),
        ],
    ),
    resume_optimizer=False,
    resume_epoch_or_step=False,
    resume_dataloader=False,
    batch_processor=train_batch_processor,
    stop_by="step",
    num_steps=num_steps,
    device=None,
    sync_bn=sync_bn,
    callbacks=[
        stat_callback,
        loss_show_callback,
        dict(
            type="CosLrUpdater",
            max_steps=num_steps,
            warmup_by="step",
            warmup_len=2000,
            step_log_interval=1000,
        ),
        ckpt_callback,
        tensorboard_callback,
    ],
    train_metrics=[
        dict(type="LossShow"),
        dict(
            type="EndPointError",
            use_mask=True,
        ),
    ],
    val_metrics=[
        dict(
            type="EndPointError",
            use_mask=True,
        ),
    ],
)

float_predictor = dict(
    type="Predictor",
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
                ignore_extra=False,
                verbose=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(
            type="EndPointError",
            use_mask=True,
        ),
    ],
    callbacks=[
        dict(type="SaveCalibdata", output_dir="ptq_V21/calib_data",),
        # dict(type="SaveDisp", output_dir="work_dirs/disp_result",task_name="disp_result"),
        stat_callback,
        val_metric_updater,
    ],
    log_interval=log_freq,
)

onnx_cfg = dict(
    model=deploy_model,
    stage="float",
    inputs=deploy_inputs,
    kwargs=dict(
        verbose=False,
        opset_version=11,
        input_names=["infra1", "infra2"],
        output_names=["disp", "spx", "initdisp", "initspx"],
    ),
    out_dir="ptq_V21",
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(ckpt_dir, "float-checkpoint-last.pth.tar"),
                verbose=False,
                allow_miss=False,
            ),
        ],
    ),
)
