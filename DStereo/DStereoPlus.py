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
from hat.utils.distributed import get_dist_info
from DStereo.common import disp2rgb, depth2rgb, disp2depth, uncert2rgb

VERSION = ConfigVersion.v2

task_name = "DStereoV23"


training_step = "float"
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
log_freq = 1
wandb_project = "dstereo_vis"
wandb_name = f"{task_name}-{training_step}"
wandb_tags = ""
wandb_resume = None
wandb_run_id = None

wandb_log_every_steps = 1000
wandb_log_samples = True
wandb_samples_per_batch = 2
val_interval = 10000
enable_freeze_bn = False
freeze_bn_affine = False
freeze_bn_until_step = 200000

sync_bn = True
# device_ids = [0,1,2,3,4,5,6,7] # 4卡 【4，5，6，7】
device_ids = [0,]

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
base_lr = 0.0001
# base_lr = 0.001
# num_steps = 200000
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
        infra1=torch.randn((1, 3, 640, 352)),
        infra2=torch.randn((1, 3, 640, 352)),
        # infra1=torch.randn((1, 3, 480, 640)),
        # infra2=torch.randn((1, 3, 480, 640)),
        # infra1=torch.randn((1, 3, 640, 352)),
        # infra2=torch.randn((1, 3, 640, 352)),
    )
)


data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="StereoMultiData",
        test_mode=False,
        dataset_list=[
            "Sceneflow",
            "DStereoDataset",
            "DStereoDataset",
        ],
        aug_args=[0.3, 0.5, 0.0, 0.0],
        res_args=[-1, -1, True],
        norm_args=["MixVarGENet"],
        crop_args=["random", 320, 640],
        debug=False,
        max_disp=maxdisp,
        img_open_mode="bgr",
    ),
    sampler=dict(type="InterleaveConcatSampler", shuffle=True, seed=0),
    batch_size=train_batch_size_per_gpu,
    pin_memory=True,
    prefetch_factor=4,
    shuffle=False,
    num_workers=data_num_workers,
    collate_fn=collate_disp_cat,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="StereoMultiData",
        test_mode=True,
        dataset_list=[
            #"BallCar",
            # "Sceneflow", 
            # "TartanAir",             
            # "IRS",
            # "FallingThings",
            # "SIDODDataset",
            "DStereoDataset",
        ],
        # ballcar_root="/root/ballcar_datasets",
        aug_args=None,
        res_args=[640, 352, False],
        norm_args=["MixVarGENet"],
        crop_args=["center", 640, 352],
        debug=False,
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
wandb_callback = dict(
    type="WandbCallback",
    project=wandb_project,
    name=wandb_name,
    tags=wandb_tags.split(",") if wandb_tags else None,
    resume=wandb_resume,
    run_id=wandb_run_id,
    config=dict(
        task=task_name,
        maxdisp=maxdisp,
        base_lr=base_lr,
        num_steps=num_steps,
        batch_size=train_batch_size_per_gpu,
    ),
    log_every_steps=wandb_log_every_steps,
    log_system_metrics=True,
    log_time_metrics=True,
    log_train_loss=True,
    log_train_subloss=True,
    log_lr=True,
    log_samples=wandb_log_samples,
    samples_per_batch=wandb_samples_per_batch,
    maxdisp=maxdisp,
)


def loss_collector(outputs):
    if isinstance(outputs, dict):
        return outputs.get("losses")
    return None


train_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    enable_amp=False,
    loss_collector=loss_collector,
)
val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    enable_amp=False,
    loss_collector=loss_collector,
)

def _extract_losses(model_outs):
    if not isinstance(model_outs, dict):
        return []
    losses = model_outs.get("losses")
    if losses is not None:
        if isinstance(losses, torch.Tensor):
            return [losses]
        if isinstance(losses, (list, tuple)):
            return [loss for loss in losses if loss is not None]
        return []
    indexed = []
    for k, v in model_outs.items():
        if not isinstance(k, str) or not k.startswith("losses_"):
            continue
        idx = k.split("losses_", 1)[-1]
        if idx.isdigit():
            indexed.append((int(idx), v))
    if not indexed:
        return []
    return [v for _, v in sorted(indexed, key=lambda x: x[0])]


def _named_loss_items(losses):
    if not losses:
        return []
    items = [("init_smooth_l1", losses[0])]
    for idx, loss in enumerate(losses[1:]):
        items.append((f"iter_{idx}_weighted_l1", loss))
    return items


def update_loss_metric(metrics, batch, model_outs):
    labels = batch["gt_disp"]
    masks = (labels > 0) & (labels < maxdisp)
    preds = None
    if isinstance(model_outs, dict):
        losses = model_outs.get("losses")
        if isinstance(losses, torch.Tensor):
            metrics[0].update(losses)
        elif isinstance(losses, (list, tuple)) and len(losses) > 0:
            metrics[0].update(sum(losses))
        preds = model_outs.get("pred_disps")
    elif isinstance(model_outs, (list, tuple)) and len(model_outs) > 0:
        preds = model_outs[0]
    if preds is not None:
        metrics[1].update(labels, preds, masks)



def update_metric(metrics, batch, model_outs):
    labels = batch["gt_disp"]
    preds, disp_4x, init_disp = model_outs
    # preds, init_disp = model_outs
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
    metric_update_func=update_loss_metric,
    step_log_freq=log_freq,
    epoch_log_freq=log_freq,
    log_prefix="val_" + task_name,
)

onnx_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=log_freq,
    epoch_log_freq=log_freq,
    log_prefix="onnx_" + task_name,
)

val_callbacks = [val_metric_updater]
val_callback = dict(
    type="Validation",
    val_interval=val_interval,
    interval_by="step",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=val_callbacks,
    val_model=None,
    val_on_train_end=False,
)
ckpt_callback = dict(
    type="Checkpoint",
    interval_by="step",
    save_interval=val_interval,
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,
    monitor_metric_key="EPE",
)

train_callbacks = [
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
    val_callback,
]
if enable_freeze_bn:
    train_callbacks.insert(
        0,
        dict(
            type="FreezeBN",
            freeze_affine=freeze_bn_affine,
            unfreeze_step=freeze_bn_until_step,
        ),
)
train_callbacks.append(wandb_callback)

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
    callbacks=train_callbacks,
    train_metrics=[
        dict(type="LossShow"),
        dict(
            type="EndPointError",
            use_mask=True,
        ),
    ],
    val_metrics=[
        dict(type="LossShow"),
        dict(
            type="EndPointError",
            use_mask=True,
        ),
    ],
)


predict_callbacks = [
    dict(type="SaveCalibdata", output_dir="ptq_V21/calib_data",),
    # dict(type="SaveDisp", output_dir="work_dirs/disp_result",task_name="disp_result"),
    stat_callback,
    val_metric_updater,
]

# almost train_data_loader, but disable augs
calib_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="StereoMultiData",
        test_mode=False,
        dataset_list=[
            "Sceneflow",
            "DStereoDataset",
        ],
        aug_args=None,
        res_args=[-1, -1, True],
        norm_args=["MixVarGENet"],
        crop_args=["center", 640, 352],
        debug=False,
        max_disp=maxdisp,
        img_open_mode="bgr",
    ),
    sampler=dict(type="InterleaveConcatSampler", shuffle=False, seed=0, sampler_len=100),
    batch_size=test_batch_size_per_gpu,
    pin_memory=True,
    prefetch_factor=4,
    shuffle=False,
    num_workers=data_num_workers,
    collate_fn=collate_disp_cat,
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
    data_loader=[calib_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(type="LossShow"),
        dict(
            type="EndPointError",
            use_mask=True,
        ),
    ],
    callbacks=predict_callbacks,
    log_interval=log_freq,
)

quantonnx_predictor = dict(
    type="Predictor",
    model=dict(
        type="OnnxStereoModel",
        onnx_path="ptq_V21/Bin_model/DStereo_quantized_model.onnx",
    ),
    data_loader=[calib_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(
            type="EndPointError",
            use_mask=True,
        ),
    ],
    callbacks=[
        onnx_metric_updater,
        stat_callback,
        dict(
            type="SaveDisp",
            output_dir="ptq_V21/vis/quant",
            task_name="onnx_disp",
            maxdisp=maxdisp,
        ),
    ],
    log_interval=log_freq,
)

floatonnx_predictor = dict(
    type="Predictor",
    model=dict(
        type="OnnxStereoModel",
        onnx_path="ptq_V21/float.onnx",
    ),
    data_loader=[calib_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(
            type="EndPointError",
            use_mask=True,
        ),
    ],
    callbacks=[
        onnx_metric_updater,
        stat_callback,
        dict(
            type="SaveDisp",
            output_dir="ptq_V21/vis/float",
            task_name="onnx_disp",
            maxdisp=maxdisp,
        ),
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
