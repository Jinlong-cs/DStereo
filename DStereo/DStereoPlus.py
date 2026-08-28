import copy
import os

from DStereo.common import depth2rgb, disp2depth, disp2rgb, uncert2rgb

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from horizon_plugin_pytorch.march import March
from PIL import Image
from torch.utils.data.distributed import DistributedSampler

from hat.data.collates.collates import collate_disp_cat
from hat.data.datasets.multi_disp_dataset.resize_aware import (
    DEFAULT_RESIZE_SCALES,
    build_resize_aware_specs,
    resize_aware_config,
)
from hat.data.samplers.interleave_concat_sampler import InterleaveConcatSampler
from hat.metrics.loss_show import LossShow
from hat.models.backbones.mixvargenet import MixVarGENetConfig
from hat.utils.config import ConfigVersion
from hat.utils.distributed import get_dist_info

VERSION = ConfigVersion.v2

task_name = "DStereoV23_DiscoverStereo"


training_stage = "float"
data_num_workers = 4
march = March.BAYES_E
# ckpt_dir = "work_dirs/ckpt_models/%s" % task_name
ckpt_dir = "work_dirs/discover_experiments/%s" % task_name
checkpoint_path = (
    "tmp_pretrained_models/mixvargenet_imagenet/float-checkpoint-last.pth.tar"
)
local_train = not os.path.exists("/running_package")
train_batch_size_per_gpu = 8
test_batch_size_per_gpu = 1
log_freq = 1
wandb_project = "DStereo-DiscoverStereo"
wandb_name = f"{task_name}-{training_stage}"
wandb_tags = "discover,150k,preproc352x640,maxdisp96"
wandb_resume = None
wandb_run_id = None

wandb_log_every_steps = 1000
wandb_log_samples = True
wandb_samples_per_batch = 2
val_interval = 5000
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
        # NCHW
        # Vertical
        # infra1=torch.randn((1, 3, 640, 352)),
        # infra2=torch.randn((1, 3, 640, 352)),
        # Horizontal
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
            # "Sceneflow",
            "DStereoDataset",
            "DStereoDataset",

        ],
        aug_args=[0.3, 0.5, 0.0, 0.0],
        res_args=[-1, -1, True],
        norm_args=["MixVarGENet"],
        # crop_args=["random", 640, 352],  # vertical
        crop_args=["random", 352, 640],    # horizontal
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
        aug_args=None,
        # res_args=[640, 352, False], # vertical
        res_args=[352, 640, False],   # horizontal
        norm_args=["MixVarGENet"],
        # crop_args=["center", 640, 352],  # vertical
        crop_args=["center", 352, 640],  # horizontal
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
    log_freq=500,
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
    log_time_metrics=True,
    log_train_loss=True,
    log_train_subloss=True,
    log_lr=True,
    log_samples=wandb_log_samples,
    samples_per_batch=wandb_samples_per_batch,
    maxdisp=maxdisp,
    log_checkpoints=True,
    ckpt_dir=ckpt_dir,
    ckpt_name_prefix=training_stage + "-",
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

val_callback = dict(
    type="Validation",
    interval_by="step",
    val_interval=val_interval,
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=None,
    val_on_train_end=False,
)

ckpt_callback = dict(
    type="Checkpoint",
    interval_by="step",
    save_interval=val_interval,
    save_dir=ckpt_dir,
    name_prefix=training_stage + "-",
    strict_match=True,
    mode="min",
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
    val_callback,
    ckpt_callback,
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


class _StereoScaleSampler(DistributedSampler):
    """Attach one resize scale to every sample in a global batch."""

    def __init__(
        self,
        dataset,
        batch_size,
        scales,
        shuffle=True,
        seed=0,
        num_replicas=None,
        rank=None,
    ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=False,
            seed=seed,
            drop_last=True,
        )
        self.interleave = InterleaveConcatSampler(
            dataset,
            shuffle=shuffle,
            seed=seed,
        )
        self.batch_size = batch_size
        self.scales = tuple(scales)
        global_batch_size = batch_size * self.num_replicas
        self.num_batches = len(self.interleave) // global_batch_size
        self.num_batches -= self.num_batches % len(self.scales)
        self.num_samples = self.num_batches * batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        self.interleave.set_epoch(self.epoch)
        stream = list(self.interleave)
        global_batch_size = self.batch_size * self.num_replicas
        for batch_index in range(self.num_batches):
            start = batch_index * global_batch_size
            global_indices = stream[start : start + global_batch_size]
            rank_start = self.rank * self.batch_size
            rank_indices = global_indices[
                rank_start : rank_start + self.batch_size
            ]
            scale = self.scales[batch_index % len(self.scales)]
            yield from ((index, scale) for index in rank_indices)


float_resize_scales = DEFAULT_RESIZE_SCALES
float_resize_args = resize_aware_config(
    float_resize_scales,
    max_disp=maxdisp,
)
float_s100_spec, float_s080_spec = build_resize_aware_specs(
    float_resize_scales
)

float_data_loader = copy.deepcopy(data_loader)
float_data_loader["dataset"].update(
    res_args=[352, 640, False],
    resize_aware_args=float_resize_args,
)
float_data_loader["sampler"] = dict(
    type=_StereoScaleSampler,
    batch_size=train_batch_size_per_gpu,
    scales=float_resize_scales,
    shuffle=True,
    seed=seed,
)
float_data_loader["drop_last"] = True


def _build_float_val_loader(scale):
    loader = copy.deepcopy(val_data_loader)
    loader["dataset"].update(
        resize_aware_args=resize_aware_config(
            (scale,),
            max_disp=maxdisp,
        ),
    )
    return loader


float_val_data_loader_s100 = _build_float_val_loader(1.0)
float_val_data_loader_s080 = _build_float_val_loader(0.8)
float_val_data_loader = [
    float_val_data_loader_s100,
    float_val_data_loader_s080,
]


def _restore_float_prediction(prediction, spec):
    height_end = spec.tensor_height - spec.pad_bottom
    width_end = spec.tensor_width - spec.pad_right
    content = prediction[
        ...,
        spec.pad_top : height_end,
        spec.pad_left : width_end,
    ]
    canonical = F.interpolate(
        content,
        size=(spec.base_height, spec.base_width),
        mode="bilinear",
        align_corners=False,
    )
    return canonical / spec.horizontal_scale


def _update_float_val_metric(metrics, batch, model_outs, spec):
    labels = batch["metric_gt_disp"]
    predictions = _restore_float_prediction(model_outs[0], spec)
    masks = (labels > 0) & (labels < maxdisp)
    metrics[0].update(labels, predictions, masks)


def update_float_s100_val_metric(metrics, batch, model_outs):
    _update_float_val_metric(metrics, batch, model_outs, float_s100_spec)


def update_float_s080_val_metric(metrics, batch, model_outs):
    _update_float_val_metric(metrics, batch, model_outs, float_s080_spec)


float_val_metric_s100 = dict(
    type="EndPointError",
    name="EPE_s100",
    use_mask=True,
)
float_val_metric_s080 = dict(
    type="EndPointError",
    name="EPE_s080",
    use_mask=True,
)
float_val_metrics = [float_val_metric_s100, float_val_metric_s080]

float_val_metric_updater_s100 = copy.deepcopy(val_metric_updater)
float_val_metric_updater_s100.update(
    metric_update_func=update_float_s100_val_metric,
    metrics=[float_val_metric_s100],
    log_prefix="val_s100_" + task_name,
)
float_val_metric_updater_s080 = copy.deepcopy(val_metric_updater)
float_val_metric_updater_s080.update(
    metric_update_func=update_float_s080_val_metric,
    metrics=[float_val_metric_s080],
    log_prefix="val_s080_" + task_name,
)

float_val_callback = copy.deepcopy(val_callback)
float_val_callback.update(
    data_loader=float_val_data_loader,
    callbacks=[
        [float_val_metric_updater_s100],
        [float_val_metric_updater_s080],
    ],
    share_callbacks=False,
)

float_ckpt_dir = os.environ.get("DSTEREO_RUN_DIR", ckpt_dir)
float_ckpt_callback = copy.deepcopy(ckpt_callback)
float_ckpt_callback.update(
    save_dir=float_ckpt_dir,
    monitor_metric_key="EPE_s100",
)

float_wandb_callback = copy.deepcopy(wandb_callback)
float_wandb_callback.update(
    name=f"{task_name}_ResizeAware_S100_S080-{training_stage}",
    tags=(
        "discover,v2,resize-aware,dual-scale,scale1.00,scale0.80,"
        "canonical640x352,tensor512x288,maxdisp96,200k"
    ).split(","),
    ckpt_dir=float_ckpt_dir,
)
float_wandb_callback["config"].update(
    resize_aware_scales=list(float_resize_scales),
    resize_base_shape=[352, 640],
    resize_content_shapes={
        "s100": list(float_s100_spec.content_shape),
        "s080": list(float_s080_spec.content_shape),
    },
    resize_tensor_shapes={
        "s100": list(float_s100_spec.tensor_shape),
        "s080": list(float_s080_spec.tensor_shape),
    },
    validation_metrics=["val/epe_s100", "val/epe_s080"],
    checkpoint_monitor="EPE_s100",
    resize_size_divisor=float_s100_spec.size_divisor,
)

float_train_callbacks = copy.deepcopy(train_callbacks)
float_train_callbacks[-3:] = [
    float_val_callback,
    float_ckpt_callback,
    float_wandb_callback,
]

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=float_data_loader,
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
                allow_miss=False,
                ignore_extra=False,
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
    callbacks=float_train_callbacks,
    train_metrics=[
        dict(type="LossShow"),
        dict(
            type="EndPointError",
            use_mask=True,
        ),
    ],
    val_metrics=float_val_metrics,
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
            # "Sceneflow",
            "DStereoDataset",
        ],
        aug_args=None,
        res_args=[-1, -1, True],
        norm_args=["MixVarGENet"],
        crop_args=["center", 352, 640],  # horizontal
        debug=False,
        max_disp=maxdisp,
        img_open_mode="bgr",
    ),
    sampler=dict(type="InterleaveConcatSampler", shuffle=True, seed=666, sampler_len=50),
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
                    ckpt_dir, "float-checkpoint-best.pth.tar",
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
            type="SaveDispInfer",
            output_dir="ptq_V21/vis/quant",
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
            type="SaveDispInfer",
            output_dir="ptq_V21/vis/float",
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
                checkpoint_path=os.path.join(ckpt_dir, "float-checkpoint-best.pth.tar"),
                verbose=False,
                allow_miss=False,
            ),
        ],
    ),
)
