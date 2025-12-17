import copy
import os

import torch
from horizon_plugin_pytorch.march import March

from hat.engine.processors.loss_collector import collect_loss_by_index
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "sndr_efficientnet_lite_v3"
batch_size_per_gpu = 128
device_ids = [0]
ckpt_dir = "./tmp_models/%s/ckpt/" % task_name
log_dir = "./tmp_models/%s/log/" % task_name
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.BAYES
convert_mode = "fx"
img_h, img_w = (128, 128)
bn_kwargs = dict(eps=1e-5, momentum=0.1)


# SNDREfficientnet_lite for ImageNet 128.
# basic level latency (including head): 3240us.
# the head latency: ~271us
# backbone: ~2969us
# Params: 4.70M
# FLOPS: 126.6M * 2
# --------------------------------------------------------
Block_v1 = (
    dict(
        kernel_size=3,
        num_repeat=1,
        in_filters=32,
        out_filters=16,
        expand_ratio=1,
        id_skip=True,
        strides=1,
        se_ratio=0.25,
    ),
    dict(
        kernel_size=3,
        num_repeat=2,
        in_filters=16,
        out_filters=24,
        expand_ratio=6,
        id_skip=True,
        strides=2,
        se_ratio=0.25,
    ),
    dict(
        kernel_size=5,
        num_repeat=2,
        in_filters=24,
        out_filters=40,
        expand_ratio=6,
        id_skip=True,
        strides=2,
        se_ratio=0.25,
    ),
    dict(
        kernel_size=3,
        num_repeat=3,
        in_filters=40,
        out_filters=80,
        expand_ratio=6,
        id_skip=True,
        strides=2,
        se_ratio=0.25,
    ),
    dict(
        kernel_size=5,
        num_repeat=3,
        in_filters=80,
        out_filters=112,
        expand_ratio=6,
        id_skip=True,
        strides=1,
        se_ratio=0.25,
    ),
    dict(
        kernel_size=5,
        num_repeat=4,
        in_filters=112,
        out_filters=192,
        expand_ratio=6,
        id_skip=True,
        strides=2,
        se_ratio=0.25,
    ),
    dict(
        kernel_size=3,
        num_repeat=1,
        in_filters=192,
        out_filters=320,
        expand_ratio=6,
        id_skip=True,
        strides=1,
        se_ratio=0.25,
    ),
)

# efficientnet_lite0
model_v1 = dict(
    type="Classifier",
    backbone=dict(
        type="SNDREfficientnet",
        num_classes=1000,
        coefficient_params=(1.0, 1.0, img_h, 0.2),
        blocks_args=Block_v1,
        model_type="lite",
        activation="relu",
        use_se_block=False,
        bn_kwargs=bn_kwargs,
        include_top=True,
    ),
    losses=dict(type="CEWithLabelSmooth"),
)
# --------------------------------------------------------

# Less Units on ImageNet 128.
# modify the number of channels
# if the input_size is 128, reduce the number of downsamples to ues HTA
# basic level latency (including head): 3571us.
# the head latency: ~301us
# the backbone latency: ~3271us
# Params: 4.36M
# FLOPS: 49.8M * 2
# --------------------------------------------------------
Block_v2 = (
    dict(
        kernel_size=3,
        num_repeat=1,
        in_filters=32,
        out_filters=32,
        expand_ratio=1,
        id_skip=True,
        strides=1,
        se_ratio=0.25,
    ),
    dict(
        kernel_size=3,
        num_repeat=2,
        in_filters=32,
        out_filters=64,
        expand_ratio=6,
        id_skip=True,
        strides=2,
        se_ratio=0.25,
    ),
    dict(
        kernel_size=3,
        num_repeat=2,
        in_filters=64,
        out_filters=128,
        expand_ratio=6,
        id_skip=True,
        strides=2,
        se_ratio=0.25,
    ),
    dict(
        kernel_size=3,
        num_repeat=2,
        in_filters=128,
        out_filters=160,
        expand_ratio=6,
        id_skip=True,
        strides=1,
        se_ratio=0.25,
    ),
    dict(
        kernel_size=3,
        num_repeat=3,
        in_filters=160,
        out_filters=192,
        expand_ratio=6,
        id_skip=True,
        strides=2,
        se_ratio=0.25,
    ),
    dict(
        kernel_size=3,
        num_repeat=1,
        in_filters=192,
        out_filters=256,
        expand_ratio=6,
        id_skip=True,
        strides=1,
        se_ratio=0.25,
    ),
)

model_v2 = dict(
    type="Classifier",
    backbone=dict(
        type="SNDREfficientnet",
        num_classes=1000,
        coefficient_params=(1.0, 1.0, img_h, 0.2),
        blocks_args=Block_v2,
        model_type="lite",
        activation="relu",
        downsample_times=4,
        use_se_block=False,
        bn_kwargs=bn_kwargs,
        include_top=True,
    ),
    losses=dict(type="CEWithLabelSmooth"),
)
# --------------------------------------------------------

# Less channels and units on ImageNet 128.
# modify the number of channels
# if the input_size is 128, reduce the number of downsamples to ues HTA
# basic level latency (including head): 2020us.
# the head latency: ~257us
# the backbone latency: ~1723us
# Params: 2.08M
# FLOPS: 12.2M * 2
# --------------------------------------------------------
Block_v3 = (
    dict(
        kernel_size=3,
        num_repeat=1,
        in_filters=32,
        out_filters=32,
        expand_ratio=1,
        id_skip=True,
        strides=1,
        se_ratio=0.25,
    ),
    dict(
        kernel_size=3,
        num_repeat=2,
        in_filters=32,
        out_filters=64,
        expand_ratio=2,
        id_skip=True,
        strides=2,
        se_ratio=0.25,
    ),
    dict(
        kernel_size=3,
        num_repeat=2,
        in_filters=64,
        out_filters=96,
        expand_ratio=2,
        id_skip=True,
        strides=2,
        se_ratio=0.25,
    ),
    dict(
        kernel_size=3,
        num_repeat=1,
        in_filters=96,
        out_filters=128,
        expand_ratio=2,
        id_skip=True,
        strides=1,
        se_ratio=0.25,
    ),
    dict(
        kernel_size=3,
        num_repeat=2,
        in_filters=128,
        out_filters=160,
        expand_ratio=2,
        id_skip=True,
        strides=2,
        se_ratio=0.25,
    ),
    dict(
        kernel_size=3,
        num_repeat=1,
        in_filters=160,
        out_filters=256,
        expand_ratio=2,
        id_skip=True,
        strides=1,
        se_ratio=0.25,
    ),
)

model_v3 = dict(
    type="Classifier",
    backbone=dict(
        type="SNDREfficientnet",
        num_classes=1000,
        coefficient_params=(1.0, 1.0, img_h, 0.2),
        blocks_args=Block_v3,
        model_type="lite",
        activation="relu",
        downsample_times=4,
        use_se_block=False,
        bn_kwargs=bn_kwargs,
        include_top=True,
    ),
    losses=dict(type="CEWithLabelSmooth"),
)


# Choose model
model = model_v3
# Deploy model.
deploy_model = copy.deepcopy(model)
deploy_model["backbone"]["flat_output"] = False
deploy_model["losses"] = None
deploy_inputs = dict(img=torch.randn((1, 3, 128, 128)))


deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2QAT", convert_mode=convert_mode),
        dict(type="QAT2Quantize", convert_mode=convert_mode),
    ],
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ImageNet",
        data_path="./tmp_data/imagenet/train_lmdb/",
        transforms=[
            dict(
                type="TorchVisionAdapter",
                interface="RandomResizedCrop",
                size=128,
                scale=(0.08, 1.0),
                ratio=(3.0 / 4.0, 4.0 / 3.0),
            ),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=5,
    pin_memory=True,
)
val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ImageNet",
        data_path="./tmp_data/imagenet/val_lmdb/",
        transforms=[
            dict(type="TorchVisionAdapter", interface="Resize", size=144),
            dict(type="TorchVisionAdapter", interface="CenterCrop", size=128),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=5,
    pin_memory=True,
)

batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    batch_transforms=[
        dict(type="TorchVisionAdapter", interface="RandomHorizontalFlip"),
        dict(
            type="TorchVisionAdapter",
            interface="ColorJitter",
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
        ),
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
    loss_collector=collect_loss_by_index(1),
)
val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    batch_transforms=[
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
    loss_collector=None,
)


def update_metric(metrics, batch, model_outs):
    target = batch["labels"]
    preds, losses = model_outs
    for metric in metrics:
        metric.update(target, preds)


metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=1000,
    epoch_log_freq=1,
    log_prefix=task_name,
)
val_metric_updater = copy.deepcopy(metric_updater)
val_metric_updater["log_prefix"] = "Validation " + task_name

stat_callback = dict(
    type="StatsMonitor",
    log_freq=1000,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,
    mode="max",
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=None,
    val_on_train_end=False,
)

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=4e-5)},
        lr=0.4,
        momentum=0.9,
    ),
    batch_processor=batch_processor,
    num_epochs=240,
    num_steps=2,
    stop_by="step",
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="CosLrUpdater",
            warmup_by="epoch",
            warmup_len=5,
            step_log_interval=1000,
        ),
        metric_updater,
        # val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
    val_metrics=[
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
)

# Note: The transforms of the dataset during calibration can be
# consistent with that during training or validation, or customized.
# Default used `val_batch_processor`.
calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop("sampler")  # Calibration do not support DDP or DP
calibration_data_loader["batch_size"] = batch_size_per_gpu * 2
calibration_batch_processor = copy.deepcopy(val_batch_processor)
calibration_step = 100

calibration_trainer = dict(
    type="Calibrator",
    model=model,
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
            dict(type="Float2Calibration", convert_mode=convert_mode),
        ],
    ),
    data_loader=calibration_data_loader,
    batch_processor=calibration_batch_processor,
    num_steps=calibration_step,
    device=None,
    callbacks=[
        stat_callback,
        val_callback,
        ckpt_callback,
    ],
    val_metrics=[
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
    log_interval=calibration_step / 10,
)

qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        qconfig_params=dict(
            activation_qat_qkwargs=dict(
                averaging_constant=0,
            ),
            weight_qat_qkwargs=dict(
                qscheme=torch.per_channel_symmetric,
                ch_axis=0,
                averaging_constant=1,
            ),
        ),
        converters=[
            dict(type="Float2QAT", convert_mode=convert_mode),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=4e-5)},
        lr=0.001,
        momentum=0.9,
    ),
    batch_processor=batch_processor,
    num_epochs=30,
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            step_log_interval=1000,
            lr_decay_id=[15, 25],
            lr_decay_factor=0.1,
        ),
        metric_updater,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
    val_metrics=[
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
)

# just for saving int_infer pth and pt
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
            dict(type="QAT2Quantize", convert_mode=convert_mode),
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

compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=task_name,
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["pyramid"],
)

# predictor
float_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-last.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

calibration_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT", convert_mode=convert_mode),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-best.pth.tar"
                ),
                verbose=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=100,
)

qat_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        qconfig_params=dict(
            activation_qkwargs=dict(
                averaging_constant=0,
            ),
            weight_qkwargs=dict(
                averaging_constant=1,
            ),
        ),
        converters=[
            dict(type="Float2QAT", convert_mode=convert_mode),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

int_infer_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT", convert_mode=convert_mode),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
            ),
            dict(type="QAT2Quantize", convert_mode=convert_mode),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

# onnx
onnx_cfg = dict(
    model=copy.deepcopy(deploy_model),
    stage="float",
    inputs=deploy_inputs,
    model_convert_pipeline=float_predictor["model_convert_pipeline"],
    kwargs=dict(
        # verbose=True,
        opset_version=11,  # 需要注意 snpe 1.51需要指定 onnx op 版本
    ),
)
