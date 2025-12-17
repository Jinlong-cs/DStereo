import copy
import os

import torch
from horizon_plugin_pytorch.march import March

from hat.engine.processors.loss_collector import collect_loss_by_index
from hat.models.backbones.mixvargenet import MixVarGENetConfig
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "mixvargenet_cls_id6"
num_classes = 1000
batch_size_per_gpu = 64
device_ids = [3]
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.BAYES
convert_mode = "fx"

bn_kwargs = dict(eps=1e-5, momentum=0.1)

# basic mixvargenet settings on ImageNet128.
# basic level latency (including head): 2262us.
# the head latency: ~252us
# backbone: ~2010us
# FLOPS: 355M * 2
channel_list = [32, 32, 32, 64, 64, 96]
net_config_v1 = [
    [
        MixVarGENetConfig(
            in_channels=channel_list[0],
            out_channels=channel_list[1],
            head_op="mixvarge_f2",
            stack_ops=[],
            stack_factor=1,
            stride=1,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 2
    [
        MixVarGENetConfig(
            in_channels=channel_list[1],
            out_channels=channel_list[2],
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 4
    [
        MixVarGENetConfig(
            in_channels=channel_list[2],
            out_channels=channel_list[3],
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 8
    [
        MixVarGENetConfig(
            in_channels=channel_list[3],
            out_channels=channel_list[4],
            head_op="mixvarge_f2",
            stack_ops=[
                "mixvarge_f2",
                "mixvarge_f2",
                "mixvarge_f2",
                "mixvarge_f2",
                "mixvarge_f2",
                "mixvarge_f2",
            ],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 16
    [
        MixVarGENetConfig(
            in_channels=channel_list[4],
            out_channels=channel_list[5],
            head_op="mixvarge_f2",
            stack_ops=["mixvarge_f2", "mixvarge_f2"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 32
]

# less units on ImageNet128.
# basic level latency (including head): 2199us.
# the head latency: ~252us
# backbone: ~1947us
# FLOPS: 305M * 2
channel_list = [32, 32, 32, 64, 64, 96]
net_config_v2 = [
    [
        MixVarGENetConfig(
            in_channels=channel_list[0],
            out_channels=channel_list[1],
            head_op="mixvarge_f2",
            stack_ops=[],
            stack_factor=1,
            stride=1,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 2
    [
        MixVarGENetConfig(
            in_channels=channel_list[1],
            out_channels=channel_list[2],
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 4
    [
        MixVarGENetConfig(
            in_channels=channel_list[2],
            out_channels=channel_list[3],
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 8
    [
        MixVarGENetConfig(
            in_channels=channel_list[3],
            out_channels=channel_list[4],
            head_op="mixvarge_f2",
            stack_ops=[
                "mixvarge_f2",
                "mixvarge_f2",
                "mixvarge_f2",
                "mixvarge_f2",
            ],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 16
    [
        MixVarGENetConfig(
            in_channels=channel_list[4],
            out_channels=channel_list[5],
            head_op="mixvarge_f2",
            stack_ops=["mixvarge_f2", "mixvarge_f2"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 32
]

# less channels on ImageNet128.
# basic level latency (including head): 2139us.
# the head latency: ~252us
# backbone: ~1887us
# FLOPS: 289M * 2
channel_list = [32, 32, 32, 32, 64, 96]
net_config_v3 = [
    [
        MixVarGENetConfig(
            in_channels=channel_list[0],
            out_channels=channel_list[1],
            head_op="mixvarge_f2",
            stack_ops=[],
            stack_factor=1,
            stride=1,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 2
    [
        MixVarGENetConfig(
            in_channels=channel_list[1],
            out_channels=channel_list[2],
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 4
    [
        MixVarGENetConfig(
            in_channels=channel_list[2],
            out_channels=channel_list[3],
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 8
    [
        MixVarGENetConfig(
            in_channels=channel_list[3],
            out_channels=channel_list[4],
            head_op="mixvarge_f2",
            stack_ops=[
                "mixvarge_f2",
                "mixvarge_f2",
                "mixvarge_f2",
                "mixvarge_f2",
                "mixvarge_f2",
                "mixvarge_f2",
            ],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 16
    [
        MixVarGENetConfig(
            in_channels=channel_list[4],
            out_channels=channel_list[5],
            head_op="mixvarge_f2",
            stack_ops=["mixvarge_f2", "mixvarge_f2"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 32
]

# less channels and units on ImageNet128.
# basic level latency (including head): 2019us.
# the head latency: ~252us
# backbone: ~1767us
# FLOPS: 237M * 2
channel_list = [32, 32, 32, 32, 64, 96]
net_config_v4 = [
    [
        MixVarGENetConfig(
            in_channels=channel_list[0],
            out_channels=channel_list[1],
            head_op="mixvarge_f2",
            stack_ops=[],
            stack_factor=1,
            stride=1,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 2
    [
        MixVarGENetConfig(
            in_channels=channel_list[1],
            out_channels=channel_list[2],
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 4
    [
        MixVarGENetConfig(
            in_channels=channel_list[2],
            out_channels=channel_list[3],
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 8
    [
        MixVarGENetConfig(
            in_channels=channel_list[3],
            out_channels=channel_list[4],
            head_op="mixvarge_f2",
            stack_ops=[
                "mixvarge_f2",
                "mixvarge_f2",
                "mixvarge_f2",
                "mixvarge_f2",
            ],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 16
    [
        MixVarGENetConfig(
            in_channels=channel_list[4],
            out_channels=channel_list[5],
            head_op="mixvarge_f2",
            stack_ops=["mixvarge_f2", "mixvarge_f2"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 32
]

# less expand ratio on ImageNet128.
# basic level latency (including head): 2060us.
# the head latency: ~252us
# backbone: ~1808us
# FLOPS: 228M * 2
channel_list = [32, 32, 32, 64, 64, 96]
net_config_v5 = [
    [
        MixVarGENetConfig(
            in_channels=channel_list[0],
            out_channels=channel_list[1],
            head_op="mixvarge_f2",
            stack_ops=[],
            stack_factor=1,
            stride=1,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 2
    [
        MixVarGENetConfig(
            in_channels=channel_list[1],
            out_channels=channel_list[2],
            head_op="mixvarge_f2",
            stack_ops=["mixvarge_f2", "mixvarge_f2"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 4
    [
        MixVarGENetConfig(
            in_channels=channel_list[2],
            out_channels=channel_list[3],
            head_op="mixvarge_f2",
            stack_ops=["mixvarge_f2", "mixvarge_f2"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 8
    [
        MixVarGENetConfig(
            in_channels=channel_list[3],
            out_channels=channel_list[4],
            head_op="mixvarge_f1",
            stack_ops=[
                "mixvarge_f1",
                "mixvarge_f1",
                "mixvarge_f1",
                "mixvarge_f1",
                "mixvarge_f1",
                "mixvarge_f1",
            ],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 16
    [
        MixVarGENetConfig(
            in_channels=channel_list[4],
            out_channels=channel_list[5],
            head_op="mixvarge_f2",
            stack_ops=["mixvarge_f2", "mixvarge_f2"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 32
]

# less expand ratio and units and channels on ImageNet128.
# basic level latency (including head): 1859us.
# the head latency: ~252us
# backbone: ~1607us
# FLOPS: 189M * 2
channel_list = [32, 32, 32, 32, 64, 96]
net_config_v6 = [
    [
        MixVarGENetConfig(
            in_channels=channel_list[0],
            out_channels=channel_list[1],
            head_op="mixvarge_f2",
            stack_ops=[],
            stack_factor=1,
            stride=1,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 2
    [
        MixVarGENetConfig(
            in_channels=channel_list[1],
            out_channels=channel_list[2],
            head_op="mixvarge_f2",
            stack_ops=["mixvarge_f2", "mixvarge_f2"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 4
    [
        MixVarGENetConfig(
            in_channels=channel_list[2],
            out_channels=channel_list[3],
            head_op="mixvarge_f2",
            stack_ops=["mixvarge_f2", "mixvarge_f2"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 8
    [
        MixVarGENetConfig(
            in_channels=channel_list[3],
            out_channels=channel_list[4],
            head_op="mixvarge_f1",
            stack_ops=[
                "mixvarge_f1",
                "mixvarge_f1",
                "mixvarge_f1",
                "mixvarge_f1",
            ],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 16
    [
        MixVarGENetConfig(
            in_channels=channel_list[4],
            out_channels=channel_list[5],
            head_op="mixvarge_f2",
            stack_ops=["mixvarge_f2", "mixvarge_f2"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 32
]

# extreme less expand ratio and units and channels on ImageNet128.
# basic level latency (including head): 1805us.
# the head latency: ~252us
# backbone: ~1553us
# FLOPS: 105M * 2
channel_list = [32, 32, 32, 32, 64, 96]
net_config_v7 = [
    [
        MixVarGENetConfig(
            in_channels=channel_list[0],
            out_channels=channel_list[1],
            head_op="mixvarge_f1",
            stack_ops=[],
            stack_factor=1,
            stride=1,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 2
    [
        MixVarGENetConfig(
            in_channels=channel_list[1],
            out_channels=channel_list[2],
            head_op="mixvarge_f1",
            stack_ops=["mixvarge_f1", "mixvarge_f1"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 4
    [
        MixVarGENetConfig(
            in_channels=channel_list[2],
            out_channels=channel_list[3],
            head_op="mixvarge_f1",
            stack_ops=["mixvarge_f1", "mixvarge_f1"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 8
    [
        MixVarGENetConfig(
            in_channels=channel_list[3],
            out_channels=channel_list[4],
            head_op="mixvarge_f1",
            stack_ops=[
                "mixvarge_f1",
                "mixvarge_f1",
                "mixvarge_f1",
                "mixvarge_f1",
            ],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 16
    [
        MixVarGENetConfig(
            in_channels=channel_list[4],
            out_channels=channel_list[5],
            head_op="mixvarge_f1",
            stack_ops=["mixvarge_f1", "mixvarge_f1"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),
    ],  # stride 32
]

model = dict(
    type="Classifier",
    backbone=dict(
        type="MixVarGENet",
        net_config=net_config_v6,
        num_classes=1000,
        bn_kwargs=bn_kwargs,
        # NOTE: you need change avgpool when use
        # resolution not equal to 224 and include top
        include_top=True,
        bias=True,
    ),
    losses=dict(type="CEWithLabelSmooth"),
)

deploy_model = dict(
    type="Classifier",
    backbone=dict(
        type="MixVarGENet",
        net_config=net_config_v6,
        num_classes=1000,
        bn_kwargs=bn_kwargs,
        # NOTE: you need change avgpool when use
        # resolution not equal to 224 and include top
        include_top=True,
        bias=True,
    ),
    losses=None,
)
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
    num_workers=2,
    pin_memory=True,
)
val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ImageNet",
        data_path="./tmp_data/imagenet/val_lmdb/",
        transforms=[
            dict(type="TorchVisionAdapter", interface="Resize", size=128),
            dict(type="TorchVisionAdapter", interface="CenterCrop", size=128),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=2,
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
    step_log_freq=50,
    epoch_log_freq=1,
    log_prefix=task_name,
)
val_metric_updater = copy.deepcopy(metric_updater)
val_metric_updater["log_prefix"] = "Validation " + task_name

stat_callback = dict(
    type="StatsMonitor",
    log_freq=50,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,
    mode="max",
    save_on_train_end=True,
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
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="CosLrUpdater",
            warmup_by="step",
            warmup_len=1000,
            step_log_interval=1000,
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

onnx_cfg = dict(
    model=deploy_model,
    stage="float",
    inputs=deploy_inputs,
    model_convert_pipeline=float_predictor["model_convert_pipeline"],
)
