import copy
import os

import torch
from horizon_plugin_pytorch.march import March

from hat.engine.processors.loss_collector import collect_loss_by_index
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "timesformer_cls"
num_classes = 101
batch_size_per_gpu = 4
device_ids = [0, 1, 2, 3]
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.NASH
convert_mode = "fx"

model = dict(
    type="Classifier",
    backbone=dict(
        type="TimeSformer",
        num_classes=101,
        pretrained=True,
        pretrained_model="./tmp_pretrained_models/timesformer/jx_vit_base_p16_224-80ecf9dd.pth",  # noqa E501
    ),
    losses=dict(type="SoftTargetCrossEntropy"),
)
deploy_model = dict(
    type="Classifier",
    backbone=dict(
        type="TimeSformer",
        num_classes=101,
    ),
    losses=None,
)
deploy_inputs = dict(img=torch.randn((1, 3, 8, 224, 224)))

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
        type="UCF101",
        root="./tmp_orig_data/ucf101",
        file_path="./tmp_orig_data/ucf101/ucf101_train_list.txt",
        mode="train",
        transforms=[
            dict(
                type="NormalizeVideo",
                mean=(0.45, 0.45, 0.45),
                std=(0.225, 0.225, 0.225),
                tensor_shape=(1, 1, 1, 3),
            ),
            dict(
                type="JitterScaleVideo",
                min_size=224,
                max_size=320,
            ),
            dict(
                type="RandomCropVideo",
                target_size=224,
            ),
            dict(
                type="RandomFlipVideo",
            ),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler, drop_last=True),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="UCF101",
        root="./tmp_orig_data/ucf101",
        file_path="./tmp_orig_data/ucf101/ucf101_val_list.txt",
        mode="test",
        transforms=[
            dict(
                type="NormalizeVideo",
                mean=(0.45, 0.45, 0.45),
                std=(0.225, 0.225, 0.225),
                tensor_shape=(1, 1, 1, 3),
            ),
            dict(
                type="JitterScaleVideo",
                min_size=256,
                max_size=256,
            ),
            dict(
                type="RandomCropVideo",
                target_size=224,
            ),
        ],
    ),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    loss_collector=collect_loss_by_index(1),
)

val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
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
    step_log_freq=100,
    epoch_log_freq=1,
    log_prefix=task_name,
)
val_metric_updater = copy.deepcopy(metric_updater)
val_metric_updater["log_prefix"] = "Validation " + task_name
val_metric_updater["step_log_freq"] = 5000

stat_callback = dict(
    type="StatsMonitor",
    log_freq=500,
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
    log_interval=200,
    val_on_train_end=False,
)

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={
            "weight": dict(weight_decay=1e-4),
            "bias": dict(weight_decay=1e-4),
        },
        lr=0.0025,
        momentum=0.9,
        weight_decay=1e-4,
    ),
    batch_processor=batch_processor,
    num_epochs=15,
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            step_log_interval=100,
            lr_decay_id=[11, 14],
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

# Note: The transforms of the dataset during calibration can be
# consistent with that during training or validation, or customized.
# Default used `val_batch_processor`.
calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop("sampler")  # Calibration do not support DDP or DP
calibration_data_loader["batch_size"] = batch_size_per_gpu * 4
calibration_data_loader["dataset"]["transforms"] = val_data_loader["dataset"][
    "transforms"
]
calibration_batch_processor = copy.deepcopy(val_batch_processor)
calibration_step = 10

calibration_trainer = dict(
    type="Calibrator",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        qconfig_params=dict(
            activation_calibration_observer="percentile",
            activation_calibration_qkwargs=dict(
                percentile=99.5,
            ),
        ),
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
        params={
            "weight": dict(weight_decay=4e-5),
            "bias": dict(weight_decay=4e-5),
        },
        lr=0.000625,
        momentum=0.9,
        weight_decay=4e-5,
    ),
    batch_processor=batch_processor,
    num_epochs=15,
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            step_log_interval=1000,
            lr_decay_id=[11, 14],
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
                    ckpt_dir, "float-checkpoint-best.pth.tar"
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
