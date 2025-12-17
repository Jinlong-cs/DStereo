import os

import torch
import torchvision
from horizon_plugin_pytorch.quantization import March

from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "fas_vargnetv2_multihead_example"
num_classes_fas = 1
batch_size_per_gpu = 144
device_ids = [0, 1, 2, 3]


ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.BAYES

train_rec_list = [
    # # # # guangqi-A11-zhihua2311-A
    "MultiMode/shanyun.gao/training_set/IR/ov2311/liveness-for-A11/2021-08-17/result/positive_order/train_no_deform_pure_expand_12_144.rec",  # noqa
    "MultiMode/shanyun.gao/training_set/IR/ov2311/liveness-for-A11/2021-08-17/result/negative_order/train_no_deform_pure_expand_12_144.rec",  # noqa
    # guangqi-A57-zhihua2311-A
    "MultiMode/shanyun.gao/training_set/IR/ov2311/liveness-for-A57/2021-08-17/result/positive_order/train_no_deform_pure_expand_12_144.rec",  # noqa
    "MultiMode/shanyun.gao/training_set/IR/ov2311/liveness-for-A57/2021-08-17/result/negative_order/train_no_deform_pure_expand_12_144.rec",  # noqa
]
train_database_labels = [0, 0, 1, 1]


model = dict(
    type="FasClassifier",
    backbone=dict(
        type="VargNetV2",
        num_classes=-1,
        model_type="tinyvargnetv2",
        include_top=False,
        bn_kwargs={},
    ),
    head=dict(
        type="FasAdaptiveHead",
        in_channels=256,
        database_labels=train_database_labels,
    ),
    loss=dict(
        type="FasMultiheadFocalLoss",
        name="fas_loss",
    ),
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="FasRecDataset",
        imgrec_path_list=train_rec_list,
        database_labels=train_database_labels,
        transforms=[
            dict(
                type="RandomCrop",
                size=(128, 128),
                center_crop_prob=0.3,
            ),
            dict(type="SpatialVariantBrightness", p=0.5, brightness=0.6),
            dict(
                type="ToTensor",
                to_yuv=False,
            ),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)


test_inputs = dict(img=torch.randn((1, 3, 128, 128)))

val_rec_list = [
    "MultiMode_2/mm_algorithms_data/anti-spoofing/test-set/IR/ov2311/liveness-for-s202da/S202DA-test/all_woincar/test_no_deform_pure_expand_12_144.rec",  # noqa
]
val_database_labels = [
    1,
]
val_database_names = [
    "s202da",
]

test_model = dict(
    type="FasClassifier",
    backbone=dict(
        type="VargNetV2",
        num_classes=-1,
        model_type="tinyvargnetv2",
        include_top=False,
        bn_kwargs={},
    ),
    head=dict(
        type="FasAdaptiveHead",
        in_channels=256,
        database_labels=val_database_labels,
    ),
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="FasRecDataset",
        imgrec_path_list=val_rec_list,
        database_labels=val_database_labels,
        transforms=[
            dict(
                type="ToTensor",
                to_yuv=True,
            ),
            dict(
                type="TorchVisionAdapter",
                interface="CenterCrop",
                size=128,
            ),
            dict(
                type="TorchVisionAdapter",
                interface="Normalize",
                mean=128.0,
                std=128.0,
            ),
        ],
    ),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)


def loss_collector(outputs: list):
    losses = []
    if len(outputs) == 2 and isinstance(outputs[0], list):
        if isinstance(outputs[1], dict):
            assert "fas_loss" in outputs[1].keys()
            losses.append(outputs[1]["fas_loss"])
        else:
            losses.append(outputs[1])
    return losses


batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    loss_collector=loss_collector,
    batch_transforms=[
        dict(
            type="TorchVisionAdapter",
            interface="RandomHorizontalFlip",
            p=0.5,
        ),
        dict(
            type="TorchVisionAdapter",
            interface="RandomApply",
            transforms=torch.nn.ModuleList(
                [
                    torchvision.transforms.ColorJitter(brightness=0.5),
                ]
            ),
            p=0.8,
        ),
        dict(
            type="TorchVisionAdapter",
            interface="RandomApply",
            transforms=torch.nn.ModuleList(
                [
                    torchvision.transforms.ColorJitter(contrast=0.8),
                ]
            ),
            p=0.8,
        ),
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
)
val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    loss_collector=None,
)

train_metrics = [
    dict(type="LossShow", name="fas_loss"),
    dict(type="FasSigmoidAccuracy", thr=0.5, name="fas_accuracy"),
]


def update_metric(metrics, batch, model_outs):
    fas_label = batch["fas_label"]
    car_cls = batch["database_labels"]
    for metric in metrics:
        if "loss" in metric.name:
            metric.update(loss_collector(model_outs)[0])
        elif "fas_accuracy" in metric.name:
            if len(model_outs) == 2 and isinstance(model_outs[0], list):
                preds = model_outs[0]
            else:
                preds = model_outs
            metric.update(preds, fas_label, car_cls)
        else:
            assert 0


metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=500,
    epoch_log_freq=1,
    log_prefix=task_name,
)

val_metrics = [
    dict(type="FasTARTRR", database_names=val_database_names, name="TAR_TRR")
]


def val_update_metric(metrics, batch, model_outs):
    fas_label = batch["fas_label"]
    car_cls = batch["database_labels"]
    for metric in metrics:
        if "TAR_TRR" in metric.name:
            metric.update(model_outs, fas_label, car_cls)
        else:
            assert 0


val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=val_update_metric,
    step_log_freq=-1,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=500,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=test_inputs,
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_interval=10,
    val_model=None,
)

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        params={"weight": dict(weight_decay=1e-4)},
        lr=0.01,
    ),
    batch_processor=batch_processor,
    num_epochs=10,
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            lr_decay_id=[5, 15],
            lr_decay_factor=0.1,
            warmup_by="step",
            warmup_len=2000,
            step_log_interval=500,
        ),
        metric_updater,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=train_metrics,
    val_metrics=val_metrics,
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
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        params={"weight": dict(weight_decay=1e-4)},
        lr=0.001,
    ),
    batch_processor=batch_processor,
    num_epochs=10,
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            step_log_interval=500,
            lr_decay_id=[5, 10],
            lr_decay_factor=0.1,
            warmup_by="step",
            warmup_len=2000,
        ),
        metric_updater,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=train_metrics,
    val_metrics=val_metrics,
)


int_infer_trainer = dict(
    type="Trainer",
    model=model,
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
            ),
            dict(type="QAT2Quantize"),
        ],
    ),
    data_loader=None,
    optimizer=None,
    batch_processor=None,
    num_epochs=0,
    device=None,
    callbacks=[ckpt_callback, trace_callback],
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
