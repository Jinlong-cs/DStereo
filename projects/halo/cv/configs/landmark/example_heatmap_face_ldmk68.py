import copy
import os
import warnings

import torch
from horizon_plugin_pytorch.quantization import March

from hat.engine.processors.loss_collector import collect_loss_by_regex

warnings.filterwarnings("ignore")
DEBUG = True
NUM_LDMK = 68
LDMK_PAIRS = [
    [0, 16],
    [1, 15],
    [2, 14],
    [3, 13],
    [4, 12],
    [5, 11],
    [6, 10],
    [7, 9],
    [17, 26],
    [18, 25],
    [19, 24],
    [20, 23],
    [21, 22],
    [31, 35],
    [32, 34],
    [36, 45],
    [37, 44],
    [38, 43],
    [39, 42],
    [40, 47],
    [41, 46],
    [48, 54],
    [49, 53],
    [50, 52],
    [61, 63],
    [60, 64],
    [67, 65],
    [58, 56],
    [59, 55],
]


training_step = os.environ.get("HAT_TRAINING_STEP", "float")
task_name = "face_ldmk_heatmap_id0"

if DEBUG:
    batch_size_per_gpu = 32
    device_ids = [0, 1, 2, 3]
else:
    batch_size_per_gpu = 32
    device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
ckpt_dir = "./tmp_models/%s" % task_name
if not os.path.isdir("tmp_models"):
    os.makedirs("./tmp_models")
if not os.path.isdir(ckpt_dir):
    os.makedirs(ckpt_dir)
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.BERNOULLI2


# Train Model
model = dict(
    type="LdmkModel",
    backbone=dict(
        type="VargNetV2", num_classes=1000, include_top=False, bn_kwargs={}
    ),
    mode="train",
    decoder=dict(
        type="LdmkDecoder",
        in_channels=256,
        out_channels=128,
        num_stage=3,
        use_deconv=False,
    ),
    vector_head=None,
    coords_head=None,
    feat_stride=4,
    heatmap_head=dict(
        type="LdmkHeatmapHead",
        in_channels=128,
        num_ldmk=NUM_LDMK,
        is_train=True,
        loss_func=dict(type="LdmkLoss", loss_type="l2"),
    ),
    cls_head=None,
    loss_weights={"heatmap": 1.0},
)


# Val Model
val_model = dict(
    type="LdmkModel",
    backbone=dict(
        type="VargNetV2", num_classes=1000, include_top=False, bn_kwargs={}
    ),
    mode="val",
    decoder=dict(
        type="LdmkDecoder",
        in_channels=256,
        out_channels=128,
        num_stage=3,
        use_deconv=False,
    ),
    vector_head=None,
    coords_head=None,
    feat_stride=4,
    heatmap_head=dict(
        type="LdmkHeatmapHead",
        in_channels=128,
        num_ldmk=NUM_LDMK,
        is_train=False,
    ),
    cls_head=None,
)


# Deploy Model
deploy_model = copy.deepcopy(val_model)
deploy_model["mode"] = "deploy"


# data
train_recs = [
    "face_ldmk/lmks_large_pose_68pts_train_23.rec",
    "face_ldmk/lmks_large_pose_68pts_train_23.rec",
]
val_recs = [
    "face_ldmk/lmks_large_pose_68pts_train_23.rec",
]


# transform
train_transforms = [
    dict(
        type="CropRecROI",
        crop_type="random",
        target_shape=(128, 128, 3),
        base_roi=[48, 48, 208, 208],
        crop_jitter_range=0.25,
        random_type="gaussian",
    ),
    dict(
        type="GenerateGaussianHeatmap",
        num_ldmk=NUM_LDMK,
        feat_stride=4,
        heatmap_shape=(32, 32),
        sigma=2,
        encoding_method="standard",
    ),
    dict(type="ToTensor"),
]
val_transforms = [
    dict(
        type="CropRecROI",
        crop_type="center",
        target_shape=(128, 128, 3),
        base_roi=[48, 48, 208, 208],
        crop_jitter_range=0.0,
    ),
    dict(type="ToTensor"),
]


# datasets
train_datasets = [
    dict(
        type="LdmkRecDataset",
        filename=rec_file,
        num_ldmk=NUM_LDMK,
        use_3d=False,
        task_type="face",
        transforms=train_transforms,
        ldmk_pairs=LDMK_PAIRS,
    )
    for rec_file in train_recs
]
val_datasets = [
    dict(
        type="LdmkRecDataset",
        filename=rec_file,
        num_ldmk=NUM_LDMK,
        use_3d=False,
        task_type="face",
        transforms=val_transforms,
        ldmk_pairs=LDMK_PAIRS,
    )
    for rec_file in val_recs
]


# dataloader
test_inputs = dict(img=torch.randn((1, 3, 128, 128)))
data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type=torch.utils.data.ConcatDataset,
        datasets=train_datasets,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=5,
    pin_memory=False,
)


val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(type=torch.utils.data.ConcatDataset, datasets=val_datasets),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=5,
    pin_memory=False,
)


batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    batch_transforms=[
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
    loss_collector=collect_loss_by_regex("total_loss"),
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
)


def update_metric(metrics, batch, model_outs):
    for metric, key in zip(metrics, model_outs):
        metric.update(model_outs[key])


metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=10,
    epoch_log_freq=1,
    log_prefix=task_name,
)


def update_val_metric(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_val_metric,
    step_log_freq=10,
    epoch_log_freq=1,
    log_prefix="Validation" + task_name,
)


stat_callback = dict(
    type="StatsMonitor",
    log_freq=10,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    test_model=deploy_model,
    test_inputs=test_inputs,
    strict_match=True,
    save_hash=False,
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=val_model,
    val_on_train_end=False,
)


float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        params={"weight": dict(weight_decay=0)},
        lr=1e-3,
    ),
    batch_processor=batch_processor,
    num_epochs=200,
    num_steps=20,
    stop_by="step" if DEBUG else "epoch",
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            step_log_interval=1000,
            lr_decay_id=[100, 150],
            lr_decay_factor=0.1,
        ),
        metric_updater,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow", name="LdmkLoss"),
        dict(type="LossShow", name="Loss"),
    ],
    val_metrics=[
        dict(
            type="NormalizedMeanError",
            num_ldmk=NUM_LDMK,
            norm_type="ION",
            mode="heatmap",
            decoding_method="shift",
            feat_stride=4,
            name="LdmkION",
        ),
    ],
)
float_solver = dict(
    trainer=float_trainer,
    quantize=False,
    allow_not_init=True,
    strict_match=True,
    pretrain_checkpoint="vargnetv2.pth",
    allow_miss=True,
    ignore_extra=True,
)


qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        params={"weight": dict(weight_decay=0)},
        lr=1e-3,
    ),
    batch_processor=batch_processor,
    num_epochs=200,
    num_steps=20,
    stop_by="step" if DEBUG else "epoch",
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            step_log_interval=1000,
            lr_decay_id=[100, 150],
            lr_decay_factor=0.1,
        ),
        metric_updater,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow", name="LdmkLoss"),
        dict(type="LossShow", name="Loss"),
    ],
    val_metrics=[
        dict(
            type="NormalizedMeanError",
            num_ldmk=NUM_LDMK,
            norm_type="ION",
            mode="heatmap",
            decoding_method="shift",
            feat_stride=4,
            name="LdmkION",
        ),
    ],
)
qat_solver = dict(
    trainer=qat_trainer,
    quantize=True,
    check_quantize_model=True,
    pre_step="float",
    pre_step_checkpoint=os.path.join(
        ckpt_dir, "float-checkpoint-last.pth.tar"
    ),
    strict_match=True,
)


int_trainer = dict(
    type="Trainer",
    model=model,
    data_loader=None,
    optimizer=None,
    batch_processor=None,
    num_epochs=0,
    device=None,
    callbacks=[ckpt_callback, val_callback],
    val_metrics=[
        dict(
            type="NormalizedMeanError",
            num_ldmk=NUM_LDMK,
            norm_type="ION",
            mode="heatmap",
            name="LdmkION",
            decoding_method="shift",
            feat_stride=4,
        ),
    ],
)
int_solver = dict(
    trainer=int_trainer,
    quantize=True,
    check_quantize_model=True,
    pre_step="qat",
    pre_step_checkpoint=os.path.join(ckpt_dir, "qat-checkpoint-last.pth.tar"),
    strict_match=True,
)

step2solver = dict(float=float_solver, qat=qat_solver, int_infer=int_solver)

compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=task_name,
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["pyramid"],
)
