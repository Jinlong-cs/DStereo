import copy
import os
import warnings

import torch
from horizon_plugin_pytorch.quantization import March

from hat.engine.processors.loss_collector import collect_loss_by_regex

warnings.filterwarnings("ignore")
DEBUG = True
NUM_LDMK = 4
NUM_CLASS = 2

training_step = os.environ.get("HAT_TRAINING_STEP", "float")
task_name = "smoke_kps_heatmap_id0"

if DEBUG:
    batch_size_per_gpu = 16
    device_ids = [0, 1]
else:
    batch_size_per_gpu = 32
    device_ids = [0, 1]
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
model = {
    "type": "SmokeKpsModel",
    "mode": "train",
    "backbone": {
        "type": "VargNetV2",
        "num_classes": 1000,
        "include_top": False,
        "bn_kwargs": {},
    },
    "decoder": {
        "type": "LdmkDecoder",
        "in_stride": 32,
        "out_stride": 4,
        "in_channels": 256,
        "out_channels": 128,
    },
    "vector_head": None,
    "feat_stride": 4,
    "heatmap_head": {
        "type": "SmokeKpsHeatmapHead",
        "in_channels": 128,
        "num_ldmk": NUM_LDMK,
        "is_train": True,
        "loss_func": {"type": "LdmkLoss", "loss_type": "l2"},
    },
    "cls_head": {
        "type": "SmokeKpsClsHead",
        "in_channels": 256,
        "middle_dim": 32,
        "output_dim": 3,
        "is_train": True,
        "loss_func": {"type": "LdmkLoss", "loss_type": "l2"},
    },
    "loss_weights": {"heatmap": 1.0, "cls": 1.0, "vis": 1.0},
}


# Val Model
val_model = {
    "type": "SmokeKpsModel",
    "mode": "val",
    "backbone": {
        "type": "VargNetV2",
        "num_classes": 1000,
        "include_top": False,
        "bn_kwargs": {},
    },
    "decoder": {
        "type": "LdmkDecoder",
        "in_stride": 32,
        "out_stride": 4,
        "in_channels": 256,
        "out_channels": 128,
    },
    "vector_head": None,
    "feat_stride": 4,
    "heatmap_head": {
        "type": "SmokeKpsHeatmapHead",
        "in_channels": 128,
        "num_ldmk": NUM_LDMK,
        "is_train": False,
        "loss_func": {"type": "LdmkLoss", "loss_type": "l2"},
    },
    "cls_head": {
        "type": "SmokeKpsClsHead",
        "in_channels": 256,
        "middle_dim": 32,
        "output_dim": 3,
        "is_train": False,
        "loss_func": {"type": "LdmkLoss", "loss_type": "l2"},
    },
}


# Deploy Model
deploy_model = copy.deepcopy(val_model)
deploy_model["mode"] = "deploy"


# data
root_dir = "tmp_orig_data/action/smoke_kps"
FILENAME = "{}/keypoint_rawdata_0005_210906_train_ir_sorted.rec".format(
    root_dir
)
train_recs = [
    FILENAME,
    FILENAME,
]
val_recs = [
    FILENAME,
]


# transform
train_transforms = [
    dict(
        type="RandomRotateCrop",
        net_input_size=(128, 128),
        rot_prob=1.0,
        rot_angle_range=30,
        center_shift_prob=1.0,
        center_shift_range=0.01,
        norm_ratio=1.25,
        norm_method="longside_square",
        norm_jitter_range=0.25,
        keep_ldmk_complt_ratio=1,
        return_normalized_ldmk=False,
        net_target_size=(128, 128),
        base_len=1.0,
    ),
    dict(
        type="OneFromMultiple",
        transforms=[
            dict(
                type="RandomNoise",
                prob=1,
                min=-5,
                max=5,
            ),
            dict(
                type="GaussianNoise",
                prob=1,
                mean=0,
                sigma=1,
            ),
            dict(
                type="SaltPepperNoise",
                prob=1,
                s_ratio=0.05,
                p_ratio=0.05,
            ),
        ],
        probs=[0.3, 0.3, 0.3],
    ),
    dict(
        type="RandomBrightnessContrast",
        brightness_limit=(-0.2, 0.2),
        contrast_limit=(-0.2, 0.2),
        brightness_by_max=True,
        p=1,
    ),
    dict(
        type="HueSaturationValue",
        hue_range=(-20, 20),
        sat_range=(-30, 30),
        val_range=(-20, 20),
        p=1,
    ),
    dict(
        type="GenerateGaussianHeatmap",
        num_ldmk=NUM_LDMK,
        feat_stride=4,
        heatmap_shape=(32, 32),
        sigma=2,
        encoding_method="ellipse",
    ),
    dict(type="ToTensor"),
]
val_transforms = [
    dict(
        type="RandomRotateCrop",
        net_input_size=(128, 128),
        rot_prob=0.0,
        center_shift_prob=0.0,
        norm_ratio=1.0,
        norm_jitter_range=0.0,
        return_normalized_ldmk=False,
        net_target_size=(128, 128),
        base_len=1.0,
    ),
    dict(
        type="GenerateGaussianHeatmap",
        num_ldmk=NUM_LDMK,
        feat_stride=4,
        heatmap_shape=(32, 32),
        sigma=2,
        encoding_method="ellipse",
    ),
    dict(type="ToTensor"),
]


# datasets
train_datasets = [
    {
        "type": "SmokeKpsRecDataset",
        "filename": rec_file,
        "num_ldmk": NUM_LDMK,
        "num_class": NUM_CLASS,
        "task_type": "smoke_kps",
        "transforms": train_transforms,
    }
    for rec_file in train_recs
]
val_datasets = [
    {
        "type": "SmokeKpsRecDataset",
        "filename": rec_file,
        "num_ldmk": NUM_LDMK,
        "num_class": NUM_CLASS,
        "task_type": "smoke_kps",
        "transforms": val_transforms,
    }
    for rec_file in val_recs
]


# dataloader
data_loader = {
    "type": torch.utils.data.DataLoader,
    "dataset": {
        "type": torch.utils.data.ConcatDataset,
        "datasets": train_datasets,
    },
    "sampler": {"type": torch.utils.data.DistributedSampler},
    "batch_size": batch_size_per_gpu,
    "shuffle": True,
    "num_workers": 5,
    "pin_memory": False,
}


val_data_loader = {
    "type": torch.utils.data.DataLoader,
    "dataset": {
        "type": torch.utils.data.ConcatDataset,
        "datasets": val_datasets,
    },
    "sampler": {"type": torch.utils.data.DistributedSampler},
    "batch_size": batch_size_per_gpu,
    "shuffle": False,
    "num_workers": 5,
    "pin_memory": False,
}


# batch_processor
batch_processor = {
    "type": "BasicBatchProcessor",
    "need_grad_update": True,
    "batch_transforms": [
        {"type": "BgrToYuv444", "rgb_input": True},
        {
            "type": "TorchVisionAdapter",
            "interface": "Normalize",
            "mean": 128.0,
            "std": 128.0,
        },
    ],
    "loss_collector": collect_loss_by_regex("total_loss"),
}

val_batch_processor = {
    "type": "BasicBatchProcessor",
    "need_grad_update": False,
    "batch_transforms": [
        {"type": "BgrToYuv444", "rgb_input": True},
        {
            "type": "TorchVisionAdapter",
            "interface": "Normalize",
            "mean": 128.0,
            "std": 128.0,
        },
    ],
}


# metric_updater
def update_metric(metrics, batch, model_outs):
    for metric, key in zip(metrics, model_outs):
        metric.update(model_outs[key])


def update_val_metric(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


metric_updater = {
    "type": "MetricUpdater",
    "metric_update_func": update_metric,
    "step_log_freq": 10,
    "epoch_log_freq": 1,
    "log_prefix": task_name,
}


val_metric_updater = {
    "type": "MetricUpdater",
    "metric_update_func": update_val_metric,
    "step_log_freq": 10,
    "epoch_log_freq": 1,
    "log_prefix": "Validation" + task_name,
}


# callback
stat_callback = {
    "type": "StatsMonitor",
    "log_freq": 10,
}

ckpt_callback = {
    "type": "Checkpoint",
    "save_dir": ckpt_dir,
    "name_prefix": training_step + "-",
    "strict_match": True,
    "save_hash": False,
}

val_callback = {
    "type": "Validation",
    "data_loader": val_data_loader,
    "batch_processor": val_batch_processor,
    "callbacks": [val_metric_updater],
    "val_model": val_model,
    "val_on_train_end": False,
}


# float_trainer
float_trainer = {
    "type": "distributed_data_parallel_trainer",
    "model": model,
    "data_loader": data_loader,
    "optimizer": {
        "type": torch.optim.AdamW,
        "params": {"weight": {"weight_decay": 0}},
        "lr": 1e-3,
    },
    "batch_processor": batch_processor,
    "num_epochs": 2,
    "num_steps": 20,
    "stop_by": "step" if DEBUG else "epoch",
    "device": None,
    "callbacks": [
        stat_callback,
        {
            "type": "StepDecayLrUpdater",
            "step_log_interval": 1000,
            "lr_decay_id": [100, 150],
            "lr_decay_factor": 0.1,
        },
        metric_updater,
        val_callback,
        ckpt_callback,
    ],
    "train_metrics": [
        {"type": "LossShow", "name": "LdmkLoss"},
        {"type": "LossShow", "name": "Loss"},
    ],
    "val_metrics": [
        {
            "type": "SmokeKpsNME",
            "num_ldmk": NUM_LDMK,
            "norm_type": "norm12",
            "mode": "heatmap",
            "decoding_method": "shift",
            "feat_stride": 4,
            "name": "SmokeKpsNME_12",
        },
        {
            "type": "SoftmaxAccuracy",
            "cls_type": "classes",
            "thresh": 0.5,
            "name": "SmokeKpsAcc",
        },
        {
            "type": "SoftmaxRecall",
            "cls_type": "classes",
            "thresh": 0.5,
            "name": "SmokeKpsRec",
        },
        {
            "type": "SoftmaxPrecision",
            "cls_type": "classes",
            "thresh": 0.5,
            "name": "SmokeKpsPre",
        },
    ],
}

float_solver = {
    "trainer": float_trainer,
    "quantize": False,
    "allow_not_init": True,
    "strict_match": True,
    "pretrain_checkpoint": "vargnetv2.pth",
    "allow_miss": True,
    "ignore_extra": True,
}


# qat_trainer
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
    num_epochs=2,
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
        {
            "type": "SmokeKpsNME",
            "num_ldmk": NUM_LDMK,
            "norm_type": "norm12",
            "mode": "heatmap",
            "decoding_method": "shift",
            "feat_stride": 4,
            "name": "SmokeKpsNME_12",
        },
        {
            "type": "SoftmaxAccuracy",
            "cls_type": "classes",
            "thresh": 0.5,
            "name": "SmokeKpsAcc",
        },
        {
            "type": "SoftmaxRecall",
            "cls_type": "classes",
            "thresh": 0.5,
            "name": "SmokeKpsRec",
        },
        {
            "type": "SoftmaxPrecision",
            "cls_type": "classes",
            "thresh": 0.5,
            "name": "SmokeKpsPre",
        },
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

# int_trainer
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
        {
            "type": "SmokeKpsNME",
            "num_ldmk": NUM_LDMK,
            "norm_type": "norm12",
            "mode": "heatmap",
            "decoding_method": "shift",
            "feat_stride": 4,
            "name": "SmokeKpsNME_12",
        }
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
