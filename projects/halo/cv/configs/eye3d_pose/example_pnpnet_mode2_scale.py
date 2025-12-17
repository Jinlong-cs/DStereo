import copy
import json
import os

import torch
from horizon_plugin_pytorch.march import March

from hat.data.datasets.eye3d_pose_dataset import Eye3dPoseDataset
from hat.data.transforms.eye3d_pose import Eye3dPoseTransformList
from hat.engine.processors.loss_collector import collect_loss_by_index
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")
DEBUG = False

task_name = "pnpnet_mode2_id2_ldmk_scale_0425"
device_ids = [0]
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.BAYES
batch_size_per_gpu = 256

argvs = [
    {
        "txtpath": "/horizon-bucket/MultiMode_2/mm_algorithms_data/pnpnet/train_da_40pts.txt"
    },
    {
        "synthetic_argv": {
            "length": 100000,
            "face_model": "/horizon-bucket/MultiMode_2/mm_algorithms_data/pnpnet/facemodel.txt",
            "range_list": [
                {
                    "mean": [0, 0, 500],
                    "scale": [150, 150, 300],
                    "pose_scale": [30, 40, 50],
                    "pose_mean": [0, 0, 0],
                    "ratio": 2,
                },
                # {
                #     "mean": [0, 0, 400],
                #     "scale": [500, 200, 200],
                #     "pose_scale": [30, 40, 50],
                #     "pose_mean": [0, 0, 0],
                #     "ratio": 1,
                # },
            ],
        }
    },
]
argvs_val = [
    {
        "txtpath": "/horizon-bucket/MultiMode_2/mm_algorithms_data/pnpnet/test_da_40pts.txt"
    },
]
data_list = [
    {
        "input_shape": [29, 1, 1],
        "ldmk_idx": [36, 39, 45, 42, 30, 48, 54, 19, 21, 22, 24, 51, 57],
        "transform_args": {
            "t_mean": 521.34283,
            "t_std": 69.54176,
            "use_pnp_eye3d": True,
            "use_pnp_eye3d_test": True,
            "use_face_center": False,
        },
    },
    {
        "input_shape": [29, 1, 1],
        "ldmk_idx": [36, 39, 45, 42, 30, 19, 21, 22, 24],
        "transform_args": {
            "t_mean": 521.34283,
            "t_std": 69.54176,
            "use_pnp_eye3d": False,
            "use_pnp_eye3d_test": False,
            "use_face_center": True,
        },
    },
]
ldmk_mean_scale = json.load(
    open(
        "/horizon-bucket/MultiMode_2/mm_algorithms_data/pnpnet/ldmk_mean_scale.txt"
    )
)
augm = {"distribution": "normal", "params": {"clip": 0.04, "std": 0.02}}
transform = Eye3dPoseTransformList(
    data_list, augm, ldmk_mean_scale=ldmk_mean_scale
)
transform_val = Eye3dPoseTransformList(
    data_list, ldmk_mean_scale=ldmk_mean_scale
)
dataset = Eye3dPoseDataset(argvs, transform)
dataset_val = Eye3dPoseDataset(argvs_val, transform_val)
input_channels = dataset[0]["img"].shape[0]
data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dataset,
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=0 if DEBUG else 4,
    pin_memory=True,
)
val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dataset_val,
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=0 if DEBUG else 0,
    pin_memory=True,
)

model = dict(
    type="Eye3dPoseModel",
    num_mod=len(data_list),
    input_channels=input_channels,
    backbone_scale_list=[128] * 4,
    neck_scale_list=[64] * 2,
    losses=torch.nn.L1Loss(),
    deploy=False,
    bn_kwargs={
        "eps": 2e-05,
        "momentum": 0.1,
    },
)
deploy_model = dict(
    type="Eye3dPoseModel",
    num_mod=len(data_list),
    input_channels=input_channels,
    backbone_scale_list=[128] * 4,
    neck_scale_list=[64] * 2,
    deploy=True,
    bn_kwargs={
        "eps": 2e-05,
        "momentum": 0.1,
    },
)
# deploy_inputs = dict(img=torch.randn((1, input_channels, 1, 1)))
deploy_inputs = dict(img=dataset[0]["img"].view(1, -1, 1, 1))

deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2QAT"),
        dict(type="QAT2Quantize"),
    ],
)

batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    batch_transforms=[],
    loss_collector=collect_loss_by_index(1),
)
val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    batch_transforms=[],
    loss_collector=None,
)


def update_metric(metrics, batch, model_outs):
    target = batch["gt_eye3d_pose"]
    preds, losses = model_outs
    for metric in metrics:
        if isinstance(metric.name, str) and "loss" in metric.name.lower():
            metric.update(losses)
        else:
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

stat_callback = dict(
    type="StatsMonitor",
    log_freq=100,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,
    mode="max",
    monitor_metric_key="depth_mode_0",
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
)

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        params={"weight": dict(weight_decay=0.0)},
        lr=5e-4,
    ),
    batch_processor=batch_processor,
    num_epochs=20,
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="CosLrUpdater",
            warmup_by="epoch",
            warmup_len=0,
            step_log_interval=1000,
            stop_lr=1e-6,
        ),
        metric_updater,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow", name="L1Loss"),
        dict(
            type="Eye3dPoseAngleAndDist",
            num_mode=len(data_list),
            std=data_list[0]["transform_args"]["t_std"],
        ),
    ],
    val_metrics=[
        dict(
            type="Eye3dPoseAngleAndDist",
            num_mode=len(data_list),
            std=data_list[0]["transform_args"]["t_std"],
        ),
    ],
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
                ),  # noqa
            ),
            dict(type="Float2QAT"),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=4e-5)},
        lr=0.0001,
        momentum=0.9,
    ),
    batch_processor=batch_processor,
    num_epochs=5,
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
        dict(type="LossShow", name="L1Loss"),
        dict(
            type="Eye3dPoseAngleAndDist",
            num_mode=len(data_list),
            std=data_list[0]["transform_args"]["t_std"],
        ),
    ],
    val_metrics=[
        dict(
            type="Eye3dPoseAngleAndDist",
            num_mode=len(data_list),
            std=data_list[0]["transform_args"]["t_std"],
        ),
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
                    ckpt_dir, "qat-checkpoint-last.pth.tar"
                ),  # noqa
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
    input_source=["ddr"],
)

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
        dict(
            type="Eye3dPoseAngleAndDist",
            is_seperate=True,
            num_mode=len(data_list),
            std=data_list[0]["transform_args"]["t_std"],
            is_3d_error_normed=False,
        ),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

onnx_cfg = dict(
    model=copy.deepcopy(deploy_model),
    stage="float",
    inputs=deploy_inputs,
    model_convert_pipeline=dict(
        type="LoadCheckpoint",
        checkpoint_path=os.path.join(
            ckpt_dir, "float-checkpoint-last.pth.tar"
        ),
    ),
)
