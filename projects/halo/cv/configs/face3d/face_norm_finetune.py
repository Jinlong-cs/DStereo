import copy
import os

import torch
from face3d_datahub import get_dataset
from horizon_plugin_pytorch.quantization import March

from hat.engine.processors.loss_collector import collect_loss_by_index

try:
    import lpips
except ImportError:
    raise ImportError("Please install lpips")
import warnings

warnings.filterwarnings("ignore")

DEBUG = True  # False# True

training_step = os.environ.get("HAT_TRAINING_STEP", "float")
task_name = "face_norm_id_debug"
if DEBUG:
    batch_size_per_gpu = 16
    device_ids = [3]  # 2, 3]
else:
    batch_size_per_gpu = 96
    device_ids = [0, 1, 2, 3, 4, 5, 6, 7]

ckpt_dir = "./tmp_models/%s" % task_name
log_dir = "./tmp_models/%s/log" % task_name
log_freq = 50
enable_tensorboard = True
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
# march = March.BERNOULLI2
march = March.BAYES
undistort = False
train_datasets = ["kinect471"]
val_datasets = ["kinect120"]

train_data = get_dataset(train_datasets)


# Model for training
model = dict(
    type="Face3dModel",
    modeltype="pp",
    ldmk_norm=True,
    undistort=undistort,
    use_flame=True,
    backbone=dict(
        type="VargNetV2",
        input_channels=5,
        num_classes=1000,
        include_top=False,
        bn_kwargs={},
    ),
    head=dict(
        type="Face3dHead",
        kernel_size=4,
        in_channels=256,
    ),
    flame=dict(
        type="FLAME",
        flame_model_path="face3d_data/new_generic_model.pkl",
        flame_lmk_embedding_path="face3d_data/fined_landmark_embedding.npy",
    ),
    flame_tex=dict(
        type="FLAMETex",
        tex_path="face3d_data/FLAME_albedo_from_BFM.npz",
    ),
    renderer=dict(
        type="NVRenderer",
        obj_filename="face3d_data/head_template_mesh.obj",
        render_size=(256, 256),
    ),
    lpips=lpips.LPIPS(
        pretrained=True,
        pnet_rand=True,
        model_path="face3d_data/lpips.pth",
        verbose=True,
        net="vgg",
    ),
    loss_weights={
        "ldmk": 512,  # 256.0,
        "eye3d": 100,
        "photo": 10,  # 10.0,
        "lpips": 30,  # 30.0,
        "shape_reg": 0.01,
        "exp_reg": 0.01,
        "tex_reg": 0.01,
        "flame_transl_xy_reg": 0,
        "flame_transl_z_reg": 0,
        "flame_global_pose_reg": 0,
    },
)

# Deploy Model for saving checkpoint
deploy_model = copy.deepcopy(model)
deploy_model["deploy"] = True

deploy_inputs = dict(img=torch.randn((1, 3, 128, 128)))

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Face3dDataset",
        image_path_list=train_data["image_list"],
        mask_path_list=train_data["mask_list"],
        anno_path_list=train_data["anno_list"],
        transforms=[
            dict(
                type="VirtualCameraNorm",
                norm_ratio=1.2,
                norm_method="longside_square",
                undistort=undistort,
            ),
            dict(
                type="PositionEncoding",
                net_input_size=(128, 128),
                concat_img=True,
            ),
            dict(type="ToTensor"),
        ],
        stage="finetune",
        modeltype="pp",
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

val_data = get_dataset(val_datasets)
val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Face3dDataset",
        image_path_list=val_data["image_list"],
        mask_path_list=val_data["mask_list"],
        anno_path_list=val_data["anno_list"],
        transforms=[
            dict(
                type="VirtualCameraNorm",
                norm_ratio=1.2,
                norm_method="longside_square",
                undistort=undistort,
            ),
            dict(
                type="PositionEncoding",
                net_input_size=(128, 128),
                concat_img=True,
            ),
            dict(type="ToTensor"),
        ],
        stage="finetune",
        modeltype="pp",
    ),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=1,
    pin_memory=False,
)

batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    batch_transforms=[
        # dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
    loss_collector=collect_loss_by_index(0),
)
val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    batch_transforms=[
        # dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
    loss_collector=collect_loss_by_index(0),
)


def update_metric(metrics, batch, model_outs):
    for metric, loss in zip(metrics, model_outs):
        metric.update(loss)


metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=100,
    epoch_log_freq=1,
    log_prefix=task_name,
)


stat_callback = dict(
    type="StatsMonitor",
    log_freq=50,
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,
)
ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,
    mode="max",
)


# NOTE: use pretrain params from pretrain stage
float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path="tmp_models/yisu_pretrain/float-checkpoint-last.pth.tar",
                allow_miss=True,
                ignore_extra=True,
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        params={"weight": dict(weight_decay=0)},
        lr=1e-3,
    ),
    batch_processor=batch_processor,
    device=None,
    stop_by="epoch" if DEBUG else "epoch",
    num_epochs=50,
    num_steps=300,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            # warmup_by="epoch",
            # warmup_len=3,
            step_log_interval=100,
            lr_decay_id=[20, 35],
            lr_decay_factor=0.1,
        ),
        metric_updater,
        ckpt_callback,
    ],
    sync_bn=False,
    train_metrics=[
        dict(type="LossShow", name="Loss"),
        dict(type="LossShow", name="Ldmk"),
        dict(type="LossShow", name="Eye3D"),
        dict(type="LossShow", name="Photo"),
        dict(type="LossShow", name="Lpips"),
        dict(type="LossShow", name="Shape"),
        dict(type="LossShow", name="Exp"),
        dict(type="LossShow", name="Tex"),
        dict(type="LossShow", name="FLAME_reg"),
    ],
    val_metrics=[
        dict(type="LossShow", name="Loss"),
        dict(type="LossShow", name="Ldmk"),
        dict(type="LossShow", name="Eye3D"),
        dict(type="LossShow", name="Photo"),
        dict(type="LossShow", name="Lpips"),
        dict(type="LossShow", name="Shape"),
        dict(type="LossShow", name="Exp"),
        dict(type="LossShow", name="Tex"),
        dict(type="LossShow", name="FLAME_reg"),
    ],
)


def update_val_metric(metrics, batch, model_outs):
    for metric in metrics:
        if isinstance(metric.name, list) and len(metric.name) == 4:
            metric.update(batch, model_outs["pose"])
        elif "Eye3dMAE_x_left" in metric.name:
            metric.update(batch, model_outs["left_eye_pred"])
        elif "Eye3dMAE_x_right" in metric.name:
            metric.update(batch, model_outs["right_eye_pred"])
        else:
            print(metric.name)
            assert 0


val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_val_metric,
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)
val_metric = [
    dict(type="PoseMAE", name=["Roll", "Pitch", "Yaw", "MAE"], modeltype="pp"),
    dict(
        type="Eye3dMAE",
        name=["Eye3dMAE_x_left", "Eye3dMAE_y_left", "Eye3dMAE_z_left"],
        is_left=True,
    ),
    dict(
        type="Eye3dMAE",
        name=["Eye3dMAE_x_right", "Eye3dMAE_y_right", "Eye3dMAE_z_right"],
        is_left=False,
    ),
]
float_predictor = dict(
    type="Predictor",
    model=copy.deepcopy(deploy_model),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    callbacks=[val_metric_updater],
    metrics=val_metric,
    log_interval=log_freq,
)
