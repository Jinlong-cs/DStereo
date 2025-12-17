import copy
import os

import torch
import torchvision
from face3d_datahub import get_dataset
from horizon_plugin_pytorch.quantization import March

from hat.engine.processors.loss_collector import collect_loss_by_index

try:
    import lpips
except ImportError:
    raise ImportError("Please install lpips")
import warnings

from hat.data.transforms.face3d import (
    CropRoIJitter,
    GridMask,
    SimpleNormGenGridMap,
    SimpleNormPositionEncoding,
    ToTensor,
)

warnings.filterwarnings("ignore")

DEBUG = True

training_step = os.environ.get("HAT_TRAINING_STEP", "float")
task_name = "face3d_pp_norm_use_clahe_aug_bigp_cons"

if DEBUG:
    batch_size_per_gpu = 4
    pretrain_batch_size_per_gpu = 4
    device_ids = [3]  # 2, 3]
else:
    batch_size_per_gpu = 24
    pretrain_batch_size_per_gpu = 96
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
train_datasets = ["workshop_train_pts_c"]
val_datasets = [
    "workshop_val_pts_c_cam09"
]  # , workshop_val_pts_c_cam09, workshop_val_pts_c workshop_val_large, workshop_val
float_lr = [80, 5e-4, [25, 60]]
pretrain_lr = [120, 1e-3, [35, 90, 110]]
freeze_bn_lr = [5, 1e-5, [10]]
qat_lr = [10, 1e-5, [25, 35]]
pretrain_load_checkpoint_path = "tmp_models/face3d_pp_norm_use_clahe_aug_cons/pretrain-checkpoint-last.pth.tar"
float_load_checkpoint_path = "tmp_models/face3d_pp_norm_use_clahe_aug_bigp_cons/pretrain-checkpoint-last.pth.tar"
freeze_bn_checkpoint_path = os.path.join(
    ckpt_dir, "float-checkpoint-last.pth.tar"
)
qat_load_checkpoint_path = os.path.join(
    ckpt_dir, "float-checkpoint-last.pth.tar"
)

# ================ transform ====================
transforms_1 = torchvision.transforms.Compose(
    [
        SimpleNormGenGridMap(
            net_input_size=128,
            # center_shift_prob=1.0,
            # center_shift_range=0.025,
            norm_ratio=1.2,
            expand_crop_hw=640,
            norm_method="longside_square",
            undistort=undistort,
            # norm_jitter_range=0.1,
            # virtual_intrinsic=np.array([[1000, 0, 960], [0, 1000, 640], [0, 0, 1]]),
        ),
        SimpleNormPositionEncoding(
            net_input_size=128,
        ),
        ToTensor(),
    ]
)
transforms_2 = torchvision.transforms.Compose(
    [
        CropRoIJitter(
            jitter_prob=0.8,
            exp_ratio=1.0,
            exp_jitter=0.05,
            center_shift=0.05,
        ),
        SimpleNormGenGridMap(
            net_input_size=128,
            # center_shift_prob=1.0,
            # center_shift_range=0.025,
            norm_ratio=1.2,
            expand_crop_hw=640,
            norm_method="longside_square",
            undistort=undistort,
            # norm_jitter_range=0.1,
            # virtual_intrinsic=np.array([[1000, 0, 960], [0, 1000, 640], [0, 0, 1]]),
        ),
        SimpleNormPositionEncoding(
            net_input_size=128,
        ),
        GridMask(
            use_h=True,
            use_w=True,
            rotate=10,
            offset=False,
            ratio=0.85,
            mode=1,
            prob=0.8,
            limit_d_ratio_min=0.2,
            limit_d_ratio_max=0.35,
        ),
        ToTensor(),
    ]
)

independent_transform_list = [transforms_1, transforms_2]


# =========================================model==============================
pretrain_model = dict(
    type="Face3dModel",
    modeltype="pp",
    ldmk_norm=True,
    undistort=undistort,
    use_flame=True,
    use_grid_sample=True,
    consistency_nums=len(independent_transform_list),
    max_depth_meter=2.0,
    backbone=dict(
        type="VargNetV2",
        alpha=1.0,
        input_channels=5,
        num_classes=1000,
        include_top=False,
        disable_quanti_input=True,
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
    lpips=lpips.LPIPS(
        pretrained=True,
        pnet_rand=True,
        model_path="face3d_data/lpips.pth",
        verbose=True,
        net="vgg",
    ),
    loss_weights={
        "ldmk": 512,  # 256.0,
        "photo": 0,  # 10.0,
        "lpips": 0,  # 30.0,
        "eye3d": 0,
        "shape_reg": 5e-2,
        "exp_reg": 1e-2,
        "tex_reg": 1e-2,
        "flame_transl_xy_reg": 128,
        "flame_transl_z_reg": 640,
        "flame_global_pose_reg": 0,
        "shape_cons": 5,
    },
)
model = copy.deepcopy(pretrain_model)
model["flame_tex"] = dict(
    type="FLAMETex", tex_path="face3d_data/FLAME_albedo_from_BFM.npz"
)
model["renderer"] = dict(
    type="NVRenderer",
    obj_filename="face3d_data/head_template_mesh.obj",
    render_size=(256, 256),
)
model["loss_weights"] = {
    "ldmk": 512,  # 256.0,
    "photo": 10,  # 10.0,
    "lpips": 30,  # 30.0,
    "eye3d": [2000, 2000, 500],
    "shape_reg": 3e-2,
    "exp_reg": 1e-2,
    "tex_reg": 1e-2,
    "flame_transl_xy_reg": 1,  # 128,
    "flame_transl_z_reg": 1,  # 640,
    "shape_cons": 5,
}
# Deploy Model for saving checkpoint
pretrain_deploy_model = copy.deepcopy(pretrain_model)
pretrain_deploy_model["deploy"] = True
pretrain_deploy_model["loss_weights"] = None
deploy_model = copy.deepcopy(model)
deploy_model["deploy"] = True
deploy_model["loss_weights"] = None
deploy_inputs = dict(img=torch.randn((1, 3, 128, 128)))

# =========================================data==============================
train_data = get_dataset(train_datasets)
data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Face3dDataset",
        image_path_list=train_data["image_list"],
        mask_path_list=train_data["mask_list"],
        anno_path_list=train_data["anno_list"],
        transforms=[
            dict(type="ImageCLAHE"),
        ],
        independent_transform_list=independent_transform_list,
        stage="finetune",
        modeltype="pp",
        is_gray=False,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

pretrain_data_loader = copy.deepcopy(data_loader)
pretrain_data_loader["batch_size"] = pretrain_batch_size_per_gpu

val_data = get_dataset(val_datasets)
val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Face3dDataset",
        image_path_list=val_data["image_list"],
        mask_path_list=val_data["mask_list"],
        anno_path_list=val_data["anno_list"],
        transforms=[
            dict(type="ImageCLAHE"),
            dict(
                type="SimpleNormGenGridMap",
                net_input_size=128,
                norm_ratio=1.2,
                expand_crop_hw=640,
                norm_method="longside_square",
                undistort=undistort,
                # virtual_intrinsic=np.array([[1000, 0, 960], [0, 1000, 640], [0, 0, 1]]),
            ),
            dict(type="SimpleNormPositionEncoding", net_input_size=128),
            dict(type="ToTensor"),
        ],
        stage="finetune",
        modeltype="pp",
        is_gray=False,
    ),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=2,
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


# =========================================callback==============================


def update_metric(metrics, batch, model_outs):
    for metric, loss in zip(metrics, model_outs):
        metric.update(loss)


def update_val_metric(metrics, batch, model_outs):
    for metric in metrics:
        if isinstance(metric.name, list) and len(metric.name) == 4:
            metric.update(batch, model_outs["pose"])
        elif "left" in metric.name[0]:
            metric.update(batch, model_outs["left_eye_pred"])
        elif "right" in metric.name[0]:
            metric.update(batch, model_outs["right_eye_pred"])
        else:
            print(metric.name)
            assert 0


metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
)

val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_val_metric,
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=log_freq,
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
    save_interval=1,
    strict_match=True,
    mode="min",
    monitor_metric_key="MAE",
)

freeze_bn_ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix="freeze_bn-" + training_step + "-",
    save_interval=1,
    strict_match=True,
    mode="min",
    monitor_metric_key="MAE",
)

freeze_module_callback = dict(
    type="FreezeModule",
    # modules=[["face_ldmk_backbone", "face3d_finetune_post_module", "face3d_finetune_head"]],
    modules=[["backbone"]],
    step_or_epoch=[0],
    update_by="epoch",
    only_batchnorm=True,
)

val_metric = [
    dict(type="PoseMAE", name=["Roll", "Pitch", "Yaw", "MAE"], modeltype="pp"),
    dict(
        type="Eye3dMAE",
        name=["Eye_x_left", "Eye_y_left", "Eye_z_left"],
        is_left=True,
    ),
    dict(
        type="Eye3dMAE",
        name=["Eye_x_right", "Eye_y_right", "Eye_z_right"],
        is_left=False,
    ),
]

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=deploy_model,
    val_on_train_end=False,
    log_interval=log_freq,
)

pretrain_val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=pretrain_deploy_model,
    val_on_train_end=False,
    log_interval=log_freq,
)

# ==========================train=======================
pretrain_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=pretrain_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=pretrain_load_checkpoint_path,
                allow_miss=True,
                ignore_extra=True,
            ),
        ],
    )
    if pretrain_load_checkpoint_path
    else None,
    data_loader=pretrain_data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        params={"weight": dict(weight_decay=4e-5)},
        lr=pretrain_lr[1],
    ),
    batch_processor=batch_processor,
    device=None,
    stop_by="epoch" if DEBUG else "epoch",
    num_epochs=pretrain_lr[0],
    num_steps=200,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            warmup_by="epoch",
            warmup_len=3,
            step_log_interval=log_freq,
            lr_decay_id=pretrain_lr[2],
            lr_decay_factor=0.1,
        ),
        metric_updater,
        # pretrain_val_callback,
        ckpt_callback,
    ],
    sync_bn=False,
    train_metrics=[
        dict(type="LossShow", name="Loss"),
        dict(type="LossShow", name="Lmk"),
        # dict(type="LossShow", name="Photo"),
        # dict(type="LossShow", name="Lpips"),
        dict(type="LossShow", name="Shape"),
        dict(type="LossShow", name="Exp"),
        dict(type="LossShow", name="Tex"),
        dict(type="LossShow", name="FLAME_reg"),
    ],
    val_metrics=val_metric,
)

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=float_load_checkpoint_path,
                allow_miss=True,
                ignore_extra=True,
            ),
        ],
    )
    if float_load_checkpoint_path
    else None,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        params={"weight": dict(weight_decay=4e-5)},
        lr=float_lr[1],
    ),
    batch_processor=batch_processor,
    device=None,
    stop_by="epoch" if DEBUG else "epoch",
    num_epochs=float_lr[0],
    num_steps=200,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            warmup_by="epoch",
            warmup_len=1,
            step_log_interval=log_freq,
            lr_decay_id=float_lr[2],
            lr_decay_factor=0.1,
        ),
        metric_updater,
        # val_callback,
        ckpt_callback,
        # trace_callback,
    ],
    sync_bn=False,
    train_metrics=[
        dict(type="LossShow", name="Loss"),
        dict(type="LossShow", name="Lmk"),
        dict(type="LossShow", name="Eye3d"),
        dict(type="LossShow", name="Photo"),
        dict(type="LossShow", name="Lpips"),
        dict(type="LossShow", name="Shape"),
        dict(type="LossShow", name="Exp"),
        dict(type="LossShow", name="Tex"),
        dict(type="LossShow", name="FLAME_reg"),
    ],
    val_metrics=val_metric,
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
                checkpoint_path=qat_load_checkpoint_path,
            ),
            dict(type="Float2QAT"),
        ],
    )
    if qat_load_checkpoint_path
    else None,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        params={"weight": dict(weight_decay=0)},
        lr=1e-5,
    ),
    batch_processor=batch_processor,
    device=None,
    stop_by="epoch",
    num_epochs=1,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            warmup_by="epoch",
            warmup_len=1,
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
        dict(type="LossShow", name="Lmk"),
        dict(type="LossShow", name="Photo"),
        dict(type="LossShow", name="Lpips"),
        dict(type="LossShow", name="Shape"),
        dict(type="LossShow", name="Exp"),
        dict(type="LossShow", name="Tex"),
        dict(type="LossShow", name="FLAME_reg"),
    ],
    val_metrics=val_metric,
)


freeze_bn_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=freeze_bn_checkpoint_path,
                ignore_extra=True,
                allow_miss=True,
            ),
        ],
    )
    if freeze_bn_checkpoint_path
    else None,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        params={"weight": dict(weight_decay=4e-5)},
        lr=freeze_bn_lr[1],
    ),
    batch_processor=batch_processor,
    device=None,
    stop_by="epoch" if DEBUG else "epoch",
    num_epochs=freeze_bn_lr[0],
    num_steps=300,
    callbacks=[
        stat_callback,
        freeze_module_callback,
        dict(
            type="StepDecayLrUpdater",
            warmup_by="epoch",
            warmup_len=1,
            step_log_interval=log_freq,
            lr_decay_id=freeze_bn_lr[2],
            lr_decay_factor=0.1,
        ),
        metric_updater,
        val_callback,
        freeze_bn_ckpt_callback,
    ],
    sync_bn=False,
    train_metrics=[
        dict(type="LossShow", name="Loss"),
        dict(type="LossShow", name="Lmk"),
        dict(type="LossShow", name="Photo"),
        dict(type="LossShow", name="Lpips"),
        dict(type="LossShow", name="Shape"),
        dict(type="LossShow", name="Exp"),
        dict(type="LossShow", name="Tex"),
        dict(type="LossShow", name="FLAME_reg"),
    ],
    val_metrics=val_metric,
    # find_unused_parameters=False,
)
# ==========================compile=======================
compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name="dms_face_3d_pose_pp",
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "dms_face_3d_pose_pp.hbm"),
    layer_details=True,
    input_source=["ddr", "pyramid", "ddr"],
)

# ==========================predict=======================
pretrain_predictor = dict(
    type="Predictor",
    model=copy.deepcopy(pretrain_deploy_model),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    callbacks=[val_metric_updater],
    metrics=val_metric,
    log_interval=log_freq,
)

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

qat_predictor = dict(
    type="Predictor",
    model=copy.deepcopy(deploy_model),
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
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    callbacks=[val_metric_updater],
    metrics=val_metric,
    log_interval=log_freq,
)


int_infer_predictor = dict(
    type="Predictor",
    model=copy.deepcopy(deploy_model),
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(type="QAT2Quantize"),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    callbacks=[val_metric_updater],
    metrics=val_metric,
    log_interval=log_freq,
)
