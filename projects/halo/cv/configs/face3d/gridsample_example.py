import copy
import os

import numpy as np
import torch
from face3d_datahub import get_dataset
from horizon_plugin_pytorch.quantization import March

from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.models.backbones.mixvargenet import MixVarGENetConfig

try:
    import lpips
except ImportError:
    raise ImportError("Please install lpips")
import warnings

from hat.data.transforms.detection import ToTensor
from hat.data.transforms.face3d import (
    CropRoIJitter,
    SimpleNormGenGridMap,
    SimpleNormPositionEncoding,
)
from hat.data.transforms.gaze import RandomColorJitter  # noqa

warnings.filterwarnings("ignore")

DEBUG = False
INPUT_SIZE = 128
TARGET_SIZE = 256
BACKBONE = "mixvargenet0.75"
EXP_ID = "exp16"
VIR_K = np.array([[1750, 0, 800], [0, 1750, 500], [0, 0, 1]])
USE_FLOAT_PRETRAIN = True

training_step = os.environ.get("HAT_TRAINING_STEP", "float")
task_name = f"gridsample_{BACKBONE}_{EXP_ID}"

if DEBUG:
    batch_size_per_gpu = 16
    pretrain_batch_size_per_gpu = 96
    device_ids = [1, 2, 3]
else:
    batch_size_per_gpu = 64
    pretrain_batch_size_per_gpu = 96
    device_ids = [0, 1, 2, 3, 4, 5, 6, 7]

ckpt_dir = "./tmp_models/%s" % task_name
log_dir = "./tmp_models/%s/log" % task_name
log_freq = 50
enable_tensorboard = True
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.BAYES
use_dist = False
# profiler = dict(type="SimpleProfiler")
# profiler = dict(
#     type="PythonProfiler",
#     dirpath="profiler",
#     filename="profile.log",
# )
train_datasets = [
    "workshop_10270_dms_train_crop640",
    "workshop_10293_dms_train_crop640",
    "workshop_10307_dms_train_crop640",
]
val_datasets = [
    "workshop_10270_dms_val_crop640",
    "workshop_10293_dms_val_crop640",
    "workshop_10307_dms_val_crop640",
    "cd569_10308_ov2311_val_crop640",
]
float_lr = [40, 5e-4, [15, 30]]
# float_lr = [80, 1e-3, [25, 60]
pretrain_lr = [120, 5e-4, [45, 80, 110]]
freeze_bn_lr = [5, 1e-5, [10]]
qat_lr = [10, 1e-5, [25, 35]]

PRETRAINS = {
    "resnet50": "/horizon-bucket/HDLTAlgorithm/models/bayes_release_models/resnet50_cls/float-checkpoint-best.pth.tar",  # noqa
    "vargnet1.0": "/horizon-bucket/HDLTAlgorithm/models/bayes_release_models/vargnetv2_cls/float-checkpoint-best.pth.tar",  # noqa
    "mixvargenet0.75": "/horizon-bucket/HDLTAlgorithm/models/bayes_release_models/mixvargenet_cls_alpha075/float-checkpoint-best.pth.tar",  # noqa
}
pretrain_load_checkpoint_path = PRETRAINS[BACKBONE]

if not USE_FLOAT_PRETRAIN:
    float_load_checkpoint_path = (
        f"tmp_models/{task_name}/pretrain-checkpoint-last.pth.tar"  # noqa
    )
else:
    float_load_checkpoint_path = "/horizon-bucket/interaction/models/yuhao.dou/gridsample_mixvargenet_exp03/float-checkpoint-last.pth.tar"  # noqa

qat_load_checkpoint_path = os.path.join(
    ckpt_dir, "float-checkpoint-last.pth.tar"
)

# =========================================model==============================
mixvargenet_size = 0.75
net_config = [
    [
        MixVarGENetConfig(
            in_channels=16,
            out_channels=16,
            head_op="mixvarge_f4_gb16",
            stack_ops=[],
            stack_factor=1,
            stride=1,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # stride 2
    ],
    [
        MixVarGENetConfig(
            in_channels=16,
            out_channels=32,
            head_op="mixvarge_f4_gb16",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=2,
        ),  # stride 4
    ],
    [
        MixVarGENetConfig(
            in_channels=32,
            out_channels=48,
            head_op="mixvarge_f4_gb16",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[4],
            extra_downsample_num=2,
        ),  # stride 8
    ],
    [
        MixVarGENetConfig(
            in_channels=48,
            out_channels=96,
            head_op="mixvarge_f4_gb16",
            stack_ops=[
                "mixvarge_f4_gb16",
                "mixvarge_f4_gb16",
                "mixvarge_f4_gb16",
                "mixvarge_f4_gb16",
                "mixvarge_f4_gb16",
                "mixvarge_f4_gb16",
            ],
            stack_factor=1,
            stride=2,
            fusion_strides=[4, 8],
            extra_downsample_num=2,
        ),  # stride 16
    ],
    [
        MixVarGENetConfig(
            in_channels=96,
            out_channels=192,
            head_op="mixvarge_f2_gb16",
            stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
            stack_factor=1,
            stride=2,
            fusion_strides=[8, 16],
            extra_downsample_num=0,
        ),  # stride 32
    ],
]
bn_kwargs = dict(eps=2e-5, momentum=0.1)

ARCHS = {
    "resnet50": dict(
        type="ResNet50",
        num_classes=1000,
        bn_kwargs={},
        include_top=False,
    ),
    "vargnet1.0": dict(
        type="VargNetV2",
        alpha=1.0,
        input_channels=3,
        num_classes=1000,
        include_top=False,
        disable_quanti_input=True,
        bn_kwargs={},
    ),
    "mixvargenet0.75": dict(
        type="MixVarGENet",
        net_config=net_config,
        num_classes=1000,
        bn_kwargs=bn_kwargs,
        include_top=False,
        bias=True,
    ),
}
NUM_CHANNELS = {
    "resnet50": 2048,
    "vargnet1.0": 256,
    "mixvargenet0.75": 192,
}

pretrain_model = dict(
    type="Face3dModel",
    use_grid_sample=True,
    backbone=ARCHS[BACKBONE],
    head=dict(
        type="Face3dHead",
        kernel_size=INPUT_SIZE // 32,
        in_channels=NUM_CHANNELS[BACKBONE],
    ),
    post_process=dict(
        type="FLAMEProcess",
        use_dist=use_dist,
        flame=dict(
            type="FLAME",
            flame_model_path="face3d_data/new_generic_model.pkl",
            flame_lmk_embedding_path="face3d_data/fined_landmark_embedding.npy",
            # flame_model_path="face3d_data/flame/flame2023_no_jaw.pkl",
            # flame_lmk_embedding_path="face3d_data/flame/2dw3d_landmark_embedding.npy",
        ),
        max_depth_meter=2.0,
        model_type="pp",
    ),
    loss=dict(
        type="Face3dLoss",
        consistency_nums=0,
        lpips=lpips.LPIPS(
            pretrained=True,
            pnet_rand=True,
            model_path="face3d_data/lpips.pth",
            verbose=True,
            net="vgg",
        ),
        loss_weights={
            "img_ldmk": 1.0,  # 256.0,
            "cam_ldmk": 0,
            "cam_verts": 0,
            "photo": 0,  # 10.0,
            "lpips": 0,  # 30.0,
            "eye3d": 0,
            "depth": 0,
            "eyelid": 0,
            "global_pose": 512,
            "transl": 512,
            "shape": 0,
            "jaw": 0,
            "exp": 0,
            "tex": 0,
            "light": 0,
            "shape_cons": 0,
            "shape_reg": 5e-2,
            "exp_reg": 1e-2,
            "tex_reg": 0,
        },
    ),
)

model = copy.deepcopy(pretrain_model)
model["post_process"]["flame_tex"] = dict(
    type="FLAMETex", tex_path="face3d_data/FLAME_albedo_from_BFM.npz"
)
model["post_process"]["renderer"] = dict(
    type="NVRenderer",
    obj_filename="face3d_data/head_template_mesh.obj",
)
model["post_process"]["render_img"] = True
model["loss"]["loss_weights"] = {
    "img_ldmk": 1.0,  # 256.0,
    "cam_ldmk": 0,
    "cam_verts": 0,
    "photo": 10,  # 10.0,
    "lpips": 30,  # 30.0,
    "eye3d": 2.0,
    "depth": 0,
    "eyelid": 0,
    "global_pose": 0,
    "transl": 1,
    "shape": 0,
    "jaw": 0,
    "exp": 0,
    "tex": 0,
    "light": 0,
    "shape_cons": 0.0,
    "shape_reg": 3e-2,
    "exp_reg": 1e-2,
    "tex_reg": 1e-2,
}
# validation model
val_model = copy.deepcopy(model)
val_model["post_process"]["render_img"] = False
val_model["post_process"]["render_depth"] = False

# Deploy Model for saving checkpoint
pretrain_deploy_model = copy.deepcopy(pretrain_model)
pretrain_deploy_model["deploy"] = True
pretrain_deploy_model["loss_weights"] = None
deploy_model = copy.deepcopy(model)
deploy_model["deploy"] = True
deploy_inputs = dict(img=torch.randn((1, 3, INPUT_SIZE, INPUT_SIZE)))

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
            CropRoIJitter(
                jitter_prob=0.8,
                exp_ratio=1.0,
                exp_jitter=0.05,
                center_shift=0.05,
            ),
            RandomColorJitter(
                brightness=0.5,
                contrast=0.5,
                prob=1.0,
            ),
            dict(
                type="RandomOcclusion",
                occ_type=["jaw", "forehead", "whole"],
                occ_ratio=0.2,
                prob=0.7,
                size_ratio=0.3,
            ),
            SimpleNormGenGridMap(
                net_input_size=INPUT_SIZE,
                net_target_size=TARGET_SIZE,
                norm_ratio=1.2,
                expand_crop_hw=640,
                norm_method="longside_square",
                use_dist=use_dist,
                virtual_intrinsic=VIR_K,
            ),
            SimpleNormPositionEncoding(
                net_input_size=INPUT_SIZE,
            ),
            ToTensor(),
        ],
        independent_transform_list=[],  # independent_transform_list,
        stage="finetune",
        modeltype="pp",
        only_y_channel=True,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)


pretrain_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Face3dDataset",
        image_path_list=train_data["image_list"],
        mask_path_list=train_data["mask_list"],
        anno_path_list=train_data["anno_list"],
        transforms=[
            CropRoIJitter(
                jitter_prob=0.8,
                exp_ratio=1.0,
                exp_jitter=0.05,
                center_shift=0.05,
            ),
            RandomColorJitter(
                brightness=0.5,
                contrast=0.5,
                prob=1.0,
            ),
            SimpleNormGenGridMap(
                net_input_size=INPUT_SIZE,
                net_target_size=TARGET_SIZE,
                norm_ratio=1.2,
                expand_crop_hw=640,
                norm_method="longside_square",
                use_dist=use_dist,
            ),
            SimpleNormPositionEncoding(
                net_input_size=INPUT_SIZE,
            ),
            ToTensor(),
        ],
        independent_transform_list=[],  # independent_transform_list,
        stage="pretrain",
        modeltype="pp",
        only_y_channel=True,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=pretrain_batch_size_per_gpu,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

val_data = get_dataset(val_datasets)
val_data_loader = [
    dict(
        type=torch.utils.data.DataLoader,
        dataset=dict(
            type="Face3dDataset",
            image_path_list=[image_val_data],
            mask_path_list=[mask_val_data],
            anno_path_list=[anno_val_data],
            transforms=[
                dict(
                    type="SimpleNormGenGridMap",
                    net_input_size=INPUT_SIZE,
                    net_target_size=TARGET_SIZE,
                    norm_ratio=1.2,
                    expand_crop_hw=640,
                    norm_method="longside_square",
                    use_dist=use_dist,
                    virtual_intrinsic=VIR_K,
                ),
                dict(
                    type="SimpleNormPositionEncoding",
                    net_input_size=INPUT_SIZE,
                ),
                dict(type="ToTensor"),
            ],
            stage="predict",
            modeltype="pp",
            only_y_channel=True,
        ),
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )
    for image_val_data, mask_val_data, anno_val_data in zip(
        val_data["image_list"], val_data["mask_list"], val_data["anno_list"]
    )
]

batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    batch_transforms=[
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
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
)


# =========================================callback==============================


def update_metric(metrics, batch, model_outs):
    for metric in metrics:
        if metric.name in model_outs.keys():
            metric.update(model_outs[metric.name])


def update_val_metric(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(batch["label"], model_outs)


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
    step_log_freq=1000,
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
    mode=None,
    monitor_metric_key="MAE",
)

val_metric = [
    dict(
        type="PoseMAE",
        model_type="pp",
        keys=["dms00", "dms01", "dms02", "dms03"],
    ),
    dict(type="Eye3dMAE", keys=["dms00", "dms01", "dms02", "dms03"]),
    # dict(type="Face3dSTD")
]

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=model,
    val_on_train_end=False,
    log_interval=log_freq,
)

pretrain_val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=pretrain_model,
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
            warmup_len=0,
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
        dict(type="LossShow", name="img_ldmk"),
        dict(type="LossShow", name="flame"),
        dict(type="LossShow", name="shape_reg"),
        dict(type="LossShow", name="exp_reg"),
        dict(type="LossShow", name="total_loss"),
    ],
    val_metrics=val_metric,
    find_unused_parameters=True,
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
    stop_by="step" if DEBUG else "epoch",
    num_epochs=float_lr[0],
    num_steps=1000,
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
        ckpt_callback,
    ],
    sync_bn=False,
    train_metrics=[
        dict(type="LossShow", name="img_ldmk"),
        dict(type="LossShow", name="eye3d"),
        dict(type="LossShow", name="flame"),
        dict(type="LossShow", name="photo"),
        dict(type="LossShow", name="lpips"),
        dict(type="LossShow", name="shape_reg"),
        dict(type="LossShow", name="exp_reg"),
        dict(type="LossShow", name="tex_reg"),
        dict(type="LossShow", name="total_loss"),
    ],
    val_metrics=val_metric,
    find_unused_parameters=True,
    # profiler=profiler,
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
    model=copy.deepcopy(pretrain_model),
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    callbacks=[val_metric_updater],
    metrics=val_metric,
    log_interval=log_freq,
)

float_predictor = dict(
    type="Predictor",
    model=copy.deepcopy(val_model),
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-last.pth.tar"
                ),
                allow_miss=False,
                ignore_extra=True,
                verbose=True,  # Show unexpect_key and miss_key info.
            ),
        ],
    ),
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    callbacks=val_metric_updater,
    metrics=val_metric,
    log_interval=log_freq,
)

# qat_predictor = dict(
#     type="Predictor",
#     model=copy.deepcopy(deploy_model),
#     model_convert_pipeline=dict(
#         type="ModelConvertPipeline",
#         qat_mode="fuse_bn",
#         converters=[
#             dict(type="Float2QAT"),
#             dict(
#                 type="LoadCheckpoint",
#                 checkpoint_path=os.path.join(
#                     ckpt_dir, "qat-checkpoint-best.pth.tar"
#                 ),
#             ),
#         ],
#     ),
#     data_loader=val_data_loader,
#     batch_processor=val_batch_processor,
#     device=None,
#     callbacks=[val_metric_updater],
#     metrics=val_metric,
#     log_interval=log_freq,
# )


# int_infer_predictor = dict(
#     type="Predictor",
#     model=copy.deepcopy(deploy_model),
#     model_convert_pipeline=dict(
#         type="ModelConvertPipeline",
#         qat_mode="fuse_bn",
#         converters=[
#             dict(type="Float2QAT"),
#             dict(type="QAT2Quantize"),
#         ],
#     ),
#     data_loader=val_data_loader,
#     batch_processor=val_batch_processor,
#     device=None,
#     callbacks=[val_metric_updater],
#     metrics=val_metric,
#     log_interval=log_freq,
# )
