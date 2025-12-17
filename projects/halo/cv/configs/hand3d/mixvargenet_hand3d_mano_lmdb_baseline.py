"""
1. 集群运行
python3 submitv2.py --config ../../projects/halo/cv/configs/hand3d/cluster_config2.py  --cluster project-3090-halo-cv-bcloud
2. 本地运行
python3 -W ignore tools/train.py --config projects/halo/cv/configs/hand3d/mixvargenet_hand3d_mano_demo.py --stage float

- mixvarg_bifpn_fusion_multifc_v1
- mixvarg_pan_fusion_multifc_v1
- mixvarg_fpn_fusion_multifc_v1
"""

import os
from copy import deepcopy

import numpy as np
import torch
from horizon_plugin_pytorch.quantization import March

from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.utils.config import ConfigVersion
from projects.halo.cv.configs.hand3d.model_hub.mixvarg_fpn_fusion_multifc_v1 import (
    BIFPN_CHANNELS,
    BUCKET_ROOT,
    ENABLE_GRID_SAMPLE,
    ENABLE_HEATMAP_HEAD,
    ENABLE_INSHAPE_UNIFORM,
    ENABLE_JOINTS_25D,
    ENABLE_RENDER_HEAD,
    ENCODING_CHANNELS,
    FLAT_HAND_MEAN,
    INPUT_CHANNELS,
    INPUT_IMAGE_SIZE,
    IS_LOCAL_TRAIN,
    MANO_CENTER_IDX,
    MANO_HAND_SIDE,
    MANO_ROOT,
    SMPL_HANDS_MEAN,
    VIRTUAL_CAMERA_FOCAL,
    VIRTUAL_CROP_SIZE,
    VIRTUAL_IMAGE_HW,
    VIRTUAL_NORM_RATIO,
    get_model,
)

# --------- global params -----------
VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

# --------- task params -----------
DEBUG = False
SHIP = "J5"
task_name = "dms_hand_mano_classification"

if DEBUG or IS_LOCAL_TRAIN:
    batch_size_per_gpu = 32
    device_ids = [0, 1]
    num_workers = 0
else:
    batch_size_per_gpu = 128
    device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    num_workers = 24

stop_by = "step" if DEBUG else "epoch"
num_steps = 5000
num_epochs = 60

float_lr = 5e-5 * np.sqrt(batch_size_per_gpu)
qnn_lr = 1e-5
stop_by_qat = "epoch"
num_epochs_qat = 30
num_steps_qat = 15
step_epochs_qat = 0.5

ckpt_dir = "./tmp_models/%s" % task_name
log_dir = "./tmp_models/%s/log" % task_name
if not os.path.isdir("./tmp_models"):
    os.makedirs("./tmp_models")
if not os.path.isdir(ckpt_dir):
    os.makedirs(ckpt_dir)
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

log_freq = 50
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.BAYES if SHIP == "J5" else March.BERNOULLI2

# --------- models -----------
model = get_model(
    input_image_size=INPUT_IMAGE_SIZE,
    input_channels=INPUT_CHANNELS,
    bifpn_channels=BIFPN_CHANNELS,
    encoding_channels=ENCODING_CHANNELS,
    enable_grid_sample=ENABLE_GRID_SAMPLE,
    enable_heatmap_head=ENABLE_HEATMAP_HEAD,
    enable_render_head=ENABLE_RENDER_HEAD,
    enable_joints_25d=ENABLE_JOINTS_25D,
    mano_root=MANO_ROOT,
    mano_hand_side=MANO_HAND_SIDE,
    mano_center_idx=MANO_CENTER_IDX,
    flat_hand_mean=FLAT_HAND_MEAN,
    smpl_hands_mean=SMPL_HANDS_MEAN,
)
val_model = deepcopy(model)
val_model["loss"] = None

# Deploy Model for saving checkpoint
deploy_model = deepcopy(val_model)
deploy_model["decoder"] = None
deploy_model["head"]["deploy"] = True

deploy_inputs = dict(
    img=torch.randn(4, INPUT_CHANNELS, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
)
deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2QAT"),
        dict(type="QAT2Quantize"),
    ],
)


# --------- dataloader -----------
def convert_lmdb(_lmdb_path_list, bucket_root=BUCKET_ROOT):
    _lmdb_list = []
    for lmdb in _lmdb_path_list:
        if lmdb[0].lower().startswith("multimode"):
            _lmdb_list.append(
                [
                    os.path.join(bucket_root, lmdb[0], "image_lmdb"),
                    os.path.join(bucket_root, lmdb[0], "anno_lmdb"),
                    None,
                    None,
                    lmdb[1],
                ]
            )
        else:
            _lmdb_list.append(
                [
                    os.path.join(lmdb[0], "image_lmdb"),
                    os.path.join(lmdb[0], "anno_lmdb"),
                    None,
                    None,
                    lmdb[1],
                ]
            )
    return _lmdb_list


# fmt: off
lmdb_list = convert_lmdb(
    [
        ["MultiMode_2/dynamic_gesture/datahub/lmdb_v2/FerihandIndexV1b_training", 1],
        ["MultiMode_2/dynamic_gesture/datahub/lmdb_v2/Cvpr2019IndexV1b_training", 1],
        ["MultiMode_2/qiucheng.shen/Hand3d/DataHub/lmdb_v1/Fitting2305V1b_training", 1],
    ]
)
# fmt: on


virtual_intrinsic = np.diag(
    [VIRTUAL_CAMERA_FOCAL, VIRTUAL_CAMERA_FOCAL, 1]
).astype(np.float32)
virtual_intrinsic[:2, -1] = [
    VIRTUAL_IMAGE_HW[1] // 2,
    VIRTUAL_IMAGE_HW[0] // 2,
]
data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ConcatDataset",
        with_flag=False,
        record_index=False,
        datasets=[
            dict(
                type="RepeatDataset",
                times=lmdb_list[i][-1],
                dataset=dict(
                    type="Hand3dLmdbSingleDataset",
                    image_path=lmdb_list[i][0],
                    anno_path=lmdb_list[i][1],
                    mask_path=lmdb_list[i][2],
                    depth_path=lmdb_list[i][3],
                    set_name="training",
                    enable_inshape_uniform=ENABLE_INSHAPE_UNIFORM,
                    virtual_img_shape=VIRTUAL_IMAGE_HW,
                    transforms=[
                        dict(
                            type="SimpleNormGenGridMap",
                            net_input_size=INPUT_IMAGE_SIZE,
                            norm_ratio=VIRTUAL_NORM_RATIO,
                            expand_crop_hw=VIRTUAL_CROP_SIZE,
                            norm_method="longside_square",
                            virtual_intrinsic=virtual_intrinsic,
                        ),
                        dict(
                            type="SimpleNormPositionEncoding",
                            net_input_size=INPUT_IMAGE_SIZE,
                        ),
                        dict(type="ToTensor", to_yuv=False),
                    ],
                ),
            )
            for i in range(len(lmdb_list))
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=False,
    drop_last=True,
)


# fmt: off
val_lmdb_list = convert_lmdb(
    [
        ["MultiMode_2/qiucheng.shen/Hand3d/DataHub/lmdb_v1/Fitting2305V1a_evaluation", 1],
    ]
)
# fmt: on

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ConcatDataset",
        with_flag=False,
        record_index=False,
        datasets=[
            dict(
                type="RepeatDataset",
                times=val_lmdb_list[i][-1],
                dataset=dict(
                    type="Hand3dLmdbSingleDataset",
                    image_path=val_lmdb_list[i][0],
                    anno_path=val_lmdb_list[i][1],
                    mask_path=val_lmdb_list[i][2],
                    depth_path=val_lmdb_list[i][3],
                    set_name="evaluation",
                    enable_inshape_uniform=ENABLE_INSHAPE_UNIFORM,
                    virtual_img_shape=VIRTUAL_IMAGE_HW,
                    transforms=[
                        dict(
                            type="SimpleNormGenGridMap",
                            net_input_size=INPUT_IMAGE_SIZE,
                            norm_ratio=VIRTUAL_NORM_RATIO,
                            expand_crop_hw=VIRTUAL_CROP_SIZE,
                            norm_method="longside_square",
                            virtual_intrinsic=virtual_intrinsic,
                        ),
                        dict(
                            type="SimpleNormPositionEncoding",
                            net_input_size=INPUT_IMAGE_SIZE,
                        ),
                        dict(type="ToTensor", to_yuv=False),
                    ],
                ),
            )
            for i in range(len(val_lmdb_list))
        ],
    ),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=False,
    drop_last=True,
)

# --------- processor -----------
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
    loss_collector=collect_loss_by_regex("^l_"),
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


# --------- metric -----------
def update_metric(metrics, batch, model_outs):
    convert_map = {
        "tvec": "l_tvec",
        "mano_j3d": "l_mano_ldmk3d",
        "mano_j3d_cam": "l_mano_ldmk3d_cam",
        "reproj2d": "l_reproj_gt",
        "j2d": "l_ldmk2d_gt",
        "j2d_rep": "l_j2d_repro",
        "edge": "l_edge",
        "normal": "l_normal",
        "lapla": "l_laplacian",
        "chamfer": "l_chamfer",
        "text": "l_render_image",
    }
    for metric in metrics:
        metric.update(model_outs[convert_map[metric.name]])


def val_update_metric(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(batch, model_outs)
        name, value = metric.get()
        model_outs[name] = value


metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
)
val_metric_updater = deepcopy(metric_updater)
val_metric_updater["metric_update_func"] = val_update_metric
val_metric_updater["log_prefix"] = "Validation " + task_name

stat_callback = dict(
    type="StatsMonitor",
    log_freq=log_freq,
)

# NOTE: Keep deploy is True to ensure the model in deploy mode.
ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    save_interval=1,
    name_prefix=training_step + "-",
    strict_match=True,
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=val_model,
    val_on_train_end=False,
    log_interval=log_freq,
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,
)

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        params={"weight": dict(weight_decay=0.001)},
        eps=1e-8,
        betas=(0.9, 0.999),
        lr=float_lr,
    ),
    batch_processor=batch_processor,
    device=None,
    stop_by=stop_by,
    num_epochs=num_epochs,
    num_steps=num_steps,
    find_unused_parameters=False,
    callbacks=[
        stat_callback,
        dict(
            type="CosLrUpdater",
            warmup_begin_lr=float_lr,
            warmup_by="step",
            step_log_interval=log_freq,
            stop_lr=1e-8,
        ),
        metric_updater,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow", name="tvec"),
        dict(type="LossShow", name="mano_j3d"),
        dict(type="LossShow", name="mano_j3d_cam"),
        dict(type="LossShow", name="reproj2d"),
        dict(type="LossShow", name="j2d"),
        dict(type="LossShow", name="edge"),
        dict(type="LossShow", name="normal"),
        dict(type="LossShow", name="lapla"),
    ],
    val_metrics=[
        dict(type="Hand3dNME", mode="reproj", name="nme_reproject"),
        dict(type="Hand3dNME", mode="coords", name="nme_ldmk2d"),
        dict(type="Hand3dJPE", mode="root", name="jpe_ldmk3d_root"),
        dict(type="Hand3dJPE", mode="cam", name="jpe_ldmk3d_cam"),
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
                ),
                ignore_extra=True,
                allow_miss=True,
            ),
            dict(type="Float2QAT"),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        params={"weight": dict(weight_decay=0.001)},
        eps=1e-8,
        betas=(0.9, 0.999),
        lr=qnn_lr,
    ),
    batch_processor=batch_processor,
    stop_by=stop_by_qat,
    num_epochs=num_epochs_qat,
    num_steps=num_steps_qat,
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="CosLrUpdater",
            warmup_begin_lr=qnn_lr,
            warmup_by="step",
            step_log_interval=log_freq,
            stop_lr=1e-8,
        ),
        metric_updater,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow", name="tvec"),
        dict(type="LossShow", name="mano_j3d"),
        dict(type="LossShow", name="mano_j3d_cam"),
        dict(type="LossShow", name="reproj2d"),
        dict(type="LossShow", name="j2d"),
        dict(type="LossShow", name="edge"),
        dict(type="LossShow", name="normal"),
        dict(type="LossShow", name="lapla"),
        dict(type="LossShow", name="bmc_bl"),
        dict(type="LossShow", name="bmc_rb"),
        dict(type="LossShow", name="bmc_a"),
    ],
    val_metrics=[
        dict(type="Hand3dNME", mode="reproj", name="nme_reproject"),
        dict(type="Hand3dNME", mode="coords", name="nme_ldmk2d"),
        dict(type="Hand3dJPE", mode="root", name="jpe_ldmk3d_root"),
        dict(type="Hand3dJPE", mode="cam", name="jpe_ldmk3d_cam"),
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
                ),
                ignore_extra=True,
            ),
            dict(type="QAT2Quantize"),
        ],
    ),
    data_loader=None,
    optimizer=None,
    batch_processor=batch_processor,
    num_epochs=0,
    device=None,
    callbacks=[
        ckpt_callback,
        trace_callback,
    ],
)

# ----------------------- compile -----------------------------
compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=task_name,
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "dms_hand_mano_classification.hbm"),
    layer_details=True,
    input_source=["ddr"],
    opt="O2",
    input_layout="NHWC",
    output_layout="NHWC",
)
# python3 tools/deploy/compile_perf.py -c  config.py --ckpt xx
