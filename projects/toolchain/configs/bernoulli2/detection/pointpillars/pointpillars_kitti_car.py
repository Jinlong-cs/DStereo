import copy
import os

import torch
from horizon_plugin_pytorch.quantization import March

from hat.data.collates.collates import collate_lidar3d
from hat.engine.processors.loss_collector import collect_loss_by_index
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "pointpillars_kitti_car_bernoulli2"
batch_size_per_gpu = 16
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]

ckpt_dir = f"./tmp_models/{task_name}"
base_data_dir = "./tmp_data/kitti3d/"

cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.BERNOULLI2
convert_mode = "fx"

# PointPillars settings
norm_cfg = None
use_relu6 = True

# Voxelization cfg
pc_range = [0, -40.0, -3.0, 80.0, 40.0, 1.0]
voxel_size = [0.25, 0.25, 4.0]
max_points_in_voxel = 40
max_voxels_num = 12000

class_names = ["Car"]


# model settings
model = dict(
    type="PointPillarsDetector",
    feature_map_shape=[320, 320],
    quant_begin_neck=True,
    pre_process=dict(
        type="BatchVoxelization",
        pc_range=pc_range,
        voxel_size=voxel_size,
        max_voxels_num=max_voxels_num,
        max_points_in_voxel=max_points_in_voxel,
    ),
    reader=dict(
        type="PillarFeatureNet",
        num_input_features=4,
        num_filters=(16,),
        with_distance=False,
        pool_size=(1, max_points_in_voxel),
        voxel_size=voxel_size,
        pc_range=pc_range,
        bn_kwargs=norm_cfg,
        quantize=False,
        use_conv=True,
        normalize_xyz=True,
    ),
    backbone=dict(
        type="PointPillarScatter",
        num_input_features=16,
        quantize=False,
    ),
    neck=dict(
        type="SECONDNeck",
        in_feature_channel=16,
        down_layer_nums=[3, 5, 5],
        down_layer_strides=[2, 2, 2],
        down_layer_channels=[48, 96, 192],
        up_layer_strides=[1, 2, 4],
        up_layer_channels=[96, 96, 96],
        bn_kwargs=norm_cfg,
        use_relu6=use_relu6,
        quantize=True,
        quant_scale=None,
    ),
    head=dict(
        type="PointPillarsHead",
        num_classes=len(class_names),
        in_channels=sum([96, 96, 96]),
        use_direction_classifier=True,
    ),
    anchor_generator=dict(
        type="Anchor3DGeneratorStride",
        anchor_sizes=[[1.6, 3.9, 1.56]],  # noqa B006
        anchor_strides=[
            [voxel_size[0] * 2, voxel_size[1] * 2, 0.0]
        ],  # noqa B006
        anchor_offsets=[
            [0 + voxel_size[0], -40 + voxel_size[1], -1.78]
        ],  # noqa B006
        rotations=[[0, 1.57]],  # noqa B006
        class_names=class_names,
        match_thresholds=[0.6],
        unmatch_thresholds=[0.45],
    ),
    targets=dict(
        type="LidarTargetAssigner",
        box_coder=dict(
            type="GroundBox3dCoder",
            n_dim=7,
        ),
        class_names=class_names,
        positive_fraction=-1,
    ),
    loss=dict(
        type="PointPillarsLoss",
        num_classes=len(class_names),
        loss_cls=dict(
            type="FocalLossV2",
            alpha=0.25,
            gamma=2.0,
            from_logits=False,
            reduction="none",
            loss_weight=1.0,
        ),
        loss_bbox=dict(
            type="SmoothL1Loss",
            beta=1 / 9.0,
            reduction="none",
            loss_weight=2.0,
        ),
        loss_dir=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            reduction="none",
            loss_weight=0.2,
        ),
    ),
    postprocess=dict(
        type="PointPillarsPostProcess",
        num_classes=len(class_names),
        box_coder=dict(
            type="GroundBox3dCoder",
            n_dim=7,
        ),
        use_direction_classifier=True,
        num_direction_bins=2,
        # test_cfg
        use_rotate_nms=False,
        nms_pre_max_size=1000,
        nms_post_max_size=300,
        nms_iou_threshold=0.5,
        score_threshold=0.3,
        max_per_img=100,
    ),
)

# model settings
deploy_model = dict(
    type="PointPillarsDetector",
    feature_map_shape=[320, 320],
    is_deploy=True,
    quant_begin_neck=True,
    pre_process=None,
    reader=None,
    backbone=None,
    neck=dict(
        type="SECONDNeck",
        in_feature_channel=16,
        down_layer_nums=[3, 5, 5],
        down_layer_strides=[2, 2, 2],
        down_layer_channels=[48, 96, 192],
        up_layer_strides=[1, 2, 4],
        up_layer_channels=[96, 96, 96],
        bn_kwargs=norm_cfg,
        use_relu6=use_relu6,
        quantize=True,
        quant_scale=None,
    ),
    head=dict(
        type="PointPillarsHead",
        num_classes=len(class_names),
        in_channels=sum([96, 96, 96]),
        use_direction_classifier=True,
    ),
)

deploy_inputs = dict(
    feature=torch.randn(1, 16, 320, 320),
)

db_sampler = dict(
    type="DataBaseSampler",
    enable=True,
    root_path="./tmp_data/kitti3d/",
    db_info_path="./tmp_data/kitti3d/kitti3d_dbinfos_train.pkl",
    sample_groups=[dict(Car=15)],
    db_prep_steps=[
        dict(
            type="DBFilterByDifficulty",
            filter_by_difficulty=[-1],
        ),
        dict(
            type="DBFilterByMinNumPoint",
            filter_by_min_num_points=dict(
                Car=5,
            ),
        ),
    ],
    global_random_rotation_range_per_object=[0, 0],
    rate=1.0,
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Kitti3D",
        data_path="./tmp_data/kitti3d/train_lmdb",
        transforms=[
            dict(
                type="ObjectSample",
                class_names=class_names,
                remove_points_after_sample=False,
                db_sampler=db_sampler,
            ),
            dict(
                type="ObjectNoise",
                gt_rotation_noise=[-0.15707963267, 0.15707963267],
                gt_loc_noise_std=[0.25, 0.25, 0.25],
                global_random_rot_range=[0, 0],
                num_try=100,
            ),
            dict(
                type="PointRandomFlip",
                probability=0.5,
            ),
            dict(
                type="PointGlobalRotation",
                rotation=[-0.78539816, 0.78539816],
            ),
            dict(
                type="PointGlobalScaling",
                min_scale=0.95,
                max_scale=1.05,
            ),
            dict(
                type="ShufflePoints",
                shuffle=True,
            ),
            dict(
                type="ObjectRangeFilter",
                point_cloud_range=pc_range,
            ),
            dict(type="LidarReformat"),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_lidar3d,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Kitti3D",
        data_path="./tmp_data/kitti3d/val_lmdb",
        transforms=[
            dict(type="LidarReformat"),
        ],
    ),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_lidar3d,
)
batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    loss_collector=collect_loss_by_index(0),
)
val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
    loss_collector=None,
)


def update_metric(metrics, batch, model_outs):
    preds = model_outs
    for metric in metrics:
        metric.update(preds, batch)


def update_loss(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=10000,
    epoch_log_freq=1,
    log_prefix=f"Validation_{task_name}",
)
loss_show_update = dict(
    type="MetricUpdater",
    metric_update_func=update_loss,
    step_log_freq=10,
    epoch_log_freq=1,
    log_prefix=f"loss_{task_name}",
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=10,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=f"{training_step}-",
    strict_match=True,
    mode="max",
    monitor_metric_key="mAP_3D_moderate",
)


val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=None,
    val_on_train_end=True,
    val_interval=1,
    log_interval=200,
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,
)


float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        betas=(0.95, 0.99),
        lr=2e-4,
        weight_decay=0.01,
    ),
    batch_processor=batch_processor,
    num_epochs=160,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(
            type="CyclicLrUpdater",
            target_ratio=(10, 1e-4),
            cyclic_times=1,
            step_ratio_up=0.4,
            step_log_interval=10,
        ),
        val_callback,
        ckpt_callback,
    ],
    sync_bn=True,
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=dict(
        type="Kitti3DMetricDet",
        compute_aos=True,
        current_classes=class_names,
        difficultys=[0, 1, 2],
    ),
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
calibration_step = 50

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
    val_metrics=dict(
        type="Kitti3DMetricDet",
        compute_aos=True,
        current_classes=class_names,
        difficultys=[0, 1, 2],
    ),
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
        params={"weight": dict(weight_decay=1e-4)},
        lr=1e-4,
        momentum=0.9,
    ),
    batch_processor=batch_processor,
    num_epochs=20,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(
            type="CyclicLrUpdater",
            target_ratio=(10, 1e-4),
            cyclic_times=1,
            step_ratio_up=0.4,
            step_log_interval=10,
        ),
        val_callback,
        ckpt_callback,
    ],
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=dict(
        type="Kitti3DMetricDet",
        compute_aos=True,
        current_classes=class_names,
        difficultys=[0, 1, 2],
    ),
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
                ignore_extra=True,
            ),
            dict(type="QAT2Quantize", convert_mode=convert_mode),
        ],
    ),
    data_loader=None,
    optimizer=None,
    batch_processor=None,
    num_epochs=0,
    device=None,
    callbacks=[ckpt_callback, trace_callback],
    val_metrics=dict(
        type="Kitti3DMetricDet",
        compute_aos=True,
        current_classes=class_names,
        difficultys=[0, 1, 2],
    ),
)


compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=task_name,
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["ddr"],
    opt="O3",
    output_layout="NHWC",
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
                ignore_extra=True,
                allow_miss=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=dict(
        type="Kitti3DMetricDet",
        compute_aos=True,
        current_classes=class_names,
        difficultys=[0, 1, 2],
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=100,
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
                ignore_extra=True,
                allow_miss=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=dict(
        type="Kitti3DMetricDet",
        compute_aos=True,
        current_classes=class_names,
        difficultys=[0, 1, 2],
    ),
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
                verbose=True,
                ignore_extra=True,
                allow_miss=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=dict(
        type="Kitti3DMetricDet",
        compute_aos=True,
        current_classes=class_names,
        difficultys=[0, 1, 2],
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=100,
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
    metrics=dict(
        type="Kitti3DMetricDet",
        compute_aos=True,
        current_classes=class_names,
        difficultys=[0, 1, 2],
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=100,
)
