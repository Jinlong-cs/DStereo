# Copyright (c) Horizon Robotics. All rights reserved.

import copy
import getpass
import json
import os
from collections import OrderedDict

import numpy as np
import torch
from behav_sample_tags import BehavTag
from horizon_plugin_pytorch.quantization import March
from processed_dataset import SD_DEMO_DATASET

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.data.datasets.traj_pred_dataset import BaseTrajDataset
from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
cudnn_benchmark = False
seed = None
log_rank_zero_only = True
march = March.BAYES  # March.BAYES for bup2.5 or "bayes1" for bpu2.2

# Task configurations
task_name = "behav_release_v10.0.0"
qat_ckpt_for_int_infer = "qat-checkpoint-best.pth.tar"
float_ckpt_for_int_infer = "float-checkpoint-best.pth.tar"

# Dataset configurations
dataset_name = "multi-task_perrec"
behav_train_dataset = "behav_full_train"
behav_val_dataset = "behav_full_val"

# Very import configurations!!!
local_train = True
if "bucket" in os.listdir("/"):
    local_train = False
use_state_vectors = True  # if add state_vector in the input of model
enable_high_freq = False
use_behav_state_vectors = True
num_behav_state_vectors = 14
use_online_transforms = True
is_visualize = True
shuffle_for_road_feats = True  # set True in training
freeze_backbone = False
is_training = True

# swith to high freq dataset if enable_high_freq
if enable_high_freq:
    behav_train_dataset = "perrec_high_train_behav"
    behav_val_dataset = "perrec_high_val_behav"

# Range of prediction: [x_max, x_min, y_max, y_min]
veh_range = [50.0, -30.0, 50.0, -50.0]
ped_range = [30.0, 0.0, 10.0, -10.0]
cyc_range = [50.0, -30.0, 20.0, -20.0]

tag_masks = {
    "KeepLaneWeaving": [
        [
            BehavTag.Behavior_KeepLane.value,
            BehavTag.RelationToCenterLine_Weaving.value,
        ],
    ]
}

# 0.1. Paths.
if local_train:
    device_ids = [0, 1, 2, 3]
    # Set your local path here.
    user_name = getpass.getuser()
    J5FSD_BUCKET_PATH = "/horizon-bucket/J5FSD/"
    SD_Algo_BUCKET_PATH = "/horizon-bucket/SD_Algorithm/"
    ckpt_dir = f"/jfs-public/users/{user_name}/results/{task_name}"
    viz_dir = f"/jfs-public/users/{user_name}/results/{task_name}/viz"
    tensorboard_dir = (
        f"/home/users/{user_name}/results/{task_name}/tensorboard"
    )
else:
    device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    J5FSD_BUCKET_PATH = "/bucket/input/J5FSD/"
    SD_Algo_BUCKET_PATH = "/bucket/input/SD_Algorithm/"
    ckpt_dir = "/job_data/models/checkpoint"
    viz_dir = "/job_data/models/viz"
    tensorboard_dir = os.getenv("TENSORBOARD_LOG_PATH") or os.path.join(
        "/job_data/models/tensorboard", ".aidi"
    )
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(viz_dir, exist_ok=True)
os.makedirs(tensorboard_dir, exist_ok=True)
bucket_dir_dict = {
    "SD_Algorithm": SD_Algo_BUCKET_PATH,
}

# Datasets.
dataset = SD_DEMO_DATASET["multi-task_perrec"]
bucket_dir = bucket_dir_dict[dataset["bucket"]]
behav_train_pkl = os.path.join(bucket_dir, dataset[behav_train_dataset])
behav_val_pkl = os.path.join(bucket_dir, dataset[behav_val_dataset])
behav_down_ratio = 0.1
# 0.2. Parameters.
# -- Parameters about the training.
float_epoch, quanti_epoch = 500, 50
train_num_workers, val_num_workers = 16, 16
batch_size_per_gpu = 64
float_lr, quanti_lr = 0.001, 0.0005  # default 0.001, 0.00001
weight_decay, quanti_weight_decay = 0.001, 0.001  # default 0.001, 0.0001
warmup_epoch = 1
log_freq = 100
val_interval = 5

# -- Parameters about the trajectory prediction task.
pkl_fps = 2
if enable_high_freq:
    pkl_fps = 8
seq_length = 8 * pkl_fps
seq_period = int(pkl_fps / pkl_fps)
sample_step = pkl_fps / 2
context_frames = 2 * pkl_fps
target_freq = 10

if enable_high_freq:
    num_behav_state_vectors = 62


# -- Parameters about coordinates.
image_coordinates = "bev"  # in ["bev", "img"]
img_h, img_w = 512, 512
bev_origin_x, bev_origin_y, img_resolution = 72.4, 51.2, 0.2
phy_img_h, phy_img_w = img_h * img_resolution, img_w * img_resolution
if image_coordinates == "bev":
    map_origin_params = [bev_origin_x, bev_origin_y, img_resolution]
    coords_reverse = True
elif image_coordinates == "img":
    map_origin_params = [phy_img_h, phy_img_w, img_resolution]
    coords_reverse = False
else:
    raise ValueError(f"Undefined image coordinates {image_coordinates}")

# -- Parameters about obstacles.
ego_track_id = -42
train_bounce_thr = np.pi / 8
val_bounce_thr = np.pi / 3
speed_drift_thr = [
    35.0,  # veh_drift_thr=35 means 35*2=70m/s
    3.25,  # ped_drift_thr=3.25 means 3.25*2=6.5m/s
    20.0,  # cyc_drift_thr=20.0 means 20.0*2=40m/s
]
veh_lateral_drift_thr = 1.5  # if use set: 1.5
veh_type_id = 1
ped_cyc_type_id = [2, 18]
leaving_mode = "all"
yaw_select_type = [2, 0, 2]  # veh,ped,cyc
static_thr = [0.1, 0.2, 0.2]
ped_shape_thr = 1
detect_peds_by_shape = False
if use_state_vectors:
    num_state_vectors = 3
use_instant_state_vectors = True
if_clip_state_vectors = True
max_obs_num = 32

# -- Parameters about road polyline segment features.
local_ele_seg_thr = 100
polyline_seg_len = 10
max_num_ele_seg = 128
all_supported_elements = [
    "stopline",
    "crosswalk",
    "solid_lane",
    "roadedge",
    "virtuallanelines",
]
sample_mode_of_element = {
    "stopline": "uniform",
    "crosswalk": "uniform",
    "solid_lane": "uniform",
    "roadedge": "uniform",
    "virtuallanelines": "uniform",
}
# available SAMPLE_MODE for polyline: uniform/original
# available SAMPLE_MODE for polygon: uniform
all_elements = [
    "stopline",
    "crosswalk",
    "solid_lane",
    "roadedge",
]
# all polyline_optional_feats = ["turn_dir", "pre_pre_point", "element_type"]
# sacle of feats to [-5,5]: "default":0.04, "turn_dir":5,
# "pre_pre_point":0.04, "element_type":5
polyline_optional_feats = ["turn_dir", "pre_pre_point", "element_type"]
# road feats: start_x, start_y, end_x, end_y, one-hot
road_feat_dim = 4
road_feat_scale = [0.04, 0.04, 0.04, 0.04]
if len(polyline_optional_feats) > 0:
    for feats in polyline_optional_feats:
        if feats == "turn_dir":
            road_feat_dim += 1
            road_feat_scale += [5]
        if feats == "pre_pre_point":
            road_feat_dim += 2
            road_feat_scale += [0.04, 0.04]
        if feats == "element_type":
            road_feat_dim += len(all_supported_elements)
            road_feat_scale += [5 for i in range(len(all_supported_elements))]
road_feats_front_range = 120
road_feats_back_range = -75
road_feat_quant_scale = (
    (road_feats_front_range - road_feats_back_range)
    * road_feat_scale[0]
    / 32768
)
# -- Parameters about traj polyline segment features.
# all traj_optional_feats = ["timestamp", "pid", "obstacle_class"]
# scale of feats to [-5, 5]: "timestamp":3, "pid":0.15, "obstacle_class":5
traj_optional_feats = ["timestamp", "pid", "obstacle_class"]
# traj feats: start_x, start_y, end_x, end_y, time_stamp, pid, one-hot for type
num_obs_type = 3
traj_feat_dim = 4
traj_feat_scale = [0.04, 0.04, 0.04, 0.04]
if len(traj_optional_feats) > 0:
    for feats in traj_optional_feats:
        if feats == "timestamp":
            traj_feat_dim += 1
            traj_feat_scale += [3]
        if feats == "pid":
            traj_feat_dim += 1
            traj_feat_scale += [0.15]
        if feats == "obstacle_class":
            traj_feat_dim += num_obs_type
            traj_feat_scale += [5 for _ in range(num_obs_type)]
traj_feats_front_range = 150
traj_feats_back_range = -100
traj_feat_quant_scale = (
    (traj_feats_front_range - traj_feats_back_range)
    * traj_feat_scale[0]
    / 32768
)
traj_sample_ratio = int(target_freq / pkl_fps)
if_itp_traj = True
if if_itp_traj:
    traj_polyline_len = traj_sample_ratio * (context_frames - 1)
else:
    traj_polyline_len = 3

# -- Parameters about models
subgraph_hidden_size = 64  # 64 for baseline
globalgraph_hidden_size = 64  # 64 for baseline
num_sub_graph_layers = 3  # 3 for baseline
use_out_fc = True
num_attn_head = 4
backbone_fc_out_channels = 128
mlp_predictor_hidden_size = 128
if use_out_fc:
    mlp_input_channels = backbone_fc_out_channels
else:
    mlp_input_channels = globalgraph_hidden_size * num_attn_head

# 1. Setup data loader.
offline_transforms = [
    dict(
        type="GenSeqCenter",
        anchor_frame="last_context",
        context_frames=context_frames,
        augmentation=False,
    ),
    dict(type="GetLcfTimeStamp"),
    dict(type="GenFutureTrackids"),
    dict(
        type="RemapObsCls",
        veh_type_id=veh_type_id,
        ped_cyc_type_id=ped_cyc_type_id,
        detect_peds_by_shape=detect_peds_by_shape,
        ped_shape_thr=ped_shape_thr,
    ),
    dict(
        type="GenBoundingBox",
        gen_ego_bbox=False,
        clockwise=True,
        min_shape=1,
        const_z_value=1.5,
    ),
    dict(
        type="GenBoundingBox",
        gen_ego_bbox=True,
        clockwise=True,
        min_shape=1,
        const_z_value=1.5,
    ),
    dict(
        type="PhyToBEV",
        bev_origin_x=bev_origin_x,
        bev_origin_y=bev_origin_y,
        resolution=img_resolution,
        reverse=coords_reverse,
    ),
    dict(
        type="GetSeqDataFrameMask",
        freq_ratio=1,
    ),
    dict(
        type="SelectYawArray",
        yaw_select_type=yaw_select_type,
        enable_none=False,
    ),
    dict(type="EgoCentricGt"),
    dict(
        type="FilterObstacles",
        is_training=is_training,
        is_multiagent=True,
        ego_track_id=-BaseTrajDataset.ANSWER,
        ped_cyc_type_id=ped_cyc_type_id,
        leaving_mode=leaving_mode,
        valid_distance=[
            veh_range,
            ped_range,
            cyc_range,
        ],
        num_frame_thr=2,
        bounce_thr=train_bounce_thr,
        speed_drift_thr=speed_drift_thr,
        veh_lateral_drift_thr=veh_lateral_drift_thr,
        static_thr=static_thr,
        if_train_filter_bounce=[True, True],
        if_val_filter_bounce=[True, True],
        classify_by_shape=detect_peds_by_shape,
        ped_shape_thr=ped_shape_thr,
    ),
    dict(
        type="GenStatesAndMask",
        enable_incomplete_gts=True,
        is_training=is_training,
    ),
    dict(
        type="GetTrajPredObjectsInfo",
        scene_img_height=img_h,
        scene_img_width=img_w,
        ego_track_id=-BaseTrajDataset.ANSWER,
        ped_cyc_type_id=ped_cyc_type_id,
        use_state_vectors=use_state_vectors,
        use_instant_state_vectors=use_instant_state_vectors,
        if_clip_state_vectors=if_clip_state_vectors,
        reverse=coords_reverse,
    ),
    dict(
        type="VectorNetStructuredMapServer",
        map_origin_params=map_origin_params,
        element_keys=all_elements,
        sample_mode=sample_mode_of_element,
        curve_threshold=1.08,
        polyline_seg_len=polyline_seg_len,
        polyline_optional_feats=polyline_optional_feats,
        local_ele_seg_thr=local_ele_seg_thr,
        max_num_ele_seg=max_num_ele_seg,
        image_coordinates=image_coordinates,
        shuffle=True if is_training else False,
        scale=road_feat_scale,
        valid_img_coords_key="valid_img_coords",
    ),
    dict(
        type="VectorNetTrajExtractor",
        map_origin_params=map_origin_params,
        source_freq=pkl_fps,
        target_freq=target_freq,
        ego_track_id=ego_track_id,
        max_obs_num=max_obs_num,
        local_ele_seg_thr=local_ele_seg_thr,
        image_coordinates=image_coordinates,
        reverse=coords_reverse,
        scale=traj_feat_scale,
        traj_feat_dim=traj_feat_dim,
        num_obs_type=num_obs_type,
        traj_optional_feats=traj_optional_feats,
        valid_track_ids_key="valid_track_ids",
        if_itp_traj=if_itp_traj,
    ),
    dict(
        type="BehaviorSampling",
        is_training=False,
        ego_track_id=-BaseTrajDataset.ANSWER,
        valid_track_ids_key="valid_track_ids",
    ),
    dict(
        type="GetBehavPredObjectsInfo",
        ego_track_id=-BaseTrajDataset.ANSWER,
        use_behav_state_vectors=use_behav_state_vectors,
        num_behav_state_vectors=num_behav_state_vectors,
        use_obs_interaction=False,
        use_self_boxes=False,
        use_self_states=True,
        if_norm=False,
        norm_scale=5,
        mode="scene",
    ),
]
online_transforms = [
    dict(
        type="GetBehavPredObjectsInfo",
        ego_track_id=-BaseTrajDataset.ANSWER,
        use_behav_state_vectors=use_behav_state_vectors,
        num_behav_state_vectors=num_behav_state_vectors,
        use_obs_interaction=False,
        use_self_boxes=False,
        use_self_states=True,
        if_norm=False,
        norm_scale=5,
        mode="obstacle",
    ),
]
if use_online_transforms:
    transforms = online_transforms
else:
    transforms = []

train_dataloader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="TrajPredBehavDataset",
        dataset_pkl=behav_train_pkl,
        road_scale=road_feat_scale,
        traj_scale=traj_feat_scale,
        stage="train",
        down_ratio=behav_down_ratio,
        transforms=transforms,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=train_num_workers,
)

val_dataloader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="TrajPredBehavDataset",
        dataset_pkl=behav_val_pkl,
        road_scale=road_feat_scale,
        traj_scale=traj_feat_scale,
        stage="val",
        transforms=transforms,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=val_num_workers,
)


in_channels = dict(
    graph_feat=mlp_input_channels,
)
if use_state_vectors:
    in_channels["behav_state_vector_feat"] = num_state_vectors
    if use_behav_state_vectors:
        in_channels["behav_state_vector_feat"] += num_behav_state_vectors

model_heads = OrderedDict(
    behav_head=dict(
        type="BasicBehavDecoder",
        in_channels=in_channels,
        n_hidden_layers=[256, 128],
        is_int_infer_model=False,
        loss=dict(
            type="BehavHeadLoss",
            lat_head_weight=1,
            lon_head_weight=0,
            lat_class_weight=[1.0, 2.0, 2.0],
            lon_class_weight=[1.0, 1.0, 1.0],
        ),
        post_process=dict(
            type="BehavHeadPostProcessor",
            front_range=50,
            lane_width=3.75,
        ),
    )
)

necks = OrderedDict(
    traj_neck=dict(
        type="VectorNetNeck",
        is_int_infer_model=False,
        use_state_vectors=use_state_vectors,
    ),
)

model = dict(
    type="VectorNetV2",
    backbone=dict(
        type="VectorNetBackbone",
        road_feat_channels=road_feat_dim,
        road_polyline_len=polyline_seg_len - 1,
        road_polyline_num=max_num_ele_seg,
        traj_feat_channels=traj_feat_dim,
        traj_polyline_len=traj_polyline_len,
        traj_polyline_num=max_obs_num,
        road_feat_quant_scale=road_feat_quant_scale,
        traj_feqt_quant_scale=traj_feat_quant_scale,
        num_hidden_units=subgraph_hidden_size,
        num_attn_hidden_units=globalgraph_hidden_size,
        num_sub_graph_layers=num_sub_graph_layers,
        num_attention_heads=num_attn_head,
        use_out_fc=use_out_fc,
        fc_out_channels=mlp_input_channels,
    ),
    necks=necks,
    heads=model_heads,
    enable_itp_gt=False,
    is_int_infer_model=False,
)

# 3. Setup metrics.
metric_updater = dict(
    type="MetricUpdater",
    metrics=[
        dict(type="LossShow", name="lat_behav_pred_loss"),
        dict(type="LossShow", name="lon_behav_pred_loss"),
    ],
    metric_update_func=update_metric_using_regex(
        per_metric_patterns=[  # corresponding to metrics
            dict(label_pattern=None, pred_pattern="^.*lat_behav_pred_loss$"),
            dict(label_pattern=None, pred_pattern="^.*lon_behav_pred_loss$"),
        ]
    ),
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
)


def update_metric(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


val_metric_updater = dict(
    type="BehavPredMetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=2000,
    epoch_log_freq=1,
    log_prefix="Validation_" + task_name,
    save_metric_path=ckpt_dir,
)

# 4. Setup callbacks.

loss_str = "^.*loss.*"

metrics = [
    dict(
        type="BehavPredMetric",
        name="behav_head",
        F_score_beta=2,
        F_score_weights=[0.2, 0.4, 0.4],
        save_dir=viz_dir,
        tag_masks=tag_masks,
    ),
]

per_metric_patterns = [
    dict(label_pattern=None, pred_pattern="^.*lat_behav_pred_loss$"),
    dict(label_pattern=None, pred_pattern="^.*lon_behav_pred_loss$"),
]

batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    loss_collector=collect_loss_by_regex(loss_str),
)
val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
)

val_callback = dict(
    type="Validation",
    data_loader=val_dataloader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_interval=val_interval,
    interval_by="epoch",
    val_model=None,
    val_on_train_end=True,
)

stats_callback = dict(
    type="StatsMonitor",
    log_freq=log_freq,
)

lr_update_callback = dict(
    type="CosLrUpdater",
    warmup_by="epoch",
    warmup_len=warmup_epoch,
    step_log_interval=log_freq,
)

# tensorboard
# job on aidi platform will have TENSORBOARD_LOG_PATH env,
# where the saved tb can be shown through aidi web UI
tb_lat_behav_loss_callback = dict(
    type="TensorBoard",
    save_dir=os.path.join(tensorboard_dir, "float", "loss"),
    loss_name_reg="^.*lat_behav_pred_loss$",
    update_freq=100,
    update_by="step",
)

tb_lon_behav_loss_callback = dict(
    type="TensorBoard",
    save_dir=os.path.join(tensorboard_dir, "float", "loss"),
    loss_name_reg="^.*lon_behav_pred_loss$",
    update_freq=100,
    update_by="step",
)

freeze_backbone_callback = dict(
    type="FreezeModule",
    modules=[["backbone"]],
    step_or_epoch=[0],
    update_by="step",
)
# 5. Trainer and solver.
# -- 1. float.

float_val_callback = copy.deepcopy(val_callback)

float_viz_callback = dict(
    type="BehavPredViz",
    save_path=viz_dir,
    dump_file_name="all_sample_info.pkl",
    vis_track_ids=None,
    is_visualize=is_visualize,
    with_gt=True,
)

train_callback = [
    metric_updater,
    lr_update_callback,
    stats_callback,
    float_val_callback,
    dict(
        type="Checkpoint",
        save_dir=ckpt_dir,
        name_prefix="float-",
        save_interval=val_interval,
        interval_by="epoch",
        strict_match=True,
        mode="max",
        monitor_metric_key="behav_head_global_latbehav_F-score",
    ),
    tb_lat_behav_loss_callback,
    tb_lon_behav_loss_callback,
]

if freeze_backbone:
    train_callback.append(freeze_backbone_callback)
float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=copy.deepcopy(model),
    # model_convert_pipeline=dict(
    #     type="ModelConvertPipeline",
    #     converters=[
    #         dict(
    #             type="LoadCheckpoint",
    #             checkpoint_path=os.path.join(
    #                 ckpt_dir, "float-checkpoint-best.pth.tar"
    #             ),
    #             ignore_extra=True,
    #         ),
    #     ],
    # ),
    data_loader=train_dataloader,
    optimizer=dict(
        type=torch.optim.Adam,
        lr=float_lr,
        weight_decay=weight_decay,
    ),
    batch_processor=batch_processor,
    num_epochs=float_epoch,
    callbacks=train_callback,
    sync_bn=True,
    train_metrics=metrics,
    val_metrics=metrics,
)

float_predictor = dict(
    type="Predictor",
    model=copy.deepcopy(model),
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=[val_dataloader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=metrics,
    callbacks=[
        val_metric_updater,
    ],
    log_interval=100,
)

# -- 2. qat.
qat_val_callback = copy.deepcopy(val_callback)

qat_viz_callback = copy.deepcopy(float_viz_callback)

qat_tb_save_dir = os.path.join(tensorboard_dir, "qat", "loss")
qat_tb_lat_behav_loss_callback = copy.deepcopy(tb_lat_behav_loss_callback)
qat_tb_lat_behav_loss_callback["save_dir"] = qat_tb_save_dir
qat_tb_lon_behav_loss_callback = copy.deepcopy(tb_lon_behav_loss_callback)
qat_tb_lon_behav_loss_callback["save_dir"] = qat_tb_save_dir


qat_tb_val_save_dir = os.path.join(tensorboard_dir, "qat", "train")

qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=copy.deepcopy(model),
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        qconfig_params=dict(
            activation_qat_qkwargs=dict(
                averaging_constant=0.01,
            ),
            weight_qat_qkwargs=dict(
                averaging_constant=0.01,
            ),
        ),
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
                ignore_extra=False,
            ),
            dict(type="Float2QAT"),
        ],
    ),
    data_loader=train_dataloader,
    optimizer=dict(
        type=torch.optim.Adam,
        lr=quanti_lr,
        weight_decay=quanti_weight_decay,
    ),
    batch_processor=batch_processor,
    num_epochs=quanti_epoch,
    device=None,
    callbacks=[
        metric_updater,
        lr_update_callback,
        stats_callback,
        qat_val_callback,
        dict(
            type="Checkpoint",
            save_dir=ckpt_dir,
            name_prefix="qat-",
            save_interval=val_interval,
            strict_match=True,
            mode="max",
            monitor_metric_key="behav_head_global_latbehav_F-score",
        ),
        qat_tb_lat_behav_loss_callback,
        qat_tb_lon_behav_loss_callback,
    ],
    train_metrics=metrics,
    val_metrics=metrics,
)

qat_predictor = dict(
    type="Predictor",
    model=copy.deepcopy(model),
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
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
    data_loader=[val_dataloader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=metrics,
    callbacks=[
        val_metric_updater,
        qat_viz_callback,
    ],
    log_interval=100,
)

# -- 3. int.
# ---- add description
params_desc = dict(
    context_frames=context_frames,
    num_frame_thr=2,
    veh_select_yaw_type=yaw_select_type[0],
    ped_select_yaw_type=yaw_select_type[1],
    cyc_select_yaw_type=yaw_select_type[2],
    use_state_vectors=1 if use_state_vectors else 0,
    bounce_thr=val_bounce_thr,
    norm_scale=1.0,
    veh_static_threshold=static_thr[0],
    ped_static_threshold=static_thr[1],
    cyc_static_threshold=static_thr[2],
    veh_lateral_drift_thr=veh_lateral_drift_thr,
    veh_drift_thr=speed_drift_thr[0],
    ped_drift_thr=speed_drift_thr[1],
    cyc_drift_thr=speed_drift_thr[2],
    veh_range_xmax=veh_range[0],
    veh_range_xmin=veh_range[1],
    veh_range_ymax=veh_range[2],
    veh_range_ymin=veh_range[3],
    ped_range_xmax=ped_range[0],
    ped_range_xmin=ped_range[1],
    ped_range_ymax=ped_range[2],
    ped_range_ymin=ped_range[3],
    cyc_range_xmax=cyc_range[0],
    cyc_range_xmin=cyc_range[1],
    cyc_range_ymax=cyc_range[2],
    cyc_range_ymin=cyc_range[3],
)

add_desc_pp = dict(
    type="AddDesc",
    per_tensor_desc=[],
)

add_desc_pp["per_tensor_desc"].append(
    json.dumps(
        dict(
            task=f"{task_name}_head_behav_pred_lat_prob",
            **params_desc,
        )
    )
)
add_desc_pp["per_tensor_desc"].append(
    json.dumps(
        dict(
            task=f"{task_name}_head_behav_pred_lon_prob",
            **params_desc,
        )
    )
)


deploy_model = copy.deepcopy(model)
deploy_model["post_process"] = add_desc_pp
deploy_model["losses"] = None
deploy_model["is_int_infer_model"] = True
for _, neck in deploy_model["necks"].items():
    neck["is_int_infer_model"] = True
for _, head in deploy_model["heads"].items():
    head["is_int_infer_model"] = True
    head["loss"] = None
    head["post_process"] = None

int_val_callback = copy.deepcopy(val_callback)
int_val_callback["callbacks"] = []
int_val_callback["data_loader"] = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="TrajPredBehavDataset",
        dataset_pkl=behav_val_pkl,
        road_scale=road_feat_scale,
        traj_scale=traj_feat_scale,
        stage="val",
    ),
    batch_size=1,
    shuffle=False,
    num_workers=val_num_workers,
)

hdm_max_num_obs = 16
num_all_poly = max_num_ele_seg + max_obs_num
# permute(0, 3, 1, 2) here
deploy_inputs = dict(
    struct_road_feats=torch.randn(
        (hdm_max_num_obs, road_feat_dim, max_num_ele_seg, polyline_seg_len - 1)
    ),
    struct_traj_feats=torch.randn(
        (hdm_max_num_obs, traj_feat_dim, max_obs_num, traj_polyline_len)
    ),
    attention_mask=torch.ones(
        (hdm_max_num_obs, 1, num_all_poly, num_all_poly)
    ),
)
if use_state_vectors:
    deploy_inputs["behav_state_vectors"] = torch.randn(
        (hdm_max_num_obs, num_state_vectors + num_behav_state_vectors, 1, 1)
    )

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix="int-",
    strict_match=True,
    mode="max",
    monitor_metric_key="behav_head_global_latbehav_F-score",
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,
)
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
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
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
    callbacks=[
        ckpt_callback,
        trace_callback,
    ],
    val_metrics=metrics,
)

int_infer_predictor = dict(
    type="Predictor",
    model=deploy_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
            ),
            dict(type="QAT2Quantize"),
        ],
    ),
    data_loader=[val_dataloader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=metrics,
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)
