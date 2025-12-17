# Copyright (c) Horizon Robotics. All rights reserved.

import copy
import os
from collections import OrderedDict

import numpy as np
import torch
from processed_dataset import HAT_UNITTEST_PREFIX

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.data.collates.traj_pred_collates import collate_vectornet
from hat.data.datasets.traj_pred_dataset import BaseTrajDataset
from hat.engine.processors.loss_collector import collect_loss_by_regex

task_name = "vectornet_hat_argo"
cudnn_benchmark = False
seed = None
log_rank_zero_only = True
march = "bayes1"
local_train = True
predictor_type = "mlp"

# 0.1. Paths.
if local_train:
    device_ids = [1]
    batch_size_per_gpu = 16
    # Set your local path here.
    J5FSD_BUCKET_PATH = "/horizon-bucket/J5FSD/"
    MATRIX_BUCKET_PATH = "/horizon-bucket/matrix/"
    ckpt_dir = (
        "/mnt/cephfs-adas-boschhwy/adas/shengzhe.dai/results/" + task_name
    )
    viz_dir = (
        "/mnt/cephfs-adas-boschhwy/adas/shengzhe.dai/results/"
        + task_name
        + "/viz"
    )
else:
    device_ids = [0, 1, 2, 3]
    batch_size_per_gpu = 8
    J5FSD_BUCKET_PATH = "/bucket/input/J5FSD/"
    MATRIX_BUCKET_PATH = "/bucket/input/matrix/"
    ckpt_dir = "/job_data/models/checkpoint"
    viz_dir = "/job_data/models/viz"
bucket_dir_dict = {"J5FSD": J5FSD_BUCKET_PATH, "matrix": MATRIX_BUCKET_PATH}

# Datasets
dataset_path = os.path.join(MATRIX_BUCKET_PATH, "users/xuewu.lin/argodataset/")

anchor_file = os.path.join(
    J5FSD_BUCKET_PATH,
    HAT_UNITTEST_PREFIX,
    "anchors/veh_anchor_add_ped_anchors.pkl",
)
anchor_num = 124

# 0.2. Parameters.
# -- Parameters about the training.
train_num_workers, val_num_workers = 8, 4
float_lr, quanti_lr = 0.001, 0.00001
float_epoch, quanti_epoch = 30, 5
weight_decay = 0.001
warmup_epoch = 1
log_freq = 100

# -- Parameters about the trajectory prediction task.
his_time_horizon = 2
fut_time_horizon = 3
pkl_fps = 10
seq_length = (his_time_horizon + fut_time_horizon) * pkl_fps
seq_period = 1
sample_step = 1
context_frames = his_time_horizon * pkl_fps
gt_traj_len = fut_time_horizon * pkl_fps
target_freq = 10
traj_len = int(gt_traj_len * target_freq / pkl_fps)

# -- Parameters about coordinates.
image_coordinates = "img"  # in ["bev", "img"]
img_h, img_w, img_resolution = 1024, 1024, 0.2
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

# -- Parameters and thresholds about obstacles.
# ---- Thresholds
ref_fps = 2
distance_thr = 50
num_frame_thr = 2
ped_shape_thr = 1
train_bounce_thr = np.pi / 8
val_bounce_thr = np.pi / 3
ped_drift_thr = 1.5 * ref_fps / pkl_fps
static_thr = np.array([1, 0.2, 0.2]) * ref_fps / pkl_fps
ego_track_id = -BaseTrajDataset.ANSWER
# ---- Params
veh_type_id = [1, 2, 3]
ped_cyc_type_id = []
leaving_mode = "all"
# why use type 2 here? The argoverse do not have perception yaw, therefore,
# we have calculated the yaw when generate a sample.
yaw_select_type = [2, 2, 2]
detect_peds_by_shape = True
use_state_vectors = True
use_instant_state_vectors = True
if_clip_state_vectors = False
max_obs_num = 32
# Usually, we need to enable the bouncing filter, but since the argoverse
# FPS is to big and the trajectories seem not smooth, we set False here.
if_train_filter_bounce = [True, True]
if_val_filter_bounce = [True, True]

# -- Parameters about polyline segment features.
local_ele_seg_thr = 100  # m
polyline_seg_len = 10
max_num_ele_seg = 256
all_elements = ["solid_lane", "virtuallanelines", "roadedge", "crosswalk"]
polyline_optional_feats = ["turn_dir", "pre_pre_point"]
traj_sample_ratio = int(target_freq / pkl_fps)
traj_polyline_len = traj_sample_ratio * (context_frames - 1)
# road feats: start_x, start_y, end_x, end_y, turn_dir, pre_pre_x, pre_pre_y
# then scale them to [-5, 5]
road_feat_dim = 7
road_feat_scale = [0.04, 0.04, 0.04, 0.04, 5, 0.04, 0.04]
# road feats: start_x, start_y, end_x, end_y, time_stamp, pid
# then scale them to [-5, 5]
traj_feat_dim = 6
traj_feat_scale = [0.04, 0.04, 0.04, 0.04, 3, 0.15]


# -- Parameters about models
subgraph_hidden_size = 128
globalgraph_hidden_size = 128
use_out_fc = True
num_attn_head = 4
backbone_fc_out_channels = 128
mlp_predictor_hidden_size = 128
if use_out_fc:
    mlp_input_channels = backbone_fc_out_channels
else:
    mlp_input_channels = globalgraph_hidden_size * num_attn_head


# -- Parameters about postprocessing and visualization.
use_nms = False
video_fps = pkl_fps * 5  # 5x video
viz_on_every_epoch = True

# 1. Setup data loader.
train_transforms = [
    dict(
        type="GenSeqCenter",
        anchor_frame="last_context",
        context_frames=context_frames,
        augmentation=False,
    ),
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
        type="PhyToImage",
        map_height=img_h,
        map_width=img_w,
        resolution=img_resolution,
        reverse=coords_reverse,
    ),
    dict(
        type="SelectYawArray",
        yaw_select_type=yaw_select_type,
        enable_none=False,
    ),
    dict(type="EgoCentricGt"),
]
val_transforms = copy.deepcopy(train_transforms)

train_transforms += [
    dict(
        type="FilterObstacles",
        is_training=True,
        is_multiagent=True,
        ego_track_id=ego_track_id,
        ped_cyc_type_id=ped_cyc_type_id,
        leaving_mode=leaving_mode,
        # range of prediction, veh[x_max, x_min, y_max, y_min],
        # ped[...], cycli[...]]
        valid_distance=[
            [50, -30, 50, -50],
            [30, 0, 10, -10],
            [50, -30, 20, -20],
        ],
        num_frame_thr=num_frame_thr,
        bounce_thr=train_bounce_thr,
        ped_drift_thr=ped_drift_thr,
        static_thr=static_thr,
        if_train_filter_bounce=if_train_filter_bounce,
        if_val_filter_bounce=if_val_filter_bounce,
        classify_by_shape=detect_peds_by_shape,
        ped_shape_thr=ped_shape_thr,
    ),
    dict(
        type="GenStatesAndMask",
        seq_length=seq_length,
        context_frames=context_frames,
        seq_period=seq_period,
        enable_incomplete_gts=True,
        is_training=True,
    ),
]
val_transforms += [
    dict(
        type="FilterObstacles",
        is_training=False,
        is_multiagent=True,
        ego_track_id=ego_track_id,
        ped_cyc_type_id=ped_cyc_type_id,
        leaving_mode=leaving_mode,
        # range of prediction, veh[x_max, x_min, y_max, y_min],
        # ped[...], cycli[...]]
        valid_distance=[
            [50, -30, 50, -50],
            [30, 0, 10, -10],
            [50, -30, 20, -20],
        ],
        num_frame_thr=num_frame_thr,
        bounce_thr=val_bounce_thr,
        ped_drift_thr=ped_drift_thr,  # i.e., 3 m/s
        static_thr=static_thr,
        if_train_filter_bounce=if_train_filter_bounce,
        if_val_filter_bounce=if_val_filter_bounce,
        classify_by_shape=detect_peds_by_shape,
        ped_shape_thr=ped_shape_thr,
    ),
    dict(
        type="GenStatesAndMask",
        seq_length=seq_length,
        context_frames=context_frames,
        seq_period=seq_period,
        enable_incomplete_gts=True,
        is_training=False,
    ),
]

common_transforms_2 = [
    dict(
        type="GetTrajPredObjectsInfo",
        scene_img_height=img_h,
        scene_img_width=img_w,
        ego_track_id=ego_track_id,
        ped_cyc_type_id=ped_cyc_type_id,
        use_state_vectors=use_state_vectors,
        use_instant_state_vectors=use_instant_state_vectors,
        if_clip_state_vectors=if_clip_state_vectors,
        reverse=coords_reverse,
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
    ),
    dict(
        type="ArgoverseStructredMapServer",
        map_origin_params=map_origin_params,
        element_keys=all_elements,
        curve_threshold=0.5,
        sample_mode="uniform",
        polyline_seg_len=polyline_seg_len,
        polyline_optional_feats=polyline_optional_feats,
        local_ele_seg_thr=local_ele_seg_thr,
        max_num_ele_seg=max_num_ele_seg,
        image_coordinates=image_coordinates,
        shuffle=True,
        scale=road_feat_scale,
    ),
]
train_transforms += common_transforms_2
val_transforms += common_transforms_2

val_transforms.append(
    dict(
        type="ArgoverseStructredVizHelper",
        map_origin_params=map_origin_params,
        image_coordinates=image_coordinates,
        ego_track_id=ego_track_id,
    )
)

train_dataloader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ArgoverseTdtDataset",
        data_path=dataset_path,
        mode="train",
        transforms=train_transforms,
        center="AV",
        meters_map=max(phy_img_h, phy_img_w),
        resolution=img_resolution,
        map_mode="vectorized",
        polyline_seg_len=polyline_seg_len,
        for_viz=False,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    collate_fn=collate_vectornet,
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=train_num_workers,
)

val_dataloader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ArgoverseTdtDataset",
        data_path=dataset_path,
        mode="val",
        transforms=val_transforms,
        center="AV",
        meters_map=max(phy_img_h, phy_img_w),
        resolution=img_resolution,
        map_mode="vectorized",
        for_viz=True,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    collate_fn=collate_vectornet,
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=val_num_workers,
)

enable_scale_trils = True
# 2. Setup Model.
if predictor_type == "mlp":
    model_heads = OrderedDict(
        normal_head=dict(
            type="BasicMlpDecoder",
            in_channels=mlp_input_channels,
            traj_len=traj_len,
            hidden_size=mlp_predictor_hidden_size,
            num_hidden_layers=3,
            enable_scale_trils=enable_scale_trils,
            log_std_clamp_min=-2,
            log_std_clamp_max=2,
        )
    )
    enable_itp_gt = True
    ckpt_use_head_name = "normal_head"
else:
    raise ValueError(f"Undefined prediction decoder {predictor_type}")
head_name_dict = dict(
    traj_head="normal_head",
    track_valid_head="normal_head",
)

model = dict(
    type="BasicVectorNet",
    backbone=dict(
        type="VectorNetBackbone",
        road_feat_channels=road_feat_dim,
        road_polyline_len=polyline_seg_len - 1,
        road_polyline_num=max_num_ele_seg,
        traj_feat_channels=traj_feat_dim,
        traj_polyline_len=traj_polyline_len,
        traj_polyline_num=max_obs_num,
        num_hidden_units=subgraph_hidden_size,
        num_attn_hidden_units=globalgraph_hidden_size,
        num_sub_graph_layers=3,
        num_attention_heads=num_attn_head,
        use_out_fc=use_out_fc,
        fc_out_channels=mlp_input_channels,
    ),
    heads=model_heads,
    post_process=dict(
        type="VecterNetPostProcessor",
        sample_traj=False,
    ),
    losses=dict(
        type="BasicMultipathLoss",
        head_weight=1,
        reg_loss_scale=1,
    ),
    itp_ratio=traj_sample_ratio,
    enable_itp_gt=enable_itp_gt,
    is_int_infer_model=False,
)

# 3. Setup metrics.
metric_updater = dict(
    type="MetricUpdater",
    metrics=[
        dict(type="LossShow", name="reg_loss"),
    ],
    metric_update_func=update_metric_using_regex(
        per_metric_patterns=[  # corresponding to metrics
            dict(label_pattern=None, pred_pattern="^.*reg_loss$"),
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
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix="Validation_" + task_name,
)

# 4. Setup callbacks.
if predictor_type == "mlp":
    loss_str = "^.*reg_loss.*"
    metrics = [
        dict(
            type="TrajPredMetric",
            head_names=head_name_dict,
            k_values=[1],
            name=ckpt_use_head_name,
        )
    ]
    num_traj_to_viz = 1
elif predictor_type == "multipath":
    loss_str = "^.*loss.*"
    metrics = [
        dict(
            type="TrajPredMetric",
            head_names=head_name_dict,
            k_values=[1, 5, 10, 64],
            name=ckpt_use_head_name,
        )
    ]
    metric_updater["metrics"].append(
        dict(type="LossShow", name="cls_loss"),
    )
    metric_updater["metric_update_func"] = update_metric_using_regex(
        per_metric_patterns=[  # corresponding to metrics
            dict(label_pattern=None, pred_pattern="^.*cls_loss$"),
            dict(label_pattern=None, pred_pattern="^.*reg_loss$"),
        ]
    )
    num_traj_to_viz = 5
else:
    raise ValueError(f"Undefined prediction decoder {predictor_type}")

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

# 5. Trainer and solver.
# -- 1. float.
float_viz_callback = dict(
    type="TrajPredViz",
    head_names=head_name_dict,
    height=img_h,
    width=img_w,
    resolution=img_resolution,
    map_origin=[phy_img_h, phy_img_w],
    video_save_path=viz_dir,
    video_name="float_best_viz",
    video_fps=video_fps,
    viz_on_every_epoch=viz_on_every_epoch,
    max_epoch=float_epoch,
    ego_track_id=-BaseTrajDataset.ANSWER,
    num_traj_to_viz=num_traj_to_viz,
    num_workers=val_num_workers,
)
float_val_callback = copy.deepcopy(val_callback)
# float_val_callback["callbacks"].append(float_viz_callback)

float_model = copy.deepcopy(model)
float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=float_model,
    data_loader=train_dataloader,
    optimizer=dict(
        type=torch.optim.Adam,
        lr=float_lr,
        weight_decay=weight_decay,
    ),
    batch_processor=batch_processor,
    num_epochs=float_epoch,
    callbacks=[
        metric_updater,
        lr_update_callback,
        stats_callback,
        float_val_callback,
        dict(
            type="Checkpoint",
            save_dir=ckpt_dir,
            name_prefix="float-",
            strict_match=True,
            mode="min",
            monitor_metric_key=f"head_{ckpt_use_head_name}_min_ade_1",
        ),
    ],
    sync_bn=True,
    train_metrics=metrics,
    val_metrics=metrics,
)

float_solver = dict(
    trainer=float_trainer,
    allow_miss=True,
    ignore_extra=True,
    quantize=False,
)

# -- 2. qat.
qat_val_callback = copy.deepcopy(val_callback)
qat_viz_callback = copy.deepcopy(float_viz_callback)
qat_viz_callback["video_name"] = "qat_best_viz"
qat_viz_callback["max_epoch"] = quanti_epoch
qat_val_callback["callbacks"] = [val_metric_updater]  # , qat_viz_callback]

qat_model = copy.deepcopy(model)
qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=qat_model,
    data_loader=train_dataloader,
    optimizer=dict(
        type=torch.optim.Adam,
        lr=quanti_lr,
        weight_decay=weight_decay,
    ),
    batch_processor=batch_processor,
    num_epochs=quanti_epoch,
    device=None,
    callbacks=[
        lr_update_callback,
        stats_callback,
        qat_val_callback,
        dict(
            type="Checkpoint",
            save_dir=ckpt_dir,
            name_prefix="qat-",
            strict_match=True,
            mode="min",
            monitor_metric_key=f"head_{ckpt_use_head_name}_min_ade_1",
        ),
    ],
    train_metrics=metrics,
    val_metrics=metrics,
)

qat_solver = dict(
    trainer=qat_trainer,
    quantize=True,
    check_quantize_model=True,
    pre_step="float",
    pre_step_checkpoint=os.path.join(
        ckpt_dir, "float-checkpoint-best.pth.tar"
    ),
    strict_match=True,
)

step2solver = dict(float=float_solver, qat=qat_solver)
