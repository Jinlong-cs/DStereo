# Copyright (c) Horizon Robotics. All rights reserved.

import os

import numpy as np
import torch
from configs.traj_pred.color_table import gen_roadmap_with_vl_color_table
from configs.traj_pred.processed_dataset import SD_DEMO_DATASET

from hat.data.collates.traj_pred_collates import collate_uniformpath_viz
from hat.data.datasets.traj_pred_dataset import BaseTrajDataset

task_name = "uniformpath"
cudnn_benchmark = False
seed = None
log_rank_zero_only = True
local_train = True

# 0.1. Paths.
if local_train:
    device_ids = [3]
    # Set your local path here.
    J5FSD_BUCKET_PATH = "/horizon-bucket/J5FSD/"
    J5FSD_2_BUCKET_PATH = "/horizon-bucket/J5FSD_2/"
    ckpt_dir = (
        "/mnt/cephfs-adas-boschhwy/adas/jiaqi01.chen/models/" + task_name
    )
    viz_dir = (
        "/mnt/cephfs-adas-boschhwy/adas/jiaqi01.chen/models/"
        + task_name
        + "/viz"
    )
else:
    device_ids = [0, 1, 2, 3]
    J5FSD_BUCKET_PATH = "/bucket/input/J5FSD/"
    J5FSD_2_BUCKET_PATH = "/bucket/input/J5FSD_2/"
    ckpt_dir = "/job_data/models/checkpoint"
    viz_dir = "/job_data/models/viz"

os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(viz_dir, exist_ok=True)

bucket_dir_dict = {"J5FSD": J5FSD_BUCKET_PATH, "J5FSD_2": J5FSD_2_BUCKET_PATH}

# Datasets.
dataset_name = "Lidar3M1"
train_pkl_name = None
val_pkl_name = None

dataset = SD_DEMO_DATASET[dataset_name]
bucket_dir = bucket_dir_dict[dataset["bucket"]]
map_path_func = dataset["map_path_func"]
data_token2path_mapping = dataset["data_token2path_mapping"]
use_train_pkl = (
    dataset["basic_train_pkl"]
    if train_pkl_name is None
    else dataset[train_pkl_name]
)
use_val_pkl = (
    dataset["basic_val_pkl"] if val_pkl_name is None else dataset[val_pkl_name]
)
train_dataset_pkl_path = os.path.join(bucket_dir, use_train_pkl)
val_dataset_pkl_path = os.path.join(bucket_dir, use_val_pkl)

color_table = gen_roadmap_with_vl_color_table()

# 0.2. Parameters.
batch_size_per_gpu = 16
train_num_workers, val_num_workers = 8, 4
max_epoch = 5
log_freq = 100
pkl_fps = 2
video_fps = pkl_fps * 5  # 5x video
data_shape = (7, 512, 512)
anchor_num = 124
seq_length = 8 * pkl_fps
seq_period = int(pkl_fps / 2)
sample_step = 1
context_frames = 4
traj_len = 12

bev_origin_x, bev_origin_y, img_resolution = 72.4, 51.2, 0.2
coords_reverse = True  # If use bev coordinates, this must be true,
img_h, img_w = data_shape[1:3]
x_ratio = bev_origin_x / img_resolution / img_h
y_ratio = bev_origin_y / img_resolution / img_w
train_bounce_thr = np.pi / 8
val_bounce_thr = np.pi / 3
veh_type_id = 1
ped_cyc_type_id = [2, 18]
roi_input_size = 16
roi_output_size = 8
roi_spatial_scale = int(img_h / roi_input_size)
leaving_mode = "all"
yaw_select_type = [5, 5, 5]
detect_peds_by_shape = True
ped_shape_thr = 1
use_state_vectors = True
use_instant_state_vectors = True
if_clip_state_vectors = True
assign_trajs_for_filtered_obs = True
use_data_augmentation = False
viz_on_every_epoch = True

# 1. Setup data loader.
train_transforms = [
    dict(
        type="GenSeqCenter",
        anchor_frame="last_context",
        context_frames=context_frames,
        augmentation=use_data_augmentation,
    )
]
val_transforms = [
    dict(
        type="GenSeqCenter",
        anchor_frame="last_context",
        context_frames=context_frames,
        augmentation=False,
    ),
]
common_transforms_1 = [
    dict(
        type="GenBoundingBox",
        gen_ego_bbox=True,
        clockwise=True,
        min_shape=1,
        const_z_value=1.5,
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
    # dict(type="LateralSmoothing"),
    dict(
        type="GenBoundingBox",
        gen_ego_bbox=False,
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
        type="SelectYawArray",
        yaw_select_type=yaw_select_type,
    ),
    dict(type="EgoCentricGt"),
]
train_transforms += common_transforms_1
val_transforms += common_transforms_1

train_transforms += [
    dict(
        type="FilterObstacles",
        is_training=True,
        is_multiagent=True,
        ego_track_id=-BaseTrajDataset.ANSWER,
        ped_cyc_type_id=ped_cyc_type_id,
        leaving_mode=leaving_mode,
        # range of prediction, veh[x_max, x_min, y_max, y_min],
        # ped[...], cycli[...]]
        valid_distance=[
            [50, -30, 50, -50],
            [30, 0, 10, -10],
            [50, -30, 20, -20],
        ],
        num_frame_thr=2,
        bounce_thr=train_bounce_thr,
        ped_drift_thr=1.5,  # i.e., 3 m/s
        static_thr=[1, 0.2, 0.2],
        if_train_filter_bounce=[True, True],
        if_val_filter_bounce=[True, True],
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
        ego_track_id=-BaseTrajDataset.ANSWER,
        ped_cyc_type_id=ped_cyc_type_id,
        leaving_mode=leaving_mode,
        # range of prediction, veh[x_max, x_min, y_max, y_min],
        # ped[...], cycli[...]]
        valid_distance=[
            [50, -30, 50, -50],
            [30, 0, 10, -10],
            [50, -30, 20, -20],
        ],
        num_frame_thr=2,
        bounce_thr=val_bounce_thr,
        ped_drift_thr=1.5,  # i.e., 3 m/s
        static_thr=[1, 0.2, 0.2],
        if_train_filter_bounce=[True, True],
        if_val_filter_bounce=[True, True],
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
        type="OccupancyMapRender",
        map_height=img_h,
        map_width=img_w,
        context_frames=context_frames,
        render_ego=True,
        normalize=True,
    ),
    dict(
        type="GetBEVLocalMapByTimestamp",
        image_dir=bucket_dir,
        image_suffix=".png",
        map_height=img_h,
        map_width=img_w,
        data_token2path_mapping=data_token2path_mapping,
        map_path_func=map_path_func,
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
        assign_trajs_for_filtered_obs=assign_trajs_for_filtered_obs,
        reverse=coords_reverse,
    ),
    dict(
        type="GetBEVHomography",
        src_shape=roi_input_size,
        dst_shape=roi_output_size,
        x_origin_ratio=x_ratio,
        y_origin_ratio=y_ratio,
        roi_spatial_scale=roi_spatial_scale,
        swap_xy=True,
    ),
    dict(
        type="MapAugmentation",
        src_resolution=0.2,
        dst_resolution=img_resolution,
        map_origin_x=bev_origin_x,
        map_origin_y=bev_origin_y,
        swap_xy=True,
    ),
]
train_transforms += common_transforms_2
val_transforms += common_transforms_2

train_dataloader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="PickledTdtDataset",
        pkl_path=train_dataset_pkl_path,
        seq_length=seq_length,
        seq_period=seq_period,
        sample_step=sample_step,
        context_frames=context_frames,
        transforms=train_transforms,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    collate_fn=collate_uniformpath_viz,
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=train_num_workers,
)

val_dataloader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="PickledTdtDataset",
        pkl_path=val_dataset_pkl_path,
        seq_length=seq_length,
        seq_period=seq_period,
        sample_step=sample_step,
        context_frames=context_frames,
        transforms=val_transforms,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    collate_fn=collate_uniformpath_viz,
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=val_num_workers,
)

# 2. Setup Model.
ckpt_use_head_name = "Uniformpath"

model = dict(
    type="Uniformpath",
    heads={"Uniformpath": None},
    anchor_num=anchor_num,
    post_process=dict(
        type="UniformpathPostProcessor",
        ego_track_id=-BaseTrajDataset.ANSWER,
    ),
    table=color_table,
)
head_name_dict = dict(
    traj_head="Uniformpath",
    track_valid_head="Uniformpath",
)


# 3. Setup metrics.
# --1. Validation metric updater.
def update_metric(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


# -- 1.1. Setup single or multiple metrics. Multiple metrics is used
#         for evaluate multiple heads or multiple classes respectively.
metrics = [
    dict(
        type="TrajPredMetric",
        head_names=head_name_dict,
        k_values=[1],
        name=ckpt_use_head_name,
    )
]

# -- 1.2. a validation metric updater, if you setup multiple metrics,
#         split their respective model outputs in "update_metric" func.
val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix="Validation_" + task_name,
)

# 4. Setup callbacks.
val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
)

viz_callback = dict(
    type="TrajPredViz",
    head_names=head_name_dict,
    height=img_h,
    width=img_w,
    resolution=img_resolution,
    map_origin=[bev_origin_x, bev_origin_y],
    video_save_path=viz_dir,
    video_name="qat_best_viz",
    video_fps=video_fps,
    viz_on_every_epoch=viz_on_every_epoch,
    max_epoch=max_epoch,
    ego_track_id=-BaseTrajDataset.ANSWER,
    num_traj_to_viz=5,
    num_workers=val_num_workers,
)
val_callback = dict(
    type="Validation",
    data_loader=val_dataloader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater, viz_callback],
    val_model=None,
    val_on_train_end=True,
)

qat_trainer = dict(
    type="Trainer",
    model=model,
    data_loader=train_dataloader,
    optimizer=None,
    batch_processor=None,
    num_epochs=0,
    device=None,
    callbacks=[
        val_callback,
        dict(
            type="Checkpoint",
            save_dir=ckpt_dir,
            name_prefix="qat-",
            strict_match=True,
            mode="min",
            monitor_metric_key=f"head_{ckpt_use_head_name}_min_ade_5",
        ),
    ],
    val_metrics=metrics,
)

qat_solver = dict(
    trainer=qat_trainer,
    quantize=False,
    check_quantize_model=False,
    pre_step="float",
    pre_step_checkpoint=os.path.join(
        ckpt_dir, "float-checkpoint-best.pth.tar"
    ),
    strict_match=True,
    allow_miss=True,
    ignore_extra=True,
)

step2solver = dict(qat=qat_solver)
