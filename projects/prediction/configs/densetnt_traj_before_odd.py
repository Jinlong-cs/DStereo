# Copyright (c) Horizon Robotics. All rights reserved.

import copy
import getpass
import json
import os
from collections import OrderedDict

import numpy as np
import torch
from horizon_plugin_pytorch.quantization import March
from processed_dataset import SD_DEMO_DATASET, VW_DEMO_DATASET

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.data.collates.traj_pred_collates import (
    collate_vectornet,
    collate_vectornet_viz,
)
from hat.data.datasets.traj_pred_dataset import BaseTrajDataset
from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2

cudnn_benchmark = False
seed = None
log_rank_zero_only = True
march = March.BAYES  # March.BAYES for bpu2.5 or "bayes1" for bpu2.2

# Task configurations
# task_name = "densetnt_stage_1_fulldata_qat_change_qconfig_2"
task_name = "densetnt_stage_1_odd"
qat_ckpt_for_int_infer = "qat-checkpoint-best.pth.tar"
float_ckpt_for_int_infer = "float-checkpoint-best.pth.tar"
predictor_type = "densetnt"
project = "sd_demo"
dataset_name = "multi-task_perrec"

# tiny dataset
# train_pkl_name = "densetnt_tiny_train"  # 3
# val_pkl_name = "densetnt_tiny_test"  # 1

# train_pkl_name = "small_test_densetnt" # 22w 环境重建
# val_pkl_name = "small_test_densetnt" # 3w 环境重建

train_pkl_name = "full_train"  # 120w 环境重建
val_pkl_name = "full_val"  # 16w 环境重建


# 下方数据集需要修改配置：float_epoch, qat_epoch = 20, 5
# train_pkl_name = "full_fus_data_train"  # 438w 环境融合
# val_pkl_name = "full_fus_data_val"  # 44w 环境融合
#  val_pkl_name = "full_fus_data_viz"  # 16w 环境融合

# Very import configurations!!!
local_train = False
is_training = True
if "bucket" in os.listdir("/"):
    local_train = False
use_state_vectors = True  # if add state_vector in the input of model
do_viz_after_validation = False  # 训练的时候不进行任何可视化，只有验证的时候可视化
blend_traj_by_history = False  # if scaling pred, set True for eval
use_set_predictor = False
# num_set_predictor_head根据源代码是4，这里改成1是为了省下一个torch.gather
num_set_predictor_head = 1
balance_loss = False
l1_weight = 5 if balance_loss else 1
is_soft_cls = False
adjust_teacher_forcing = False
# Range of prediction, veh[x_max, x_min, y_max, y_min], ped[...], cycli[...]
veh_range = [50.0, -30.0, 50.0, -50.0]
ped_range = [30.0, 0.0, 10.0, -10.0]
cyc_range = [50.0, -30.0, 20.0, -20.0]

# 0.1. Paths.
if local_train:
    device_ids = [3]
    # Set your local path here.
    user_name = getpass.getuser()
    J5FSD_BUCKET_PATH = "/horizon-bucket/J5FSD/"
    J5FSD_2_BUCKET_PATH = "/horizon-bucket/J5FSD_2/"
    SD_Algo_BUCKET_PATH = "/horizon-bucket/SD_Algorithm/"
    ckpt_dir = f"/jfs-public/users/{user_name}/results/{task_name}/checkpoint"
    viz_dir = f"/jfs-public/users/{user_name}/results/{task_name}/viz"
    tensorboard_dir = (
        f"/jfs-public/users/{user_name}/results/{task_name}/tensorboard"
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
    "J5FSD": J5FSD_BUCKET_PATH,
    "SD_Algorithm": SD_Algo_BUCKET_PATH,
}

# Datasets.
if project == "vw_demo":
    project_dataset = VW_DEMO_DATASET
elif project == "sd_demo":
    project_dataset = SD_DEMO_DATASET
else:
    raise ValueError(f"Unknown project {project}.")

dataset = project_dataset[dataset_name]
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

endpts_file_dir = os.path.join(
    bucket_dir,
    "11_perception_prediction/02_user/shiqi.tan/data/end_point_anchors_075_cls.pkl",  # noqa
)

float_epoch, quanti_epoch = 30, 5
if use_set_predictor:
    float_epoch = 20
    quanti_epoch = 5
train_num_workers, val_num_workers = 8, 8
batch_size_per_gpu = 32
float_lr, quanti_lr = 0.003, 0.0003  # default 0.001, 0.00001
if use_set_predictor:
    float_lr = 0.001
    quanti_lr = 0.0001
weight_decay, quanti_weight_decay = 0.001, 0.001  # default 0.001, 0.0001
float_warmup_epoch, qat_warmup_epoch = 1, 1
log_freq = 500

# -- Parameters about the trajectory prediction task.
his_time_horizon = 2
fut_time_horizon = 6

pkl_fps = 2
seq_length = (his_time_horizon + fut_time_horizon) * pkl_fps
seq_period = int(pkl_fps / 2)
sample_step = 1
context_frames = his_time_horizon * pkl_fps
gt_traj_len = fut_time_horizon * pkl_fps
target_freq = 2
fut_traj_len = int(fut_time_horizon * pkl_fps)

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
veh_lateral_drift_thr = 1.5
veh_type_id = 1
ped_cyc_type_id = [2, 18]
leaving_mode = "all"
yaw_select_type = [6, 6, 6]
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
input_num_ele_seg = 128

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
traj_sample_ratio = int(target_freq / pkl_fps)
if_itp_traj = True
if if_itp_traj:
    traj_polyline_len = traj_sample_ratio * (context_frames - 1)
else:
    traj_polyline_len = 3

# -- Parameters about models
subgraph_hidden_size = 64
globalgraph_hidden_size = 64  # 64 for baseline
num_sub_graph_layers = 3  # 3 for baseline
use_out_fc = True
num_attn_head = 4
backbone_fc_out_channels = 128
mlp_predictor_hidden_size = 128
assert (
    num_attn_head > 1
), " = 1 will cause compile failed for repeat (1,1,1,1) in globalgraph"
if use_out_fc:
    mlp_input_channels = backbone_fc_out_channels
else:
    mlp_input_channels = globalgraph_hidden_size * num_attn_head

static_thr = [0.1, 0.2, 0.2]

# - Parameters about head
k_values = [1, 5]
hidden_size = subgraph_hidden_size
ele_feat_hidden_size = subgraph_hidden_size
goals_hidden_layers = 2
goals_mlp_layers = 1
num_dyn_obs = max_obs_num
# 是否使用lane scoring
use_lane_scoring = False
# ---- GenObstaclesGoals
# --------道路元素
include_road_ele = False
# 如果使用密集采点(include_besides=True),那么 dense_goals_dis 会决定垂线上采点之间的间隔，单位 m
dense_goals_dis = 1.7
# dense_goals_dis=1
# 每个道路元素的vector被分为几段
road_ele_divide_num = 1
assert road_ele_divide_num >= 1, "road_ele_divide_num must >=1"
# 每个polyline上每连续多少点取1个点
goal_interval = 2
assert (
    goal_interval >= 1 and goal_interval <= polyline_seg_len
), "goal_interval should be an int between 0 and 10"
# 使用差分轨迹
use_diff_trajs = True
use_optim_topk_in_val = False

# --------安全区
use_safe_area = True
expand_traj_method = "CV"
use_his_traj = False  # 安全区是否要包括历史轨迹
expand_base = 4
expand_ratio = 0.2
edge_divide_num = 3  # 每条安全区边被分为几段
vertical_divide_num = 5  # 每条安全区垂线被分为几段


# ---路口垂线
add_crosswalk_vertical = True

# --------采样点总数
num_goals = 1024
# 缩放采样点坐标的数值
goal_coords_scale = [max(road_feat_scale[:4])]

# ---------去重算法
hashv = 1.0

# 补充采样点坐标
pad_coords = np.array([-40.0, 0.0])

# -- Parameters about postprocessing and visualization.
use_nms = False
video_fps = pkl_fps * 5  # 5x video
use_extend_state_vectors = False
num_extend_state_vectors = 7
if use_extend_state_vectors:
    num_state_vectors += num_extend_state_vectors

# 1. Setup data loader.
train_transforms = [
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
]
val_transforms = copy.deepcopy(train_transforms)

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
        is_training=True,
    ),
    dict(
        type="FilterObstaclesByFuture",
        is_multiagent=True,
        ego_track_id=-BaseTrajDataset.ANSWER,
        ped_cyc_type_id=ped_cyc_type_id,
        leaving_mode=leaving_mode,
        num_frame_thr=[2, 2, 2],
        bounce_thr=train_bounce_thr,
        speed_drift_thr=speed_drift_thr,
        veh_lateral_drift_thr=veh_lateral_drift_thr,
        static_thr=static_thr,
        if_filter_bounce=[True, True],
        classify_by_shape=detect_peds_by_shape,
        ped_shape_thr=ped_shape_thr,
        enable_incomplete_gts=True,
        key_name="valid_future_track_ids",
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
        filter_invalid_end=True,
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
        shuffle=True,
        scale=road_feat_scale,
        valid_img_coords_key="valid_img_coords",
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
            veh_range,
            ped_range,
            cyc_range,
        ],
        num_frame_thr=2,
        bounce_thr=val_bounce_thr,
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
        is_training=False,
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
        filter_invalid_end=False,
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
        shuffle=False,
        scale=road_feat_scale,
        valid_img_coords_key="valid_img_coords",
    ),
]

common_transforms_2 = [
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
        type="GenObstaclesGoalsV3",
        num_goals=num_goals,
        goal_coords_scale=goal_coords_scale,
        pad_coords=pad_coords,
        endpts_file_dir=endpts_file_dir,
        use_diff_trajs=use_diff_trajs,
    ),
]

train_transforms += common_transforms_2
val_transforms += common_transforms_2

# 用于可视化的函数，训练时可以注释
if do_viz_after_validation:
    val_transforms.append(
        dict(
            type="GenVisLanes",
            map_origin_params=map_origin_params,
            element_keys=all_elements,
            image_coordinates=image_coordinates,
        )
    )
    # 使用基于向量化地图的可视化
    val_transforms.append(
        dict(
            type="GetBevVectorizedMap",
            bev_origin_x=bev_origin_x,
            bev_origin_y=bev_origin_y,
            img_resolution=img_resolution,
        )
    )
# 安全区对算法端指标无影响,默认不使用
# val_transforms.append(
#     dict(
#         type="GetObstacleSafeArea",
#         traj_len=traj_len,
#         expand_base=expand_base,
#         expand_ratio=expand_ratio,
#         use_his_traj=False,
#         reverse=coords_reverse,
#         img_params=[bev_origin_x, bev_origin_y, img_resolution],
#     )
# )

if not local_train and "lmdb_path_replace_func" in dataset:
    ds_wrap_func = [dataset["lmdb_path_replace_func"]]  # just cluster train
else:
    ds_wrap_func = None


train_dataloader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="PickledTdtDatasetV2",
        pkl_path=train_dataset_pkl_path,
        transforms=train_transforms,
        wrap_func=ds_wrap_func,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    collate_fn=collate_vectornet,
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=train_num_workers,
)


if "test_dataset" in val_pkl_name:
    val_dataloader = dict(
        type=torch.utils.data.DataLoader,
        dataset=dict(
            type="TestDataset",
            pkl_path=val_dataset_pkl_path,
            transforms=val_transforms,
        ),
        sampler=dict(type=torch.utils.data.DistributedSampler),
        collate_fn=collate_vectornet_viz,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=val_num_workers,
    )
else:

    val_dataloader = dict(
        type=torch.utils.data.DataLoader,
        dataset=dict(
            type="PickledTdtDatasetV2",
            pkl_path=val_dataset_pkl_path,
            transforms=val_transforms,
            wrap_func=ds_wrap_func,
        ),
        sampler=dict(type=torch.utils.data.DistributedSampler),
        collate_fn=collate_vectornet_viz,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=val_num_workers,
    )


# 2. Setup Model.
if predictor_type == "densetnt":
    in_channels = dict(
        graph_feat=mlp_input_channels,
    )
    if use_state_vectors:
        in_channels["state_vector_feat"] = num_state_vectors
    model_heads = OrderedDict(
        normal_head=dict(
            type="DenseTNTHead",
            k_values=k_values,
            in_channels=in_channels,
            hidden_size=hidden_size,
            traj_feat_hidden_size=ele_feat_hidden_size,
            road_feat_hidden_size=ele_feat_hidden_size,
            traj_feat_num=max_obs_num,
            road_feat_num=max_num_ele_seg,
            num_goals=num_goals,
            num_dyn_obs=num_dyn_obs,
            num_road_ele=input_num_ele_seg,
            num_fut_frame=fut_time_horizon * pkl_fps,
            goal_coords_scale=goal_coords_scale[0],
            use_lane_scoring=use_lane_scoring,
            adjust_teacher_forcing=adjust_teacher_forcing,
            use_set_predictor=use_set_predictor,
            num_set_predictor_head=num_set_predictor_head,
        )
    )
    ckpt_use_head_names = ["traj_head"]
else:
    raise ValueError(f"Undefined prediction decoder {predictor_type}")

head_name_dict = dict(
    traj_head=ckpt_use_head_names[0],
)

necks = OrderedDict(
    traj_neck=dict(
        type="DenseTNTNeck",
        is_int_infer_model=False,
        use_state_vectors=use_state_vectors,
        backbone_output_all_feats=True,
    ),
)

# TODO(dukai.dong):当前DenseTNT没有后处理,
# 输出的轨迹差分需要分别在  loss,metrics和viz中转成正常轨迹
# 推荐后续在后处理中实现将轨迹差分转成正常轨迹
if use_set_predictor:
    loss = [
        dict(
            type="DenseTNTStageTwoLoss",
            k_points=max(k_values),
            goal_coords_scale=goal_coords_scale[0],
            gen_pseudo_label_by="pred",
        ),
    ]

    all_loss = ["total_loss", "head_loss", "set_loss"]

    loss_qat = [
        dict(
            type="DenseTNTloss",
        ),
        dict(
            type="DenseTNTScoreloss",
        ),
        dict(
            type="DenseTNTStageTwoLoss",
            k_points=max(k_values),
            goal_coords_scale=goal_coords_scale[0],
            gen_pseudo_label_by="pred",
        ),
    ]
    all_loss_qat = [
        "total_loss",
        "traj_l1_loss",
        "scores_nll_loss",
        "set_scores_nll_loss",
        "road_nll_loss",
        "head_loss",
        "set_loss",
    ]
else:
    loss = dict(
        type="DenseTNTloss",
    )
    all_loss = [
        "total_loss",
        "traj_l1_loss",
        "scores_nll_loss",
        "road_nll_loss",
    ]

model = dict(
    type="DenseTNT",
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
        num_sub_graph_layers=num_sub_graph_layers,
        num_attention_heads=num_attn_head,
        use_out_fc=use_out_fc,
        fc_out_channels=mlp_input_channels,
        output_all_feats=True,
        freeze_grad=use_set_predictor,
    ),
    necks=necks,
    heads=model_heads,
    post_process=None,
    losses=loss,
    is_int_infer_model=False,
)

# 3. Setup metrics.
metric_updater = dict(
    type="MetricUpdater",
    metrics=[dict(type="LossShow", name=i) for i in all_loss],
    metric_update_func=update_metric_using_regex(
        per_metric_patterns=[
            dict(label_pattern=None, pred_pattern=i) for i in all_loss
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
    type="TrajPredMetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix="Validation_" + task_name,
    display_head_names=ckpt_use_head_names,
    traj_display_k_values=k_values,
    display_cls_dict={
        "vehicle": 0,
        "pedestrain": 1,
        "cyclist": 2,
    },
    save_metric_path=ckpt_dir,
    support_behav_metric=False,
    traj_metrics_keys=[
        "min_ade",
        "min_fde",
        "miss_rate",
        "min_goal_fde",
        "minimal_goal_fde",
        "min_label_fde",
    ],
)

# 4. Setup callbacks.
if predictor_type == "mlp":
    loss_str = "^.*reg_loss.*"
    metrics = [
        dict(
            type="TrajPredMetric",
            head_names=head_name_dict,
            k_values=[1],
            name=ckpt_use_head_names[0],
        ),
    ]

    num_traj_to_viz = 1
elif predictor_type == "densetnt":
    # TODO (shengzhe.dai or shiqi.tan): 这里似乎是不应该匹配所有loss，而是只针对
    # total_loss，目前这种写法用于更新参数的loss是实际loss的二倍，相当于改了学习率
    # 但此前的所有模型在这里都有同样的问题，在不同的loss权重不都为1时，这里的问题会。
    # 使实际上的权重比例比用户设定的更小（更不悬殊）。
    loss_str = "^.*loss.*"
    metrics = [
        dict(
            type="DenseTNTtrajMetric",
            head_names=head_name_dict,
            k_values=k_values,
            name=ckpt_use_head_names[0],
            goal_coords_scale=goal_coords_scale[0],
            sorted_top_k=True,
        ),
    ]

    num_traj_to_viz = max(k_values)
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
    warmup_len=float_warmup_epoch,
    step_log_interval=log_freq,
)

# tensorboard
# job on aidi platform will have TENSORBOARD_LOG_PATH env,
# where the saved tb can be shown through aidi web UI
tb_traj_loss_callback = dict(
    type="TensorBoard",
    save_dir=os.path.join(tensorboard_dir, "float", "loss"),
    loss_name_reg="traj_l1_loss",
    update_freq=100,
    update_by="step",
)

tb_goal_score_loss_callback = dict(
    type="TensorBoard",
    save_dir=os.path.join(tensorboard_dir, "float", "loss"),
    loss_name_reg="scores_nll_loss",
    update_freq=100,
    update_by="step",
)

# 5. Trainer and solver.
# -- 1. float.
float_val_callback = copy.deepcopy(val_callback)  # 训练时候的val_callback,不需要做可视化
if use_set_predictor:
    monitor_metric_key = f"head_{ckpt_use_head_names[0]}_min_goal_fde_1"
else:
    monitor_metric_key = f"head_{ckpt_use_head_names[0]}_min_ade_1"

float_model = copy.deepcopy(model)
float_model["heads"]["normal_head"]["training_step"] = "float"
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
            monitor_metric_key=monitor_metric_key,
        ),
        tb_traj_loss_callback,
        tb_goal_score_loss_callback,
    ],
    sync_bn=True,
    train_metrics=metrics,
    val_metrics=metrics,
    # resume_optimizer=True,
    # resume_epoch_or_step=True,
)


def update_state_dict_densetnt_stage2_float(state_dict):
    remap_keys = [
        "normal_head.goals_2D_cross_attention.wq.weight",
        "normal_head.goals_2D_cross_attention.wq.bias",
        "normal_head.goals_2D_cross_attention.wk.weight",
        "normal_head.goals_2D_cross_attention.wk.bias",
        "normal_head.goals_2D_cross_attention.wv.weight",
        "normal_head.goals_2D_cross_attention.wv.bias",
    ]

    for key in remap_keys:
        tmp_key = key.replace(
            "goals_2D_cross_attention", "set_predict_cross_attn"
        )
        state_dict[tmp_key] = state_dict[key]

    return state_dict


if use_set_predictor:
    float_trainer["model_convert_pipeline"] = dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-best_stage1.pth.tar"
                ),
                state_dict_update_func=update_state_dict_densetnt_stage2_float,
                allow_miss=True,
                ignore_extra=True,
            ),
        ],
    )

float_predict_model = copy.deepcopy(model)
float_predict_model["heads"]["normal_head"]["training_step"] = "float"
float_predict_model["heads"]["normal_head"][
    "use_optim_topk"
] = use_optim_topk_in_val
float_predictor = dict(
    type="Predictor",
    model=float_predict_model,
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
    log_interval=50,
)
if do_viz_after_validation:
    float_viz_callback = dict(
        type="TrajPredViz",
        head_names=head_name_dict,
        height=img_h,
        width=img_w,
        resolution=img_resolution,
        map_origin=[bev_origin_x, bev_origin_y],
        video_save_path=viz_dir,
        video_name="float_best_viz",
        video_fps=video_fps,
        viz_on_every_epoch=False,
        max_epoch=0,
        ego_track_id=-BaseTrajDataset.ANSWER,
        num_traj_to_viz=num_traj_to_viz,
        num_workers=val_num_workers,
        use_behav_head=False,
        first_fut_frame_idx=context_frames,
        fut_len=fut_traj_len,
    )
    float_predictor["callbacks"].append(float_viz_callback)

# -- 1.5. calibration.(Can be skipped due to little effect.)
# Note: The transforms of the dataset during calibration can be
# consistent with that during training or validation, or customized.
# Default used `val_batch_processor`.
calibration_data_loader = copy.deepcopy(train_dataloader)
calibration_data_loader.pop("sampler")
calibration_batch_processor = copy.deepcopy(val_batch_processor)
calibration_val_callback = copy.deepcopy(val_callback)
calib_model = copy.deepcopy(model)
calib_model["heads"]["normal_head"]["training_step"] = "calib"
calibration_trainer = dict(
    type="Calibrator",
    model=calib_model,
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
            dict(type="Float2Calibration"),
        ],
    ),
    data_loader=calibration_data_loader,
    batch_processor=calibration_batch_processor,
    num_steps=650,
    callbacks=[
        stats_callback,
        # calibration_val_callback,
        dict(
            type="Checkpoint",
            save_dir=ckpt_dir,
            name_prefix="calibrator-",
            strict_match=True,
            # mode="min",
            # monitor_metric_key=monitor_metric_key,
        ),
    ],
    val_metrics=metrics,
)

calibration_predictor = dict(
    type="Predictor",
    model=calib_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibrator-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=calibration_data_loader,
    batch_processor=calibration_batch_processor,
    device=None,
    metrics=metrics,
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)


# -- 2. qat.
def update_state_dict_densetnt_stage2_qat(state_dict):
    from hat.utils.checkpoint import load_checkpoint

    qat_state_dict = load_checkpoint(
        os.path.join(ckpt_dir, "qat-checkpoint-best_stage1.pth.tar"),
        map_location="cpu",
    )["state_dict"]

    to_remap_keys = []
    for key in qat_state_dict:
        if "goals_2D_cross_attention" in key:
            to_remap_keys.append(key)

    for key in to_remap_keys:
        tmp_key = key.replace(
            "goals_2D_cross_attention", "set_predict_cross_attn"
        )
        qat_state_dict[tmp_key] = qat_state_dict[key]

    state_dict.update(qat_state_dict)

    return state_dict


qat_val_callback = copy.deepcopy(val_callback)
qat_val_callback["callbacks"] = [val_metric_updater]
qat_lr_update_callback = copy.deepcopy(lr_update_callback)
qat_lr_update_callback["warmup_len"] = qat_warmup_epoch
# 可视化
if do_viz_after_validation:
    qat_viz_callback = copy.deepcopy(float_viz_callback)
    qat_viz_callback["video_name"] = "qat_best_viz"
    qat_viz_callback["max_epoch"] = quanti_epoch
    qat_val_callback["callbacks"].append(qat_viz_callback)

qat_tb_save_dir = os.path.join(tensorboard_dir, "qat", "loss")
qat_tb_traj_loss_callback = copy.deepcopy(tb_traj_loss_callback)
qat_tb_traj_loss_callback["save_dir"] = qat_tb_save_dir
qat_tb_goal_score_loss_callback = copy.deepcopy(tb_goal_score_loss_callback)
qat_tb_goal_score_loss_callback["save_dir"] = qat_tb_save_dir

qat_tb_val_save_dir = os.path.join(tensorboard_dir, "qat", "train")

qat_model = copy.deepcopy(model)
qat_model["heads"]["normal_head"]["training_step"] = "qat"
qat_metric_updater = metric_updater
if use_set_predictor:
    qat_ckpt_update_func = update_state_dict_densetnt_stage2_qat
else:
    qat_ckpt_update_func = None
qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=qat_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        qconfig_params=dict(
            # 如果没有经过 calibration，且这里 averaging_constant=0.0，那么 scale 只会更新一部分
            activation_qat_qkwargs=dict(
                averaging_constant=0.01,
            ),
            weight_qat_qkwargs=dict(
                averaging_constant=0.01,
            ),
        ),
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibrator-checkpoint-last.pth.tar"
                ),
                state_dict_update_func=qat_ckpt_update_func,
                ignore_extra=True,
            ),
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
        qat_metric_updater,
        qat_lr_update_callback,
        stats_callback,
        qat_val_callback,
        dict(
            type="Checkpoint",
            save_dir=ckpt_dir,
            name_prefix="qat-",
            strict_match=True,
            mode="min",
            monitor_metric_key=monitor_metric_key,
        ),
        qat_tb_traj_loss_callback,
        qat_tb_goal_score_loss_callback,
    ],
    train_metrics=metrics,
    val_metrics=metrics,
)

qat_predict_model = copy.deepcopy(model)
qat_predict_model["heads"]["normal_head"]["training_step"] = "qat"
qat_predict_model["heads"]["normal_head"][
    "use_optim_topk"
] = use_optim_topk_in_val
qat_allow_miss = use_optim_topk_in_val
qat_predictor = dict(
    type="Predictor",
    model=qat_predict_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
                allow_miss=qat_allow_miss,
                verbose=True,
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
    log_interval=50,
)

# -- 3. int.
# ---- add description
params_desc = dict(
    context_frames=context_frames,
    image_h=img_h,
    image_w=img_w,
    resolution=img_resolution,
    BEV_origin_x=bev_origin_x,
    BEV_origin_y=bev_origin_y,
    warp_resolution=img_resolution,
    num_frame_thr=2,
    veh_select_yaw_type=yaw_select_type[0],
    ped_select_yaw_type=yaw_select_type[1],
    cyc_select_yaw_type=yaw_select_type[2],
    traj_len=gt_traj_len,
    top_k=max(k_values),
    bounce_thr=val_bounce_thr,
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
    include_road_ele=1 if include_road_ele else 0,
    use_safe_area=1 if use_safe_area else 0,
    num_goals=num_goals,
    road_ele_divide_num=road_ele_divide_num,
    dense_goals_dis=dense_goals_dis,
    expand_base=expand_base,
    expand_ratio=expand_ratio,
    edge_divide_num=edge_divide_num,
    vertical_divide_num=vertical_divide_num,
    hashv=hashv,
    goal_interval=goal_interval,
    input_num_ele_seg=input_num_ele_seg,
    pad_point_first=pad_coords[0],
    pad_point_second=pad_coords[1],
)

desc_name = "traj_pred"

add_desc_pp = dict(
    type="AddDesc",
    per_tensor_desc=[
        # num of desc == num of pred output tensors (one tensor per out stride)
        json.dumps(
            dict(
                task=f"{desc_name}_predict_trajs",
                **params_desc,
            )
        ),
        json.dumps(
            dict(
                task=f"{desc_name}_real_scores",
                **params_desc,
            )
        ),
    ],
)

deploy_model = copy.deepcopy(model)
deploy_model["post_process"] = add_desc_pp
deploy_model["losses"] = None
deploy_model["is_int_infer_model"] = True
for _, neck in deploy_model["necks"].items():
    neck["is_int_infer_model"] = True
for _, head in deploy_model["heads"].items():
    head["training_step"] = "int_infer"
int_val_callback = copy.deepcopy(val_callback)
int_val_callback["callbacks"] = []
int_val_callback["data_loader"] = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="PickledTdtDatasetV2",
        pkl_path=val_dataset_pkl_path,
        transforms=val_transforms,
        wrap_func=ds_wrap_func,
    ),
    collate_fn=collate_vectornet,
    batch_size=1,
    shuffle=False,
    num_workers=val_num_workers,
)

hdm_max_num_obs = 16
num_all_poly = input_num_ele_seg + max_obs_num
# permute(0, 3, 1, 2) here
deploy_inputs = dict(
    struct_road_feats=torch.randn(
        (
            hdm_max_num_obs,
            road_feat_dim,
            input_num_ele_seg,
            polyline_seg_len - 1,
        )
    ),
    struct_traj_feats=torch.randn(
        (hdm_max_num_obs, traj_feat_dim, max_obs_num, traj_polyline_len)
    ),
    attention_mask=torch.ones(
        (hdm_max_num_obs, 1, num_all_poly, num_all_poly)
    ),
    goal_coords=torch.randn((hdm_max_num_obs, 2, 1, num_goals)),
)
if use_state_vectors:
    deploy_inputs["state_vectors"] = torch.randn(
        (hdm_max_num_obs, num_state_vectors, 1, 1)
    )

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix="int-",
    strict_match=True,
    mode="min",
    monitor_metric_key=monitor_metric_key,
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

deploy_predict_model = copy.deepcopy(model)
deploy_predict_model["heads"]["normal_head"]["training_step"] = "int_infer"
deploy_predict_model["heads"]["normal_head"][
    "use_optim_topk"
] = use_optim_topk_in_val
int_infer_predictor = dict(
    type="Predictor",
    model=deploy_predict_model,
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

# model_profiler_solver = dict(
#     model=deploy_model,
#     inputs=deploy_inputs,
#     model_convert_pipeline=[
#         qat_predictor["model_convert_pipeline"],
#         int_infer_predictor["model_convert_pipeline"],
#     ],
#     tool=dict(
#         type="FeaturemapSimilarity",
#         similarity_func="Cosine",
#         threshold=None,
#     ),
# )

# from hat.registry import build_from_registry
# test_dataloader =  build_from_registry(val_dataloader)
# test_inputs = test_dataloader.collate_fn([test_dataloader.dataset[0]])
# road_mask = test_inputs["struct_road_masks"]
# traj_mask = test_inputs["struct_traj_masks"]
# concat_mask = torch.cat([traj_mask, road_mask], dim=1)
# attention_mask = torch.matmul(
#     concat_mask[:, :, None], concat_mask[:, None, :]
# )
# attention_mask = attention_mask[:, None, :, :]
# test_inputs["attention_mask"] = attention_mask

# test_inputs["struct_road_feats"] = test_inputs["struct_road_feats"].permute(
#     0, 3, 1, 2
# )
# test_inputs["struct_traj_feats"] = test_inputs["struct_traj_feats"].permute(
#     0, 3, 1, 2
# )
# test_inputs["future_trajectories"] = test_inputs[
#     "future_trajectories"
# ].unsqueeze(1)

# profiler_model = copy.deepcopy(float_predict_model)
# profiler_model["use_cuda"] = False
# profiler_model["necks"]["traj_neck"]["use_cuda"] = False
# model_profiler_solver = dict(
#     model=profiler_model,
#     inputs=test_inputs,
#     model_convert_pipeline=[
#         float_predictor["model_convert_pipeline"],
#         qat_predictor["model_convert_pipeline"],
#     ],
#     tool=dict(
#         type="ModelProfiler",
#         mode="FvsQ",
#         out_dir=None,
#         kwargs_dict=dict(
#             FeaturemapSimilarity=dict(
#                 similarity_func="Cosine"
#             )
#         )
#     ),
# )
