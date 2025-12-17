# Copyright (c) Horizon Robotics. All rights reserved.

import copy
import getpass
import json
import os
from collections import OrderedDict

import numpy as np
import torch
from anchor_type_split import (
    ANCHOR_TYPE_CLASSIFY_FUNC,
    ANCHORSET_124_TYPE_DICT,
    ANCHORSET_139_TYPE_DICT,
)
from color_table import (
    gen_drivable_map_color_table,
    gen_roadmap_with_vl_color_table,
)
from horizon_plugin_pytorch.quantization import March
from processed_dataset import (
    HAT_UNITTEST_PREFIX,
    SD_DEMO_DATASET,
    VW_DEMO_DATASET,
)

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.core.traj_pred_typing import ANCHOR_TYPE_BLOCKLIST
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
march = March.BAYES  # March.BAYES for bup2.5 or "bayes1" for bpu2.2

# Task configurations
task_name = "vectornet_before_odd"
anchor_set = "vw_demo_139"  # "vw_demo_139", "pilot_203"
qat_ckpt_for_int_infer = "qat-checkpoint-best.pth.tar"
float_ckpt_for_int_infer = "float-checkpoint-best.pth.tar"
predictor_type = "multipath"

# Dataset configurations
project = "sd_demo"
dataset_name = "multi-task_perrec"
train_pkl_name = (
    "densetnt_small_train"  # "full_scene_SH_train_pkl" for default
)
val_pkl_name = "densetnt_small_test"  # "full_scene_SH_val_pkl" for default

# 小数据集(实验用,需要 dukai.dong 提供的类来加载),代码链接(尚未合入master):
# https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4319/diffs#73c9f5fe4e0c477ed410c0a51e72b10f995a9e54
# train_pkl_name = "densetnt_small_train" # 22w
# val_pkl_name = "densetnt_small_test" # 3w

# Very import configurations!!!
local_train = True
is_training = True
if "bucket" in os.listdir("/"):
    local_train = False
use_state_vectors = True  # if add state_vector in the input of model
viz_on_every_epoch = False  # if viz on each epoch, set False in training
blend_traj_by_history = False  # if scaling pred, set True for eval

# Range of prediction, veh[x_max, x_min, y_max, y_min], ped[...], cycli[...]
veh_range = [50, -30, 50, -50]
ped_range = [30, 0, 10, -10]
cyc_range = [50, -30, 20, -20]

# 0.1. Paths.
if local_train:
    device_ids = [0]  # [0,1,2,3]
    # Set your local path here.
    user_name = getpass.getuser()
    J5FSD_BUCKET_PATH = "/horizon-bucket/J5FSD/"
    SD_Algo_BUCKET_PATH = "/horizon-bucket/SD_Algorithm/"
    ckpt_dir = f"/jfs-public/users/{user_name}/results/{task_name}"
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

# Anchors.
if anchor_set == "basic_124":
    anchor_file = os.path.join(
        SD_Algo_BUCKET_PATH,
        HAT_UNITTEST_PREFIX,
        "anchors/veh_anchor_add_ped_anchors.pkl",
    )
    anchor_num = 124
    anchor_type_cls_dict = ANCHORSET_124_TYPE_DICT
elif anchor_set == "vw_demo_139":
    anchor_file = os.path.join(
        SD_Algo_BUCKET_PATH,
        HAT_UNITTEST_PREFIX,
        "anchors/veh_anchor_add_ped_anchors_139.pkl",
    )
    anchor_num = 139
    anchor_type_cls_dict = ANCHORSET_139_TYPE_DICT
elif anchor_set == "pilot_203":
    anchor_file = os.path.join(
        SD_Algo_BUCKET_PATH,
        "11_perception_prediction/02_user/zepei.sun",
        "anchors_pilot_203_new.pkl",
    )
    anchor_num = 203
    anchor_type_cls_dict = ANCHORSET_139_TYPE_DICT
else:
    raise ValueError(f"Undefined anchor set {anchor_set}")
anchor_type_version = "classical"
anchor_type_dict = anchor_type_cls_dict[anchor_type_version]
anchor_type_blocklist = ANCHOR_TYPE_BLOCKLIST[anchor_type_version]

color_table = gen_roadmap_with_vl_color_table()
drivable_color_table = gen_drivable_map_color_table()

# 0.2. Parameters.
# -- Parameters about the training.
float_epoch, quanti_epoch = 30, 5
train_num_workers, val_num_workers = 8, 4
if local_train:
    batch_size_per_gpu = 32
else:
    batch_size_per_gpu = 64
float_lr, quanti_lr = 0.001, 0.00001  # default 0.001, 0.00001
weight_decay, quanti_weight_decay = 0.001, 0.001  # default 0.001, 0.0001
warmup_epoch = 1
log_freq = 100

# -- Parameters about the trajectory prediction task.
pkl_fps = 2
seq_length = 8 * pkl_fps
seq_period = int(pkl_fps / 2)
sample_step = 1
context_frames = 4
gt_traj_len = 12
target_freq = 10
traj_len = 12  # int(gt_traj_len * target_freq / pkl_fps)

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

static_thr = [0.1, 0.2, 0.2]

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
        type="PredTrajGenSeqCenter",
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
]

common_transforms_2 = [
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
]


ds_wrap_func = None
if use_extend_state_vectors:
    if not local_train:
        ds_wrap_func = [dataset["lmdb_path_replace_func"]]
    common_transforms_2 += [
        dict(
            type="SampleNaviTrajAnchor",
            bev_origin_x=bev_origin_x,
            bev_origin_y=bev_origin_y,
            img_resolution=img_resolution,
            num_anchors=anchor_num,
            traj_len=traj_len,
            navi_file_dir=J5FSD_BUCKET_PATH,
            navi_info_path_mapping={},
            navi_file_path_func=map_path_func,
            anchor_classify_func=ANCHOR_TYPE_CLASSIFY_FUNC[
                anchor_type_version
            ],
            reverse=True,
        ),
        dict(
            type="GetNavinetmapInfo",
            use_extend_state_vectors=use_extend_state_vectors,
            norm_scale=5,
        ),
    ]
train_transforms += common_transforms_2
val_transforms += common_transforms_2

val_transforms.append(
    dict(
        type="GetBEVLocalMapByTimestamp",
        image_dir=bucket_dir,
        image_suffix=".png",
        map_height=img_h,
        map_width=img_w,
        data_token2path_mapping=data_token2path_mapping,
        map_path_func=map_path_func,
    )
)
# TODO(zifan.li): 后续数据生产将不提供栅格化地图，
# 未来GetBEVLocalMapByTimestamp将逐渐弃用，
# 推荐算法同学使用基于向量化地图的可视化
# val_transforms.append(
#     dict(
#         type="GetBevVectorizedMap",
#         bev_origin_x=bev_origin_x,
#         bev_origin_y=bev_origin_y,
#         img_resolution=img_resolution,
#     )
# )
val_transforms.append(
    dict(
        type="GetObstacleSafeArea",
        traj_len=traj_len,
        expand_base=4,
        expand_ratio=0.2,
        use_his_traj=False,
        reverse=coords_reverse,
        img_params=[bev_origin_x, bev_origin_y, img_resolution],
    )
)

if not local_train and "lmdb_path_replace_func" in dataset:
    ds_wrap_func = [dataset["lmdb_path_replace_func"]]  # just cluster train
else:
    ds_wrap_func = None


train_dataloader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="PredTrajPickledTdtDataset",
        pkl_path=train_dataset_pkl_path,
        seq_length=seq_length,
        seq_period=seq_period,
        sample_step=sample_step,
        context_frames=context_frames,
        transforms=train_transforms,
        wrap_func=ds_wrap_func,
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
        type="PredTrajPickledTdtDataset",
        pkl_path=val_dataset_pkl_path,
        seq_length=seq_length,
        seq_period=seq_period,
        sample_step=sample_step,
        context_frames=context_frames,
        transforms=val_transforms,
        wrap_func=ds_wrap_func,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    collate_fn=collate_vectornet_viz,
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=val_num_workers,
)

enable_scale_trils = True
# 2. Setup Model.
if predictor_type == "mlp":
    model_heads = OrderedDict(
        traj_head=dict(
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
    ckpt_use_head_names = ["traj_head"]
elif predictor_type == "multipath":
    in_channels = dict(
        graph_feat=mlp_input_channels,
    )
    if use_state_vectors:
        in_channels["state_vector_feat"] = num_state_vectors
    model_heads = OrderedDict(
        traj_head=dict(
            type="BasicAnchorBasedDecoder",
            in_channels=in_channels,
            anchor_cfg=dict(
                anchor_file=anchor_file,
                anchor_method="kmeans",
                anchor_num=anchor_num,
            ),
            n_hidden_layers=[128, 256],
            use_momentum=True,
            log_std_clamp_min=-2,
            log_std_clamp_max=2,
            is_int_infer_model=False,
            loss=dict(
                type="SoftMultipathLoss",
                head_weight=1,
                reg_loss_scale=1,
                is_soft_cls=True,
                is_soft_reg=True,
                num_soft_trajs=3,
                soft_mode="by_l2",
                soft_trajs_threshold=20,  # if full_scense_data + 139 anchors
            ),
            post_process=dict(
                type="MultipathPostProcessor",
                filter_reverse=True,
                use_traj_nms=use_nms,
                nms_params=dict(
                    nms_threshold=0.05,
                    nms_prob_threshold=0.01,
                    nms_num_trajs=10,
                    change_probs=False,
                ),
                disable_anchor_list=[51, 34, 38, 56],
                anchor_type_dict=anchor_type_dict,
                anchor_type_blocklist=anchor_type_blocklist,
                blend_traj_by_history=blend_traj_by_history,
            ),
        ),
    )
    traj_sample_ratio = 1
    enable_itp_gt = False
    ckpt_use_head_names = ["traj_head"]
else:
    raise ValueError(f"Undefined prediction decoder {predictor_type}")

head_name_dict = dict(
    traj_head="traj_head",
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
        num_hidden_units=subgraph_hidden_size,
        num_attn_hidden_units=globalgraph_hidden_size,
        num_sub_graph_layers=num_sub_graph_layers,
        num_attention_heads=num_attn_head,
        use_out_fc=use_out_fc,
        fc_out_channels=mlp_input_channels,
    ),
    necks=necks,
    heads=model_heads,
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
    type="TrajPredMetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix="Validation_" + task_name,
    display_head_names=ckpt_use_head_names,
    traj_display_k_values=[1, 5],
    display_cls_dict={
        "vehicle": 0,
        "pedestrain": 1,
        "cyclist": 2,
    },
    save_metric_path=ckpt_dir,
    support_behav_metric=False,
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
elif predictor_type == "multipath":
    loss_str = "^.*loss.*"
    metrics = [
        dict(
            type="TrajPredMetric",
            head_names=head_name_dict,
            k_values=[1, 5],
            name=ckpt_use_head_names[0],
        ),
    ]
    metric_updater["metrics"].insert(
        0,
        dict(type="LossShow", name="cls_loss"),
    )
    per_metric_patterns = [  # corresponding to metrics
        dict(label_pattern=None, pred_pattern="^.*cls_loss$"),
        dict(label_pattern=None, pred_pattern="^.*reg_loss$"),
    ]
    metric_updater["metric_update_func"] = update_metric_using_regex(
        per_metric_patterns
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

# tensorboard
# job on aidi platform will have TENSORBOARD_LOG_PATH env,
# where the saved tb can be shown through aidi web UI
tb_cls_loss_callback = dict(
    type="TensorBoard",
    save_dir=os.path.join(tensorboard_dir, "float", "loss"),
    loss_name_reg="^.*cls_loss$",
    update_freq=100,
    update_by="step",
)

tb_reg_loss_callback = dict(
    type="TensorBoard",
    save_dir=os.path.join(tensorboard_dir, "float", "loss"),
    loss_name_reg="^.*reg_loss$",
    update_freq=100,
    update_by="step",
)

# 5. Trainer and solver.
# -- 1. float.

float_viz_callback = dict(
    type="PredTrajViz",
    head_names=head_name_dict,
    height=img_h,
    width=img_w,
    resolution=img_resolution,
    map_origin=[bev_origin_x, bev_origin_y],
    video_save_path=viz_dir,
    video_name="float_best_viz",
    video_fps=video_fps,
    viz_on_every_epoch=viz_on_every_epoch,
    max_epoch=float_epoch,
    ego_track_id=-BaseTrajDataset.ANSWER,
    num_traj_to_viz=num_traj_to_viz,
    num_workers=val_num_workers,
    use_behav_head=False,
)
float_val_callback = copy.deepcopy(val_callback)
# float_val_callback["callbacks"].append(float_viz_callback)


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
            monitor_metric_key=f"head_{ckpt_use_head_names[0]}_min_ade_1",
        ),
        tb_cls_loss_callback,
        tb_reg_loss_callback,
    ],
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
        float_viz_callback,
    ],
    log_interval=50,
)

# -- 1.5. calibration.(Can be skipped due to little effect.)
# Note: The transforms of the dataset during calibration can be
# consistent with that during training or validation, or customized.
# Default used `val_batch_processor`.
# calibration_data_loader = copy.deepcopy(train_dataloader)
# calibration_data_loader.pop("sampler")
# calibration_batch_processor = copy.deepcopy(val_batch_processor)
# calibration_val_callback = copy.deepcopy(val_callback)
# calibration_trainer = dict(
#     type="Calibrator",
#     model=copy.deepcopy(model),
#     model_convert_pipeline=dict(
#         type="ModelConvertPipeline",
#         qat_mode="fuse_bn",
#         converters=[
#             dict(
#                 type="LoadCheckpoint",
#                 checkpoint_path=os.path.join(
#                     ckpt_dir, "float-checkpoint-best.pth.tar"
#                 ),
#             ),
#             dict(type="Float2Calibration"),
#         ],
#     ),
#     data_loader=calibration_data_loader,
#     batch_processor=calibration_batch_processor,
#     num_steps=500,
#     callbacks=[
#         stats_callback,
#         calibration_val_callback,
#         dict(
#             type="Checkpoint",
#             save_dir=ckpt_dir,
#             name_prefix="calibrator-",
#             strict_match=True,
#             mode="min",
#             monitor_metric_key=f"head_{ckpt_use_head_name}_min_ade_1",
#         ),
#     ],
#     val_metrics=metrics,
# )

# -- 2. qat.
qat_val_callback = copy.deepcopy(val_callback)
qat_viz_callback = copy.deepcopy(float_viz_callback)
qat_viz_callback["video_name"] = "qat_best_viz"
qat_viz_callback["max_epoch"] = quanti_epoch
qat_val_callback["callbacks"] = [val_metric_updater]
qat_val_callback["callbacks"].append(qat_viz_callback)

qat_tb_save_dir = os.path.join(tensorboard_dir, "qat", "loss")
qat_tb_cls_loss_callback = copy.deepcopy(tb_cls_loss_callback)
qat_tb_cls_loss_callback["save_dir"] = qat_tb_save_dir
qat_tb_reg_loss_callback = copy.deepcopy(tb_reg_loss_callback)
qat_tb_reg_loss_callback["save_dir"] = qat_tb_save_dir

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
            strict_match=True,
            mode="min",
            monitor_metric_key=f"head_{ckpt_use_head_names[0]}_min_ade_1",
        ),
        qat_tb_cls_loss_callback,
        qat_tb_reg_loss_callback,
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
    log_interval=50,
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
    traj_len=12,
    bounce_thr=val_bounce_thr,
    anchor_num=anchor_num,
    use_anchor=1,
    norm_scale=1.0,
    use_traj_diff=0,
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
    per_tensor_desc=[
        # num of desc == num of pred output tensors (one tensor per out stride)
        json.dumps(
            dict(
                task=f"{task_name}_head_traj_pred_anchor_prob",
                **params_desc,
            )
        ),
        json.dumps(
            dict(
                task=f"{task_name}_head_traj_pred_anchor_mean_var",
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
    head["is_int_infer_model"] = True
    head["loss"] = None
    head["post_process"] = None

int_val_callback = copy.deepcopy(val_callback)
int_val_callback["callbacks"] = []
int_val_callback["data_loader"] = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="PredTrajPickledTdtDataset",
        pkl_path=val_dataset_pkl_path,
        seq_length=seq_length,
        seq_period=seq_period,
        sample_step=sample_step,
        context_frames=context_frames,
        transforms=val_transforms,
        wrap_func=ds_wrap_func,
    ),
    collate_fn=collate_vectornet,
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
    deploy_inputs["state_vectors"] = torch.randn(
        (hdm_max_num_obs, num_state_vectors, 1, 1)
    )

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix="int-",
    strict_match=True,
    mode="min",
    monitor_metric_key=f"head_{ckpt_use_head_names[0]}_min_ade_1",
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
