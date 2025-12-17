# Copyright (c) Horizon Robotics. All rights reserved.

import copy
import json
import os
from collections import OrderedDict

import hbdk
import numpy as np
import torch
from configs.traj_pred.anchor_type_split import (
    ANCHOR_TYPE_CLASSIFY_FUNC,
    ANCHORSET_124_TYPE_DICT,
    ANCHORSET_139_TYPE_DICT,
)
from configs.traj_pred.color_table import (
    gen_drivable_map_color_table,
    gen_roadmap_with_vl_color_table,
)
from configs.traj_pred.processed_dataset import (
    HAT_UNITTEST_PREFIX,
    SD_DEMO_DATASET,
    VW_DEMO_DATASET,
)
from horizon_plugin_pytorch.quantization import March

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.core.traj_pred_typing import ANCHOR_TYPE, ANCHOR_TYPE_BLOCKLIST
from hat.data.collates.traj_pred_collates import (
    collate_multipath,
    collate_multipath_viz,
)
from hat.data.datasets.traj_pred_dataset import BaseTrajDataset
from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.models.backbones.vargnetv2 import get_vargnetv2_stride2channels
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2

task_name = "multipath_lite_refact"
cudnn_benchmark = False
seed = None
log_rank_zero_only = True
march = March.BAYES

# *****HD map dataset*****
project = "vw_demo"
dataset_name = "Vision_Navi"
train_pkl_name = "basic_train_pkl"
val_pkl_name = "basic_test_pkl"

anchor_set = "basic_124"
qat_ckpt_for_int_infer = "qat-checkpoint-best.pth.tar"
float_ckpt_for_int_infer = "float-checkpoint-best.pth.tar"

local_train = True
viz_on_every_epoch = False

# 0.1. Paths.
if local_train:
    device_ids = [3]
    batch_size_per_gpu = 16
    train_num_workers, val_num_workers = 8, 4
    float_epoch, quanti_epoch = 20, 5
    # Set your local path here.
    J5FSD_BUCKET_PATH = "/horizon-bucket/J5FSD/"
    J5FSD_2_BUCKET_PATH = "/horizon-bucket/J5FSD_2/"
    ckpt_dir = f"/jfs-public/adas/shengzhe.dai/results/{task_name}"
    viz_dir = f"/jfs-public/adas/shengzhe.dai/results/{task_name}/viz"
else:
    device_ids = [0, 1, 2, 3]
    batch_size_per_gpu = 8
    train_num_workers, val_num_workers = 8, 4
    float_epoch, quanti_epoch = 15, 5
    J5FSD_BUCKET_PATH = "/bucket/input/J5FSD/"
    J5FSD_2_BUCKET_PATH = "/bucket/input/J5FSD_2/"
    ckpt_dir = "/job_data/models/checkpoint"
    viz_dir = "/job_data/models/viz"

os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(viz_dir, exist_ok=True)
bucket_dir_dict = {"J5FSD": J5FSD_BUCKET_PATH, "J5FSD_2": J5FSD_2_BUCKET_PATH}

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
        J5FSD_BUCKET_PATH,
        HAT_UNITTEST_PREFIX,
        "anchors/veh_anchor_add_ped_anchors.pkl",
    )
    anchor_num = 124
    anchor_type_cls_dict = ANCHORSET_124_TYPE_DICT
elif anchor_set == "vw_demo_139":
    anchor_file = os.path.join(
        J5FSD_BUCKET_PATH,
        HAT_UNITTEST_PREFIX,
        "anchors/veh_anchor_add_ped_anchors_139.pkl",
    )
    anchor_num = 139
    anchor_type_cls_dict = ANCHORSET_139_TYPE_DICT
else:
    raise ValueError(f"Undefined anchor set {anchor_set}")


vargnet_pretrain_backbone = {
    0.5: os.path.join(
        J5FSD_BUCKET_PATH,
        HAT_UNITTEST_PREFIX,
        "pretrain_backbones/vargnetv2_05_headfactor2.pth.tar",
    ),
    0.75: os.path.join(
        J5FSD_BUCKET_PATH,
        HAT_UNITTEST_PREFIX,
        "pretrain_backbones/vargnetv2_075-float.pth.tar",
    ),
    1: os.path.join(
        J5FSD_BUCKET_PATH,
        HAT_UNITTEST_PREFIX,
        "pretrain_backbones/xj5_release_models.tar",
    ),
}


def update_state_dict_multipath(state_dict):
    global input_channels
    old_params = state_dict["backbone.mod1.0.weight"]
    if old_params.shape[1] == 3:
        mean_params = torch.mean(old_params, dim=1, keepdim=True)
        mean_params = mean_params.repeat(1, input_channels - 3, 1, 1)
        new_params = torch.cat([old_params, mean_params], dim=1)
        state_dict["backbone.mod1.0.weight"] = new_params
    return state_dict


color_table = gen_roadmap_with_vl_color_table()
drivable_color_table = gen_drivable_map_color_table()

# 0.2. Parameters.
batch_size_per_gpu = 16
train_num_workers, val_num_workers = 8, 4
float_lr, quanti_lr = 0.001, 0.00001
float_epoch, quanti_epoch = 20, 5
weight_decay = 0.001
warmup_epoch = 1
log_freq = 100
pkl_fps = 2
video_fps = 10
pkl_freq_ratio = int(pkl_fps / 2)
seq_length = 8 * pkl_fps
seq_period = 1
sample_step = pkl_freq_ratio
context_frames = 4 * pkl_freq_ratio
traj_len = 12

use_drivable_area = True
if use_drivable_area:
    data_shape = (10, 512, 512)
else:
    data_shape = (7, 512, 512)
input_channels = data_shape[0]
alpha = 0.5

bev_origin_x, bev_origin_y, img_resolution = 72.4, 51.2, 0.2
coords_reverse = True  # If use bev coordinates, this must be true,
img_h, img_w = data_shape[1:3]
x_ratio = bev_origin_x / img_resolution / img_h
y_ratio = bev_origin_y / img_resolution / img_w
train_bounce_thr = np.pi / 8
val_bounce_thr = np.pi / 3
veh_type_id = 0
ped_cyc_type_id = [3, 1]
roi_input_size = 16
roi_output_size = 8
out_stride = 32
roi_spatial_scale = int(img_h / roi_input_size)
leaving_mode = "all"
yaw_select_type = [5, 5, 5]
detect_peds_by_shape = True
ped_shape_thr = 1
use_state_vectors = True
use_instant_state_vectors = True
if_clip_state_vectors = True
use_nms = False
use_data_augmentation = False
viz_on_every_epoch = False
anchor_type_version = "poor"
anchor_type = ANCHOR_TYPE[anchor_type_version]
anchor_type_dict = ANCHORSET_124_TYPE_DICT[anchor_type_version]
anchor_type_blocklist = ANCHOR_TYPE_BLOCKLIST[anchor_type_version]
num_anchors = len(anchor_type_dict.keys())
num_anchor_types = len(anchor_type)
anchor_mode = "fuse"

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
    # dict(type="LateralSmoothing"),
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
        type="GetSeqDataFrameMask",
        freq_ratio=pkl_freq_ratio,
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
        enable_incomplete_gts=True,
        is_training=True,
    ),
    dict(
        type="GenHighFreqTraj",
        get_sampled_frames=False,
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
        enable_incomplete_gts=True,
        is_training=False,
    ),
    dict(
        type="GenHighFreqTraj",
        get_sampled_frames=False,
        enable_incomplete_gts=True,
        is_training=False,
    ),
]

common_transforms_2 = [
    dict(
        type="OccupancyMapRender",
        map_height=img_h,
        map_width=img_w,
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
        type="GetBEVLocalMapByTimestamp",
        image_dir=bucket_dir,
        image_suffix="_drivable.png",
        map_height=img_h,
        map_width=img_w,
        data_token2path_mapping=data_token2path_mapping,
        map_path_func=map_path_func,
        item_key="drivable_road_map",
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
        assign_trajs_for_filtered_obs=False,
        reverse=coords_reverse,
        traj_len=traj_len,
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
    dict(
        type="GetBestAnchors",
        anchor_cfg=dict(
            anchor_file=anchor_file,
            anchor_method="kmeans",
            anchor_num=anchor_num,
        ),
        anchor_type=anchor_type,
        anchor_type_dict=anchor_type_dict,
    ),
]
train_transforms += common_transforms_2
val_transforms += common_transforms_2

if not local_train:
    ds_wrap_func = [dataset["lmdb_path_replace_func"]]

train_transforms.append(
    dict(
        type="SampleNaviTrajAnchor",
        bev_origin_x=bev_origin_x,
        bev_origin_y=bev_origin_y,
        img_resolution=img_resolution,
        num_anchors=num_anchor_types,
        traj_len=traj_len,
        navi_file_dir=J5FSD_BUCKET_PATH,
        navi_info_path_mapping={},
        navi_file_path_func=map_path_func,
        anchor_classify_func=ANCHOR_TYPE_CLASSIFY_FUNC[anchor_type_version],
        anchor_type_version=anchor_type_version,
        reverse=True,
        velo_scales=[0.5, 1, 1.5],
    )
)
val_transforms.append(
    dict(
        type="SampleNaviTrajAnchor",
        bev_origin_x=bev_origin_x,
        bev_origin_y=bev_origin_y,
        img_resolution=img_resolution,
        num_anchors=num_anchor_types,
        traj_len=traj_len,
        navi_file_dir=J5FSD_BUCKET_PATH,
        navi_info_path_mapping={},
        navi_file_path_func=map_path_func,
        anchor_classify_func=ANCHOR_TYPE_CLASSIFY_FUNC[anchor_type_version],
        anchor_type_version=anchor_type_version,
        reverse=True,
        velo_scales=[1],
    )
)


if not local_train and "lmdb_path_replace_func" in dataset:
    ds_wrap_func = [dataset["lmdb_path_replace_func"]]  # just cluster train
else:
    ds_wrap_func = None


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
        wrap_func=ds_wrap_func,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    collate_fn=collate_multipath,
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
        wrap_func=ds_wrap_func,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    collate_fn=collate_multipath_viz,
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=val_num_workers,
)

# 2. Setup Model.
bn_kwargs = dict(eps=1e-5, momentum=0.1)
model_necks = OrderedDict(
    multipath_neck=dict(
        type="MultiPathNeck",
        in_strides=[2, 4, 8, 16, 32],
        out_strides=[out_stride],
        stride2channels=get_vargnetv2_stride2channels(alpha),
        roi_input_patch=[roi_input_size, roi_input_size],
        roi_output_patch=[roi_output_size, roi_output_size],
        conv_channels=[16, 16, 16, 16],
        use_depthwise_as_avg=True,
        is_int_infer_model=False,
    ),
    anc_enc_neck=dict(
        type="AnchorEncodeNeck",
        anc_conv_channels=[1024, 128, 16],
        num_anchors=num_anchors,
        traj_len=traj_len,
        anchor_mode=anchor_mode,
        is_int_infer_model=False,
    ),
)
model_heads = OrderedDict(
    traj_head=dict(
        type="BasicAnchorBasedDecoder",
        in_channels=dict(
            rroi_feat=16,
            state_vector_feat=3,
            anchor_encode_feat=16,
        ),
        anchor_cfg=dict(
            anchor_key=f"{anchor_mode}_anchors",
            anchor_num=num_anchors,
            traj_len=traj_len,
        ),
        n_hidden_layers=[1024, 1024],
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
            anchor_type_version=anchor_type_version,
            anchor_type_dict=anchor_type_dict,
            anchor_type_blocklist=anchor_type_blocklist,
            is_multipath_lite=True,
        ),
    ),
)
ckpt_use_head_name = "traj_head"
head_name_dict = dict(
    traj_head="traj_head",
)

model = dict(
    type="MultipathV2",
    data_shape=data_shape,
    backbone=dict(
        type="VargNetV2",
        num_classes=1000,
        bn_kwargs=bn_kwargs,
        alpha=alpha,
        group_base=8,
        factor=2,
        bias=True,
        disable_quanti_input=True,
        include_top=False,
        flat_output=True,
        input_channels=input_channels,
        head_factor=2,
    ),
    necks=model_necks,
    heads=model_heads,
    table=color_table,
    map_augmentation=True,
    drivable_area_table=drivable_color_table,
    use_drivable_area=use_drivable_area,
)

# 3. Setup metrics.
# --1. Training metric (loss) updater.
metric_updater = dict(
    type="MetricUpdater",
    metrics=[
        dict(type="LossShow", name="cls_loss"),
        dict(type="LossShow", name="reg_loss"),
    ],
    metric_update_func=update_metric_using_regex(
        per_metric_patterns=[  # corresponding to metrics
            dict(label_pattern=None, pred_pattern="^.*cls_loss$"),
            dict(label_pattern=None, pred_pattern="^.*reg_loss$"),
        ]
    ),
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
)


# --2. Validation metric updater.
def update_metric(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


# -- 2.1. Setup single or multiple metrics. Multiple metrics is used
#         for evaluate multiple heads or multiple classes respectively.
metrics = [
    dict(
        type="TrajPredMetric",
        head_names=head_name_dict,
        k_values=[1, 5, 10, 64],
        name=ckpt_use_head_name,
    )
]

# -- 2.2. a validation metric updater, if you setup multiple metrics,
#         split their respective model outputs in "update_metric" func.
val_metric_updater = dict(
    type="TrajPredMetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix="Validation_" + task_name,
    display_head_name=ckpt_use_head_name,
    display_k_values=[1, 5],
    display_cls_dict={
        "vehicle": 0,
        "pedestrain": 1,
        "cyclist": 2,
    },
    save_metric_path=ckpt_dir,
)

# 4. Setup callbacks.
batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    loss_collector=collect_loss_by_regex("^.*loss.*"),
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
float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=copy.deepcopy(model),
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=vargnet_pretrain_backbone[alpha],
                state_dict_update_func=update_state_dict_multipath,
                allow_miss=True,
                ignore_extra=True,
            ),
        ],
    ),
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
        val_callback,
        dict(
            type="Checkpoint",
            save_dir=ckpt_dir,
            name_prefix="float-",
            strict_match=True,
            mode="min",
            monitor_metric_key=f"head_{ckpt_use_head_name}_min_ade_5",
        ),
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
    ],
    log_interval=50,
)

# -- 2. qat.
qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=copy.deepcopy(model),
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
            dict(type="Float2QAT"),
        ],
    ),
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
        metric_updater,
        lr_update_callback,
        stats_callback,
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
    train_metrics=metrics,
    val_metrics=metrics,
)


qat_val_callback = copy.deepcopy(val_callback)
qat_viz_callback = dict(
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
    max_epoch=quanti_epoch,
    ego_track_id=-BaseTrajDataset.ANSWER,
    num_traj_to_viz=5,
    num_workers=val_num_workers,
    freq_ratio=pkl_freq_ratio,
)
qat_val_callback["callbacks"] = [val_metric_updater, qat_viz_callback]

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
def collate_multipath_int_infer(batch):
    global roi_output_size
    global anchor_mode
    batch_data = collate_multipath(batch)
    batch_data[f"{anchor_mode}_anchors_diff_x"] = batch_data[
        f"{anchor_mode}_anchors_diff_x"
    ].reshape((-1, num_anchors * traj_len, 1, 1))
    batch_data[f"{anchor_mode}_anchors_diff_y"] = batch_data[
        f"{anchor_mode}_anchors_diff_y"
    ].reshape((-1, num_anchors * traj_len, 1, 1))
    return batch_data


# ---- add description
params_desc = dict(
    context_frames=context_frames // pkl_freq_ratio,
    image_h=data_shape[1],
    image_w=data_shape[2],
    resolution=img_resolution,
    BEV_origin_x=bev_origin_x,
    BEV_origin_y=bev_origin_y,
    warp_resolution=img_resolution,
    src_length=roi_input_size,
    dst_length=roi_output_size,
    filter_dis=50,
    num_frame_thr=2,
    static_thr=1,
    ped_static_thr=0.2,
    veh_select_yaw_type=yaw_select_type[0],
    ped_select_yaw_type=yaw_select_type[1],
    cyc_select_yaw_type=yaw_select_type[2],
    use_state_vectors=1 if use_state_vectors else 0,
    ego_length=4,
    ego_width=1.5,
    ego_offset=1.35,
    traj_len=12,
    top_k=5,
    bounce_thr=val_bounce_thr,
    anchor_num=num_anchors,
    use_drivable_area=1 if use_drivable_area else 0,
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

int_val_callback = copy.deepcopy(val_callback)
int_val_callback["callbacks"] = []
int_val_callback["data_loader"] = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="PickledTdtDataset",
        pkl_path=val_dataset_pkl_path,
        seq_length=seq_length,
        seq_period=seq_period,
        sample_step=sample_step,
        context_frames=context_frames,
        transforms=val_transforms,
        wrap_func=ds_wrap_func,
    ),
    collate_fn=collate_multipath_int_infer,
    batch_size=1,
    shuffle=False,
    num_workers=val_num_workers,
)

num_obs_in_hbm = 32
deploy_inputs = dict(
    road_map=torch.randn((1, 1, img_h, img_w)),
    rendered_obs=torch.randn((1, context_frames, img_h, img_w)),
    state_vectors=torch.randn((num_obs_in_hbm, 3, 1, 1)),
    img_homographys=hbdk.torch_script.tools.placeholder(
        (num_obs_in_hbm, roi_output_size, roi_output_size, 2),
        torch_native=True,
    ),
)
deploy_inputs[f"{anchor_mode}_anchors_diff_x"] = torch.randn(
    (num_obs_in_hbm, num_anchors * traj_len, 1, 1)
)
deploy_inputs[f"{anchor_mode}_anchors_diff_y"] = torch.randn(
    (num_obs_in_hbm, num_anchors * traj_len, 1, 1)
)
if use_drivable_area:
    deploy_inputs["drivable_road_map"] = torch.randn((1, 1, img_h, img_w))

int_trainer = dict(
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
        dict(
            type="Checkpoint",
            save_dir=ckpt_dir,
            name_prefix="int-",
            strict_match=True,
            mode="min",
            monitor_metric_key=f"head_{ckpt_use_head_name}_min_ade_5",
        ),
        dict(
            type="SaveTraced",
            save_dir=ckpt_dir,
            trace_inputs=deploy_inputs,
        ),
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
