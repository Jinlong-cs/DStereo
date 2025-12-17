# Copyright (c) Horizon Robotics. All rights reserved.

import copy
import json
import os
from collections import OrderedDict

import numpy as np
import torch
import yaml
from horizon_plugin_pytorch.quantization import March
from pnc_dataset import P3C_DATASET

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.callbacks.planning_viz import gen_roadmap_color_table
from hat.data.collates.planning_collates import collate_p3c_plan
from hat.data.datasets.traj_pred_dataset import BaseTrajDataset
from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.utils import Config

cudnn_benchmark = False
seed = None
log_rank_zero_only = True
march = March.BAYES
cfg_dir = os.path.dirname(__file__)

# Very import configurations!!!
pkl_freq_ratio = 1
seq_length = 16
seq_period = 1
sample_step = pkl_freq_ratio
context_frames = 4 * pkl_freq_ratio
traj_len = 12
encode_ego_motion = True

# 0.0. If the user wants to start the training manually, please change the
# following parameters.

# stage = "float"  # float | qat | int_infer | predict
task_name = "planning_vin_hat_p3c_16w_float_qat_int_floatVal"

# k8s setiings, for aidi platform
job_list = [
    # train
    "python3 tools/train.py --stage %s --config planning/configs/%s"
    % ("float", os.path.basename(__file__)),
    # "sleep 60m"
    # "python3 tools/train.py --stage %s --config planning/configs/%s"
    # % ("qat", os.path.basename(__file__)),
    # "python3 tools/train.py --stage %s --config planning/configs/%s"
    # % ("int_infer", os.path.basename(__file__)),
    # compile
    # "python3 tools/deploy/compile_perf.py --config planning/configs/%s",
    # validation
    # "python3 tools/predict.py --stage %s --config planning/configs/%s"
    # % ("float", os.path.basename(__file__)),
    # "python3 tools/predict.py --stage %s --config planning/configs/%s"
    # % ("qat", os.path.basename(__file__)),
    # "python3 tools/predict.py --stage %s --config planning/configs/%s"
    # % ("int_infer", os.path.basename(__file__)),
    "mkdir -p /job_data/code & \
              cp -r /running_package/code_package/* /job_data/code/",
]
k8s_config_file = os.path.join(cfg_dir, "k8s_config.py")
base_k8s_config = Config.fromfile(k8s_config_file)
k8s_config = dict()
k8s_config.update(base_k8s_config._cfg_dict)
k8s_config["job_list"] = job_list
k8s_config["job_name"] = task_name

local_train = not os.path.exists("/running_package")

# 0.1. Paths.
if local_train:
    assert (
        task_name is not None
    ), "You are training in local env, task name must be assigned."
    device_ids = [0, 1]
    # Set your local path here.
    SD_ALGO_BUCKET_PATH = "/horizon-bucket/SD_Algorithm/"
    ckpt_dir = f"/home/users/zhiqiang03.zhang/0_code/HAT/outputs/{task_name}"
else:
    device_ids = [i for i in range(k8s_config["num_gpus_per_machine"])]  # noqa
    SD_ALGO_BUCKET_PATH = "/bucket/input/SD_Algorithm/"
    ckpt_dir = "/job_data/models/checkpoint"

os.makedirs(ckpt_dir, exist_ok=True)
bucket_dir_dict = {
    "SD_Algorithm": SD_ALGO_BUCKET_PATH,
}

# Datasets.
dataset = P3C_DATASET["p3c_22w"]
bucket_dir = bucket_dir_dict[dataset["bucket"]]
map_path_func = dataset["map_path_func"]

file_token_path = os.path.join(bucket_dir, dataset["path_token_file"])
with open(file_token_path, "r") as yaml_file:
    data_token2path_mapping = yaml.load(yaml_file, Loader=yaml.FullLoader)
train_dataset_pkl_path = os.path.join(
    bucket_dir, dataset["obs_dir"], dataset["train"]
)
val_dataset_pkl_path = [
    os.path.join(bucket_dir, dataset["obs_dir"], path)
    for path in dataset["val"]
]
test_dataset_pkl_path = [
    os.path.join(bucket_dir, dataset["obs_dir"], path)
    for path in dataset["test"]
]


# 0.2. Parameters.
batch_size_per_gpu = 32
train_num_workers, val_num_workers = 8, 8
float_lr, quanti_lr = 0.00002, 0.00001
float_epoch, quanti_epoch = 25, 5
weight_decay = 0.0001
warmup_epoch = 1
log_freq = 20
veh_type_id = 1
ped_cyc_type_id = [2, 18]
ped_shape_thr = 1

data_shape = (7, 512, 512)
bev_origin_x, bev_origin_y, img_resolution = 72.4, 51.2, 0.2

grid_dim = 65
grid_horizon = 50
grid_actions = 9
grid_extent = [-51.2, 51.2, -30, 72.4]
grid_initial = [45, 32]
perturb_prob = 0

backbone_alpha = 1.0
out_strides = 4
scene_feat_size = 128
agg_size = [32, 32]
grid_cell_size = [2, 2]
motion_size = 2  # ego_spd, spd_limit


input_channels = data_shape[0]
img_h, img_w = data_shape[1:3]
bev_ego_center = (
    int(bev_origin_x / img_resolution),
    int(bev_origin_y / img_resolution),
)


color_table = gen_roadmap_color_table()


############################################################################
# 1. Setup data loader.
############################################################################
tdt_transforms = [
    dict(
        type="GenSeqCenter",
        anchor_frame="last_context",
        context_frames=context_frames,
        augmentation=False,
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
        reverse=True,
    ),
    dict(type="TDTEgoCentricGt"),
    dict(type="GetLcfTimeStamp"),
    dict(
        type="TDTFilterObstacles",
        is_training=True,
        is_multiagent=True,
        ego_track_id=-BaseTrajDataset.ANSWER,
        ped_cyc_type_id=ped_cyc_type_id,
        leaving_mode="all",
        valid_distance=50,
        static_thr=1,  # i.e., 2 m/s
        num_frame_thr=2,
        bounce_thr=np.pi / 8,
        ped_static_thr=0.2,  # i.e, 0.4 m/s
        ped_drift_thr=1.5,  # i.e., 3 m/s
        use_given_lcf_yaw=True,
        use_given_obs_yaw=False,
        classify_by_shape=False,
        ped_shape_thr=ped_shape_thr,
    ),
    dict(
        type="TDTGenStatesAndMask",
        seq_length=seq_length,
        context_frames=context_frames,
        seq_period=seq_period,
        enable_incomplete_gts=True,
    ),
    dict(
        type="TDTOccupancyMapRender",
        map_height=img_h,
        map_width=img_w,
        context_frames=context_frames,
        render_ego=False,
        render_fut=True,
    ),
    dict(
        type="TDTOccupancyMapRender",
        map_height=img_h,
        map_width=img_w,
        context_frames=context_frames,
        render_ego=True,
        render_fut=False,
    ),
    dict(
        type="TDTGetBEVLocalMapByTimestamp",
        image_dir=bucket_dir,
        image_suffix=".png",
        map_height=img_h,
        map_width=img_w,
        data_token2path_mapping=data_token2path_mapping,
        map_path_func=map_path_func,
    ),
    dict(
        type="TDTGetTrajPredObjectsInfo",
        scene_img_height=img_h,
        scene_img_width=img_w,
        ego_track_id=-BaseTrajDataset.ANSWER,
        use_given_obs_yaw=False,
        peds_use_given_obs_yaw=True,
        veh_type_id=veh_type_id,
        ped_cyc_type_id=ped_cyc_type_id,
        detect_peds_by_shape=False,
        ped_shape_thr=ped_shape_thr,
        use_state_vectors=True,
        use_instant_state_vectors=True,
        if_clip_state_vectors=True,
    ),
]

plan_train_transforms = [
    dict(
        type="PlanGaussianRandom",
        std=[1.5, 0.5, np.pi / 10],
    ),
    dict(
        type="PlanEgoMotion",
        context_frames=context_frames,
        seq_length=seq_length,
    ),
    dict(
        type="PlanEgoState",
        grid_dim=grid_dim,
        ego_track_id=-BaseTrajDataset.ANSWER,
    ),
    dict(
        type="PlanPerturbation",
        perturb_prob=perturb_prob,
        ego_center=bev_ego_center,
        map_height=img_h,
        map_width=img_w,
        context_frames=context_frames,
        seq_length=seq_length,
        res=img_resolution,
        min_displacement=3,
        min_acc=-0.3,
        max_acc=0.15,
        min_steer=-0.2,
        max_steer=0.2,
    ),
    dict(
        type="PlanEgoGridGenerate",
        interpolate_freq=10,
        grid_dim=grid_dim,
        horizon=grid_horizon,
        grid_extent=grid_extent,
    ),
    dict(
        type="PlanGridAction",
        grid_dim=grid_dim,
        horizon=grid_horizon,
        action_num=grid_actions,
    ),
]

plan_val_transforms = [
    dict(
        type="PlanEgoMotion",
        context_frames=context_frames,
        seq_length=seq_length,
    ),
    dict(
        type="PlanEgoState",
        grid_dim=grid_dim,
        ego_track_id=-BaseTrajDataset.ANSWER,
    ),
    dict(
        type="PlanEgoGridGenerate",
        interpolate_freq=10,
        grid_dim=grid_dim,
        horizon=grid_horizon,
        grid_extent=grid_extent,
    ),
    dict(
        type="PlanGridAction",
        grid_dim=grid_dim,
        horizon=grid_horizon,
        action_num=grid_actions,
    ),
]

train_transforms = tdt_transforms + plan_train_transforms
val_transforms = tdt_transforms + plan_val_transforms


train_dataloader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="P3CPickledTdtDataset",
        pkl_path=train_dataset_pkl_path,
        seq_length=seq_length,
        seq_period=seq_period,
        sample_step=sample_step,
        context_frames=context_frames,
        transforms=train_transforms,
        wrap_func=None,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    collate_fn=collate_p3c_plan,
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=train_num_workers,
)

val_dataloader = [
    dict(
        type=torch.utils.data.DataLoader,
        dataset=dict(
            type="P3CPickledTdtDataset",
            pkl_path=path,
            seq_length=seq_length,
            seq_period=seq_period,
            sample_step=sample_step,
            context_frames=context_frames,
            transforms=val_transforms,
            wrap_func=None,
        ),
        sampler=dict(type=torch.utils.data.DistributedSampler),
        collate_fn=collate_p3c_plan,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=val_num_workers,
    )
    for path in val_dataset_pkl_path
]

test_dataloader = [
    dict(
        type=torch.utils.data.DataLoader,
        dataset=dict(
            type="P3CPickledTdtDataset",
            pkl_path=path,
            seq_length=seq_length,
            seq_period=seq_period,
            sample_step=sample_step,
            context_frames=context_frames,
            transforms=val_transforms,
            wrap_func=None,
        ),
        sampler=dict(type=torch.utils.data.DistributedSampler),
        collate_fn=collate_p3c_plan,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=val_num_workers,
    )
    for path in test_dataset_pkl_path
]

############################################################################
# 2. Setup Model.
############################################################################
bn_kwargs = dict(eps=1e-5, momentum=0.1)
model_backbone = dict(
    type="PatialResNet",
    input_channels=input_channels,
    bn_kwargs=bn_kwargs,
)

model_neck = dict(
    type="RewardModel",
    in_feat_size=256,
    scene_feat_size=scene_feat_size,
    agg_size=agg_size,
    grid_cell_size=grid_cell_size,
    encode_motion=encode_ego_motion,
    motion_size=motion_size,
)

model_heads = OrderedDict(
    reward_head=dict(
        type="MDPValueDecoder",
        action_len=grid_actions,
        mdp_horizon=grid_horizon,
        initial_state=grid_initial,
        grid_dim=[grid_dim, grid_dim],
        value=-100,
        loss=dict(
            type="MDPValueLoss",
            pi_weight=1.0,
            svf_weight=100.0,
            avg_factor=batch_size_per_gpu,
        ),
    ),
)

model = dict(
    type="ValuePlanNet",
    backbone=model_backbone,
    neck=model_neck,
    heads=model_heads,
    table=color_table,
)


############################################################################
# 3. Setup metrics.
############################################################################
# --1. Training metric (loss) updater.

metric_updater = dict(
    type="MetricUpdater",
    metrics=[
        dict(type="LossShow", name="pi_loss"),
        dict(type="LossShow", name="svf_loss"),
    ],
    metric_update_func=update_metric_using_regex(
        per_metric_patterns=[  # corresponding to metrics
            dict(label_pattern=None, pred_pattern="^.*pi_loss$"),
            dict(label_pattern=None, pred_pattern="^.*svf_loss$"),
        ]
    ),
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="epoch",
)


# --2. Validation metric updater.
# -- 2.1. Setup single or multiple metrics. Multiple metrics is used
#         for evaluate multiple heads or multiple classes respectively.
metrics = [
    dict(
        type="PlanningRewardMetric",
        name=["svf_diff", "cdf_dist"],
    )
]

# -- 2.2. a validation metric updater, if you setup multiple metrics,
#         split their respective model outputs in "update_metric" func.


def update_metric(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


val_metric_updater = dict(
    type="PlanningMetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix="Validation_" + task_name,
    save_metric_path=ckpt_dir,
)

test_metric_updater = dict(
    type="PlanningMetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix="Test_" + task_name,
    save_metric_path=ckpt_dir,
)

# 4. Setup callbacks.
train_batch_processor = dict(
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

lr_update_callback = dict(
    type="CosLrUpdater",
    warmup_by="epoch",
    warmup_len=warmup_epoch,
    step_log_interval=log_freq,
)
stats_callback = dict(
    type="StatsMonitor",
    log_freq=log_freq,
)


viz_callback = dict(
    type="PlanningViz",
    height=img_h,
    width=img_w,
    resolution=img_resolution,
    bev_ego_center=bev_ego_center,
    video_save_path=ckpt_dir,
    video_name="viz_planning",
    video_fps=10,
    viz_on_every_epoch=False,
    viz_epoch=0,
)


# 5. Trainer and solver.
# -- 1. float.
float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=copy.deepcopy(model),
    data_loader=train_dataloader,
    optimizer=dict(
        type=torch.optim.Adam,
        lr=float_lr,
        weight_decay=weight_decay,
    ),
    batch_processor=train_batch_processor,
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
            monitor_metric_key="svf_diff",
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
    data_loader=test_dataloader,
    batch_processor=val_batch_processor,
    device=None,
    metrics=metrics,
    callbacks=[
        test_metric_updater,
        viz_callback,
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
    batch_processor=train_batch_processor,
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
            monitor_metric_key="svf_diff",
        ),
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
    data_loader=test_dataloader,
    batch_processor=val_batch_processor,
    device=None,
    metrics=metrics,
    callbacks=[
        test_metric_updater,
        viz_callback,
    ],
    log_interval=50,
)

# -- 3. int.
# ---- add description
params_desc = dict(
    context_frames=context_frames // pkl_freq_ratio,
    image_h=data_shape[1],
    image_w=data_shape[2],
    resolution=img_resolution,
    BEV_origin_x=bev_origin_x,
    BEV_origin_y=bev_origin_y,
    ego_length=4,
    ego_width=1.5,
    ego_offset=1.35,
    traj_len=12,
    use_drivable_area=0,
)

add_desc_pp = dict(
    type="AddDesc",
    per_tensor_desc=[
        # num of desc == num of pred output tensors (one tensor per out stride)
        json.dumps(
            dict(
                task="imit_value",
                output_name="pi",
            )
        ),
        json.dumps(
            dict(
                task="imit_value",
                output_name="svf",
            )
        ),
        json.dumps(
            dict(
                task="imit_value",
                output_name="svf_debug",
            )
        ),
        json.dumps(
            dict(
                task="imit_value",
                output_name="img_feats",
            )
        ),
        json.dumps(
            dict(
                task="imit_value",
                output_name="reward",
            )
        ),
    ],
)

deploy_model = copy.deepcopy(model)
# TODO Add DESC
# deploy_model["post_process"] = add_desc_pp
deploy_model["is_int_infer_model"] = True

for _, head in deploy_model["heads"].items():
    head["loss"] = None
    head["post_process"] = None


deploy_inputs = dict(
    road_map=torch.zeros((1, 1, img_h, img_w)),
    rendered_obs=torch.zeros(
        (1, context_frames // pkl_freq_ratio, img_h, img_w)
    ),
    plan_svf_goal=torch.zeros((1, 1, 65, 65)),
    plan_ego_motion=torch.ones(size=(1, 2, 65, 65)),
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
        dict(
            type="Checkpoint",
            save_dir=ckpt_dir,
            name_prefix="int-",
            strict_match=True,
            mode="min",
            monitor_metric_key="svf_diff",
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
    data_loader=test_dataloader,
    batch_processor=val_batch_processor,
    device=None,
    metrics=metrics,
    callbacks=[
        test_metric_updater,
    ],
    log_interval=50,
)

compile_cfg = dict(
    march=March.BAYES,
    name="plan_imitation_value",
    out_dir=ckpt_dir,
    hbm=os.path.join(ckpt_dir, "model.hbm"),
    layer_details=True,
    input_source=["ddr"],
)
