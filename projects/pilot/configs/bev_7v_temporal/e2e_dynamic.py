import copy
import json
import os
import os.path as osp
from collections import OrderedDict

import mpi4py.MPI as MPI
import torch

from hat.utils.apply_func import _as_list
from hat.utils.distributed import get_dist_info
from hat.utils.filesystem import join_path
from hat.utils.pack_type.lmdb import Lmdb
from projects.pilot.configs.bev_7v_temporal.base import (
    get_data_dict,
    get_datasets,
    get_homo_transforms,
    get_template_dataset,
    get_update_bev_dataset_func,
    save_data_to_npy,
)
from projects.pilot.configs.bev_7v_temporal.bev_3d_vehicle import (
    cls_dimension as vehicle_cls_dimension,
)
from projects.pilot.configs.bev_7v_temporal.bev_3d_vehicle import (
    nms_setting as veh_nms_setting,
)
from projects.pilot.configs.bev_7v_temporal.bev_3d_vehicle import (
    postprocess as vehicle_postprocess,
)
from projects.pilot.configs.bev_7v_temporal.bev_3d_vehicle import (
    roi_resize as vehicle_roi_resize,
)
from projects.pilot.configs.bev_7v_temporal.bev_3d_vehicle import (
    task_head as vehicle_task_head,
)
from projects.pilot.configs.bev_7v_temporal.bev_3d_vrumerge import (
    cls_dimension as vru_cls_dimension,
)
from projects.pilot.configs.bev_7v_temporal.bev_3d_vrumerge import (
    nms_setting as vru_nms_setting,
)
from projects.pilot.configs.bev_7v_temporal.bev_3d_vrumerge import (
    postprocess as vrumerge_postprocess,
)
from projects.pilot.configs.bev_7v_temporal.bev_3d_vrumerge import (
    roi_resize as vrumerge_roi_resize,
)
from projects.pilot.configs.bev_7v_temporal.bev_3d_vrumerge import (
    roi_vcs_range,
)
from projects.pilot.configs.bev_7v_temporal.bev_3d_vrumerge import (
    task_head as vrumerge_task_head,
)
from projects.pilot.configs.bev_7v_temporal.bev_3d_vrumerge import (
    task_out_size as vru_ipm_output_size,
)
from projects.pilot.configs.bev_7v_temporal.common import (
    H_persp_view_scale,
    backbone,
    bev_backbone,
    bev_fusion,
    bev_fusion_upsampling,
    bev_neck,
    bevfusion_output_size,
    bevfusion_spatial_resolution,
    bucket_root,
    cal_homo_offset_on_gpu,
    camera_view_names,
    deploy_head,
    deploy_narrow_head,
    deploy_side_head,
    deploy_stage2_inputs_key,
    front_camera_view_names,
    head,
    img_ori_size,
    img_resize_size,
    ipm_output_size,
    model_thresh,
    multi_view_collect,
    narrow_backbone,
    narrow_camera_view_names,
    narrow_head,
    narrow_pafpn_neck,
    offset_save_path,
    pafpn_neck,
    pipeline_test,
    save_prefix,
    side_backbone,
    side_camera_view_names,
    side_head,
    side_pafpn_neck,
    spatial_resolution,
    temporal_fusion,
    train_global_sample_interval,
    training_step,
    use_distorted_offset,
    vcs_origin_coord,
    vcs_plane_heights,
    vcs_range,
    warp_offset_range,
)
from projects.pilot.configs.bev_7v_temporal.e2e_base import (
    VALDATA_CLIP_LEN,
    get_dataloader,
    get_e2e_transforms,
    get_metric_updater,
    num_frames_for_pred,
    num_frames_per_clip,
    prepare_e2e_fake_dataset,
    train_data_version_dict,
    train_num_frames_per_iter,
    val_data_version_dict,
    val_metric_update_func,
    val_num_frames_per_iter,
)

TIME_DELTA = 0.1
# --------------------------FSD BASE ---------------------------
task_name = "e2e_dynamic"
job_name = "traj"
cfg_dir = os.path.dirname(__file__)

use_norm_rot = 1
e2e_use_qim = True
e2e_use_mb = True
enable_rfs = False
e2e_out_velocity = True
e2e_out_trajectory = True
loss_params = {
    "use_aux_loss": True,
    "cost_class": 2,
    "cost_bbox": 5,
    "cost_iou": 1,
    "iou_loss_type": "ciou",
}
e2e_traj_params = {
    "car_ped_cyc_type_id": {"vehicle": 0, "cyclist": 1, "pedestrain": 2},
    "static_thr": {"vehicle": 1, "cyclist": 0.7, "pedestrain": 0.2},
    "min_num_his_frame_thr": 2,
    "min_num_fut_frame_thr": 1,
    "max_context_frame_num": 4,
    "pred_traj_modal_num": 5,
    "max_his_odo_len": 12,
    "bounce_thr": 3.1415926,
    "static_obs_compensate_prob": 0.6,
    "ego_vcs_trans_flag": True,
    "obs_lcf_position_trans_flag": True,
    "obs_lcf_yaw_rotate_flag": False,
    "filter_unstable_class": True,
    "fill_inverse_traj_flag": True,
    "fill_inverse_traj_prob": 0.4,
    "save_memory_flag": False,
    "only_keep_vehicle": False,
    "ego_as_obs_flag": False,
}
e2e_predefined_classes = ("car", "bicycle", "pedestrian")
init_query_feat = False
eval_pure_det = False
eval_setting = eval_setting = {
    "e2e_dynamic": {
        "eval_category_ids": (0, 1, 2),
        "score_threshold": 0.1,
        "iou_threshold": 0.2,
        "gt_max_depth": 100,
        "depth_intervals": (
            5,
            10,
            20,
            30,
            40,
            50,
        ),
        "eval_mode": "let_iou",
        "let_iou_param": {"p_t": 0.35, "min_t": 4.0, "max_t": 8.0},
    }
}
output_labels_group = {"veh": [0], "vru": [1, 2]}

e2e_track_manager_kwargs = {
    # 0: vehicle, 1: cyclist 2:pedestrian
    "cls_score_thr": {
        0: model_thresh.get(task_name, {})
        .get("cls_score_thr", {})
        .get(
            "vehicle",
            {
                "score_threshold": 0.15,
                "filter_score_thresh": 0.1,
            },
        )
        if model_thresh
        else {
            "score_threshold": 0.15,
            "filter_score_thresh": 0.1,
        },
        1: model_thresh.get(task_name, {})
        .get("cls_score_thr", {})
        .get(
            "cyclist",
            {
                "score_threshold": 0.15,
                "filter_score_thresh": 0.1,
            },
        )
        if model_thresh
        else {
            "score_threshold": 0.15,
            "filter_score_thresh": 0.1,
        },
        2: model_thresh.get(task_name, {})
        .get("cls_score_thr", {})
        .get(
            "pedestrian",
            {
                "score_threshold": 0.15,
                "filter_score_thresh": 0.1,
            },
        )
        if model_thresh
        else {
            "score_threshold": 0.15,
            "filter_score_thresh": 0.1,
        },
    },
    "miss_tolerance": 5,
    "nms_setting": {
        "agnostic": True,
        "let_nms": veh_nms_setting["letnms"][:1]
        + vru_nms_setting["letnms"],  # import from vehicle&vrumerge
    },
    "vcs_range": vcs_range,
    "valid_range": (-30.0, -50.0, 100.0, 50.0),
    "traj_thickness": 1.0,
}
loss_weight = {
    "loss_ce": 2,
    "loss_xy": 5,
    "loss_wh": 2.5,
    "loss_ciou": 1,
    "loss_zheight": 1,
    "loss_yaw": 1,
    "loss_velocity": 0.2,
    "loss_velo_yaw_consistency": 0.05,
    "trackloss_ce": 2,
    "loss_trajprob": 5,
    "loss_trajreg": 5,
}
filter_vcs_range = {
    0: vcs_range,
    1: roi_vcs_range,
    2: roi_vcs_range,
}
cls_bev_size = {
    0: ipm_output_size,
    1: vru_ipm_output_size,
    2: vru_ipm_output_size,
}
cls_hm_kernel = {0: 13, 1: 9, 2: 9}
max_his_odo_len = 12
e2e_query_kwargs = {
    "num_veh_queries": 128,
    "num_vru_queries": 64,
    "veh_filter_scores": 0.14,
    "vru_filter_scores": 0.14,
    "num_track_queries": 60,
    "query_drop_ratio": 0.1,
    "query_fp_ratio": 0.2,
    "num_max_queries": 1000,
    "fp_rand_ref_pts_value": 0.02,
    "fp_ref_pts_bias": 0.03,
}
e2e_transformer_params = {
    "feat_shape": (
        (bevfusion_output_size[1] // 4, bevfusion_output_size[0] // 4),
        (bevfusion_output_size[1] // 8, bevfusion_output_size[0] // 8),
        (bevfusion_output_size[1] // 16, bevfusion_output_size[0] // 16),
        (bevfusion_output_size[1] // 32, bevfusion_output_size[0] // 32),
    ),
    "embedding_dim": 256,
    "num_head": 8,
    "num_queries": (
        e2e_query_kwargs["num_veh_queries"]
        + e2e_query_kwargs["num_vru_queries"]
        + e2e_query_kwargs["num_track_queries"]
    ),
    "num_det_queries": (
        e2e_query_kwargs["num_veh_queries"]
        + e2e_query_kwargs["num_vru_queries"]
    ),
    "num_decoder_layers": 6,
    "feedforward_dim": 512,
    "dropout_ratio": 0.1,
    "return_intermediate_dec": True,
    "extra_track_attn": True,
}
# ---------------------------- TASK ----------------------------
save_eval_results = f"tmp_output/{task_name}/{job_name}"
save_vis_dir = None
if os.path.exists("/job_data"):
    save_eval_results = os.path.join("/job_data", save_eval_results)
    if save_vis_dir:
        save_vis_dir = os.path.join("/job_data", save_vis_dir)
vis_setting = dict(
    bev_size=bevfusion_output_size,
    vcs_range=vcs_range,
    # 可视化默认使用veh的score
    score_threshold=e2e_track_manager_kwargs["cls_score_thr"][0][
        "score_threshold"
    ],
    project_bbox_to_cameras=True,
    anno_show=True,
    per_extra_img_size=(960, 512),
    vis_tracking=True,
    vis_trajectory=e2e_out_trajectory,
    vis_velocity=e2e_out_velocity,
    save_video=False,
)

vis_setting["camera_view_names"] = camera_view_names
evs_setting = {
    "user_name": "",
    "token": "",
    "pack_dir": "/horizon-bucket/SD_Algorithm/06_perception_bev_dynamic/01_unpack_package/sd_dynamic_11v",  # noqa
    "ipd_number": "PDT2021004-bev",
    "evs_task_name": job_name,
    "task_queue": "svc-aip-cpu",
    "evs_data_cfg_path": os.path.join(
        cfg_dir, "../datasets/e2e/evs_dataset.yaml"
    ),
}
e2e_submit_evs = True
e2e_eval_tracking = True
category2id_map = {
    0: 0,  # "car": 0,
    1: 1,  # "cyclist": 1,
    2: 2,  # "pedestrian": 2,
    3: 0,  # "truck": 3,
    4: 0,  # "tricycle": 4,
    5: 0,  # "bus": 5,
    6: 0,  # "construction": 6,
    7: 0,  # "blur": 7,
    8: -99,  # "other": 8,
    9: 0,  # "van": 9,
    11: 0,  # "bigmot": 10,
    -99: -99,
}
use_psc_rot = True
N_steps_PSC_rot = 3
# ---------------------------  DATA  ------------------------
if pipeline_test:
    train_version = "pipeline_test"
    val_version = "pipeline_test"
else:
    train_version = "v4.0.0"
    val_version = "v2.0.0"
calibration_version = "v0.0.2"

train_dataset_dict = train_data_version_dict[train_version]
calib_dataset_dict = val_data_version_dict[calibration_version]
val_dataset_dict = val_data_version_dict[val_version]

# cat the range in z dim (bottom, top)
vcs_range = vcs_range + (-3, 5)
e2e_train_target_transform = dict(
    type="ANCE2EDynamicTargetGenerator",
    vcs_range=vcs_range,
    output_labels_group=output_labels_group,
    filter_vcs_range=filter_vcs_range,
    cls_bev_size=cls_bev_size,
    cls_hm_kernel=cls_hm_kernel,
    num_frames_per_clip=num_frames_per_clip,
    num_frames_per_iter=train_num_frames_per_iter,
    max_his_odo_len=max_his_odo_len,
    trajpred_transform=[],
    use_psc_rot=use_psc_rot,
    N_steps_PSC_rot=N_steps_PSC_rot,
    category2id_map=category2id_map,
)

e2e_val_target_transform = copy.deepcopy(e2e_train_target_transform)
# according to the dataset package setting
e2e_val_target_transform["num_frames_per_clip"] = VALDATA_CLIP_LEN
e2e_val_target_transform["num_frames_per_iter"] = val_num_frames_per_iter
e2e_train_target_transform["trajpred_transform"].append(
    dict(
        type="ANCObtainHomographyTemporal",
        bev_size=ipm_output_size,
        vcs_range=vcs_range,
        return_relative=True,
    )
)
e2e_val_target_transform["trajpred_transform"].append(
    dict(
        type="ANCObtainHomographyTemporal",
        bev_size=ipm_output_size,
        vcs_range=vcs_range,
        return_relative=True,
    )
)

if e2e_out_trajectory:
    train_traj_pred_multi_tramsform = dict(
        type="TrajPredMultiTransform",
        use_fut_info_filter=True,
        vcs_range=vcs_range,
        car_ped_cyc_type_id=e2e_traj_params["car_ped_cyc_type_id"],
        static_thr=e2e_traj_params["static_thr"],
        num_sample_per_clip=num_frames_per_clip,
        min_num_his_frame_thr=e2e_traj_params["min_num_his_frame_thr"],
        min_num_fut_frame_thr=e2e_traj_params["min_num_fut_frame_thr"],
        max_context_frame_num=e2e_traj_params["max_context_frame_num"],
        num_frames_for_pred=num_frames_for_pred,
        pred_traj_modals_num=e2e_traj_params["pred_traj_modal_num"],
        max_his_odo_len=e2e_traj_params["max_his_odo_len"],
        yaw_diff_bounce_thr=e2e_traj_params["bounce_thr"],
        static_obs_compensate_prob=e2e_traj_params[
            "static_obs_compensate_prob"
        ],
        filter_unstable_class=e2e_traj_params["filter_unstable_class"],
        visibility=2,
        ego_vcs_trans_flag=e2e_traj_params["ego_vcs_trans_flag"],
        obs_lcf_position_trans_flag=e2e_traj_params[
            "obs_lcf_position_trans_flag"
        ],
        obs_lcf_yaw_rotate_flag=e2e_traj_params["obs_lcf_yaw_rotate_flag"],
        future_traj_only=True,
        fill_inverse_traj_flag=e2e_traj_params["fill_inverse_traj_flag"],
        fill_inverse_traj_prob=e2e_traj_params["fill_inverse_traj_prob"],
        only_keep_vehicle=e2e_traj_params["only_keep_vehicle"],
        ego_as_obs_flag=e2e_traj_params["ego_as_obs_flag"],
        default_ego_id=-1,
        default_ego_label=0,
        save_memory_flag=e2e_traj_params["save_memory_flag"],
    )
    val_traj_pred_multi_transform = copy.deepcopy(
        train_traj_pred_multi_tramsform
    )
    val_traj_pred_multi_transform.update(
        {
            "use_fut_info_filter": False,
            "min_num_his_frame_thr": 1,
            "min_num_fut_frame_thr": 0,
            "static_thr": {"vehicle": 2, "cyclist": 0.4, "pedestrain": 0.4},
            "yaw_diff_bounce_thr": 3.14 / 3,
            "static_obs_compensate_prob": 1.0,
            "filter_unstable_class": False,
            "fill_inverse_traj_flag": False,
            "fill_inverse_traj_prob": 1.0,
            "save_memory_flag": False,
            "num_sample_per_clip": VALDATA_CLIP_LEN,
        }
    )

    e2e_train_target_transform["trajpred_transform"].append(
        train_traj_pred_multi_tramsform,
    )
    e2e_val_target_transform["trajpred_transform"].append(
        val_traj_pred_multi_transform,
    )

(
    train_e2e_transforms,
    val_e2e_transforms,
    calib_e2e_transforms,
) = get_e2e_transforms(e2e_train_target_transform, e2e_val_target_transform)

homo_transforms = get_homo_transforms(
    transforms_list=train_e2e_transforms, camera_view_names=camera_view_names
)

train_template_dataset = get_template_dataset(
    img_load_size=img_resize_size,
    camera_view_names=camera_view_names,
    per_view_shape=img_ori_size,
    transforms=train_e2e_transforms,
    homo_transforms=homo_transforms,
    spatial_resolution=spatial_resolution,
    H_persp_view_scale=H_persp_view_scale,
    vcs_range=vcs_range,
    use_distorted_offset=use_distorted_offset,
    vcs_plane_heights=vcs_plane_heights,
    cal_homo_offset_on_gpu=cal_homo_offset_on_gpu,
    offset_save_path=offset_save_path,
    temporal_bev=True,
    length_of_clip=num_frames_per_clip,
    train_num_frames_per_iter=train_num_frames_per_iter,
)
train_template_dataset["reverse"] = True

val_template_dataset = copy.deepcopy(train_template_dataset)

# according to the dataset package setting
val_template_dataset["num_max_frames"] = VALDATA_CLIP_LEN
val_template_dataset["num_frames_per_iter"] = val_num_frames_per_iter
val_template_dataset["transforms"] = val_e2e_transforms

calib_template_dataset = copy.deepcopy(val_template_dataset)
calib_template_dataset["transforms"] = calib_e2e_transforms

e2e_dynamic_url = osp.join(cfg_dir, "../datasets/e2e/e2e_dynamic_dataset.yaml")
data_dict = get_data_dict(e2e_dynamic_url)
data_dict = join_path(bucket_root, data_dict, ["camera_module_type"])

update_trans_info = {
    "Collect3DV": "fill_fake_temporal_data",
}
added_info = [
    "e2e_dynamic_lmdb_path",
    "homo_path",
    "calib_path",
    "camera_module_type",
    "homo_noise",
]

update_bev_dataset = get_update_bev_dataset_func(
    added_info=added_info,
    update_trans_info=update_trans_info,
)

train_homo_noise = {
    "noise_value": (0.2, 0.2, 0.2, 0.0, 0.0, 0.04),
    # It is the exact value of generated noise when the noise type is
    # 'specific_cam'. When the noise type is 'random_cam' or 'random_vcs',
    # it is the upper bound value of generated random noises. The format is
    # (roll, pitch, yaw, x, y, z), the units are degrees and meters.
    "noise_type": "random_cam",
    "noise_view_names": camera_view_names,
}
val_homo_noise = None
train_with_homo_noise = False
val_with_homo_noise = False

train_datasets = get_datasets(
    train_dataset_dict,
    None,
    train_template_dataset,
    data_dict,
    update_bev_dataset,
    homo_noise=train_homo_noise if train_with_homo_noise else None,
    global_sample_interval=train_global_sample_interval,
)
save_path = os.path.join(
    save_prefix, "datasets_save", "train", task_name, "train.npy"
)
# only save once
if not os.path.exists(save_path):
    save_data_to_npy(train_datasets, save_path)

val_datasets = get_datasets(
    val_dataset_dict,
    None,
    val_template_dataset,
    data_dict,
    update_bev_dataset,
    homo_noise=val_homo_noise if val_with_homo_noise else None,
    repeat_dataset_times=3 if val_with_homo_noise else 1,
    global_sample_interval=1,
)
for val_name, val_dataset in zip(_as_list(val_version), val_datasets):
    save_path = os.path.join(
        save_prefix,
        "datasets_save",
        "val",
        task_name + "_" + val_name,
        "val.npy",
    )
    # only save once
    if not os.path.exists(save_path):
        save_data_to_npy(val_dataset, save_path)

calib_datasets = get_datasets(
    calib_dataset_dict,
    None,
    calib_template_dataset,
    data_dict,
    update_bev_dataset,
    homo_noise=val_homo_noise if val_with_homo_noise else None,
    repeat_dataset_times=3 if val_with_homo_noise else 1,
)

# 端到端任务评测过程中，需要保证一块GPU上至少有一个pack。这里对于GPU数目大于pack数的情况，
# 补充fake dataset 对pack数进行填补，fake dataset不参与评测。
# 由于启动方式（mpirun，torchrun）不同，在CONFIG中获取world_size的方式不同，否则得到的
# world_size始终为1。这里采用两种获取方法，得到的最大值即为实际的world_size.
# 采用mpirun 启动时，调用mpi4py可以得到正确的world_size。
mpi_world_size = MPI.COMM_WORLD.Get_size()
# 采用torchrun启动时并在本地运行时，调用torch.distributed可以得到正确的world_size。
_, dist_world_size_local = get_dist_info()
# 采用torchrun启动时并在集群运行时，采用world_size的环境变量可以得到正确的world_size。
dist_world_size_cluster = int(os.getenv("WORLD_SIZE", 1))
world_size = max(
    mpi_world_size, dist_world_size_local, dist_world_size_cluster
)
val_pack_number = 0
for val_dataset in val_datasets:
    assert val_dataset["dataset"].get("sync_file_lmdb") or val_dataset[
        "dataset"
    ].get("sync_file")
    if val_dataset["dataset"].get("sync_file_lmdb"):
        cur_lmdb = Lmdb(
            val_dataset["dataset"]["sync_file_lmdb"],
            False,
            True,
            readonly=True,
            map_size=1024 * 10,
        )
        pack_list = [
            json.loads(cur_lmdb.read(i).decode())["pack_dir"]
            for i in range(len(cur_lmdb))
        ]
    else:
        with open(val_dataset["dataset"]["sync_file"], "r") as f:
            lines = f.read().splitlines()
        pack_list = [json.loads(line)["pack_dir"] for line in lines]
    val_pack_number += len(set(pack_list))
if val_pack_number < world_size:
    fake_val_datasets_dict = val_data_version_dict["pipeline_test"]
    val_fake_datasets = get_datasets(
        fake_val_datasets_dict,
        None,
        val_template_dataset,
        data_dict,
        update_bev_dataset,
        homo_noise=val_homo_noise if val_with_homo_noise else None,
        repeat_dataset_times=3 if val_with_homo_noise else 1,
    )
    val_fake_datasets = prepare_e2e_fake_dataset(val_fake_datasets)
    val_datasets.extend(val_fake_datasets * (world_size - val_pack_number))
data_loader, val_data_loader, calib_data_loader = get_dataloader(
    train_datasets, val_datasets, calib_datasets
)
if training_step == "calibration":
    data_loader = calib_data_loader

val_data_loader_list = [val_data_loader]

# ----------------------------- MODEL -----------------------------
# --------------------------- e2e head ---------------------------
decode_rot_setting = None
if use_psc_rot:
    decode_rot_setting = {
        "psc_rot": {
            "N_steps_PSC_rot": N_steps_PSC_rot,
        }
    }

subtask_heads = OrderedDict(
    bbox_embed_xy=dict(
        type="MLP",
        input_dim=256,
        hidden_dim=256,
        output_dim=2,
        num_layers=3,
        is_output=False,
    ),
    bbox_embed_wh=dict(
        type="MLP",
        input_dim=256,
        hidden_dim=256,
        output_dim=2,
        num_layers=3,
        is_output=False,
        freeze_scale=False,
    ),
    class_embed=dict(type="Linear", in_channels=256, out_channels=3),
    yaw_embed=dict(
        type="Linear",
        in_channels=256,
        out_channels=N_steps_PSC_rot if use_psc_rot else 2,
    ),
    zheight_embed=dict(
        type="MLP",
        input_dim=256,
        hidden_dim=256,
        output_dim=2,
        num_layers=3,
        is_output=True,
    ),
)

extra_subtask_heads = {}
if e2e_out_velocity:
    extra_subtask_heads.update(
        velocity_embed=dict(
            type="MLP",
            input_dim=288,
            hidden_dim=256,
            output_dim=3,
            num_layers=3,
            is_output=True,
        ),
        K_embed=dict(
            type="MLP",
            input_dim=288,
            hidden_dim=256,
            output_dim=48,
            num_layers=3,
            is_output=False,
        ),
    )
if e2e_out_trajectory:
    extra_subtask_heads.update(
        trajpred_embed=dict(
            type="MLP",
            input_dim=288,
            hidden_dim=256,
            output_dim=120,
            num_layers=2,
            is_output=True,
        ),
        trajpred_logits=dict(
            type="MLP",
            input_dim=288,
            hidden_dim=256,
            output_dim=1,
            num_layers=4,
            is_output=True,
        ),
    )

transformer_head = dict(
    type="CroppedDeformableTransformer",
    **e2e_transformer_params,
)

matcher = dict(
    type="E2EHungarianMatcher",
    cls_loss_weight=loss_params["cost_class"],
    bbox_loss_weight=loss_params["cost_bbox"],
    iou_loss_weight=loss_params["cost_iou"],
    iou_loss_type=loss_params["iou_loss_type"],
    use_psc_rot=use_psc_rot,
)

criterion = dict(
    type="E2EClipMatcher",
    matcher=matcher,
    match_indices_each_layer=False,
    use_psc_rot=use_psc_rot,
    enable_rfs=enable_rfs,
)

post_process = dict(
    type="TrackerPostProcess",
    vcs_range=vcs_range,
    decode_rot_setting=decode_rot_setting,
)

track_manager = dict(
    type="TrackerManager",
    decode_rot_setting=decode_rot_setting,
    **e2e_track_manager_kwargs,
)

track_embed = dict(
    type="QuerySpatialInteractionModule",
    update_query_pos=False,
    dropout_ratio=e2e_transformer_params["dropout_ratio"],
    dim_in=256,
    hidden_dim=512,
    dim_out=512,
    replace_identity_with_tgt=False,
)

memory_bank = dict(
    type="MemoryBankModule",
    memory_bank_len=10,
    memory_bank_with_temp_attn=True,
    memory_bank_with_self_attn=True,
    num_heads=8,
    dim_in=256,
    hidden_dim=512,
    dim_out=512,
)

wlh_anchor_size = {
    "car": [
        vehicle_cls_dimension[0][1] / (vcs_range[3] - vcs_range[1]),
        vehicle_cls_dimension[0][2] / (vcs_range[2] - vcs_range[0]),
        vehicle_cls_dimension[0][0] / (vcs_range[5] - vcs_range[4]),
    ],  # car
    "cyclist": [
        vru_cls_dimension[0][1] / (vcs_range[3] - vcs_range[1]),
        vru_cls_dimension[0][2] / (vcs_range[2] - vcs_range[0]),
        vru_cls_dimension[0][0] / (vcs_range[5] - vcs_range[4]),
    ],  # cyclist
    "pedestrian": [
        vru_cls_dimension[1][1] / (vcs_range[3] - vcs_range[1]),
        vru_cls_dimension[1][2] / (vcs_range[2] - vcs_range[0]),
        vru_cls_dimension[1][0] / (vcs_range[5] - vcs_range[4]),
    ],  # pedestrian
}

motr_head = dict(
    type="E2EDynamicModule",
    transformer=transformer_head,
    in_strides=[2, 4, 8, 16, 32],
    out_strides=[2, 4, 8, 16, 32],
    decoder_strides=[4, 8, 16, 32],
    num_channels=[48, 48, 96, 192, 192],
    odometry_channels=[2, 32, 32],
    wlh_anchor_size=wlh_anchor_size,  # width and height,
    bev_size=cls_bev_size[0],
    vcs_range=vcs_range,
    default_time_delta=TIME_DELTA,
    subtask_head=subtask_heads,
    extra_subtask_head=extra_subtask_heads,
    criterion=criterion,
    query_interaction_module=track_embed if e2e_use_qim else None,
    memory_bank_module=memory_bank if e2e_use_mb else None,
    track_manager=None,
    post_process=post_process,
    e2e_query_kwargs=e2e_query_kwargs,
    calibration_model=False,
    compile_model=False,
    e2e_out_velocity=e2e_out_velocity,
    e2e_out_trajectory=e2e_out_trajectory,
    trajectory_kwargs={
        "num_traj_modal": e2e_traj_params["pred_traj_modal_num"],
        "num_frames_for_pred": num_frames_for_pred,
    },
    feature_name="veh_feats",
    use_box_refine=True,
    use_auxiliary_loss=loss_params["use_aux_loss"],
    init_query_feat=init_query_feat,
    eval_pure_det=eval_pure_det,
)
e2e_head = dict(
    type="OutputModule",
    head=motr_head,
    head_parser=None,
    loss=None,
    postprocess=None,
    target=None,
    node_name=f"bev_stage2_{task_name}_head",
    prefix=f"bev_stage2_{task_name}_head",
)

neck_identity_head = dict(
    type="OutputModule",
    head=dict(
        type="ANCIdentityHead",
        dequant_out=True,
        feature_name="bev_neck_feats",
    ),
    postprocess=dict(
        type="AddDesc",
        per_tensor_desc=[
            # num of desc == num of pred output tensors
            json.dumps(
                dict(
                    task="beve2e_trackernet",
                    **dict(
                        output_name="beve2e_trackernet_input_feature0",
                        bev_stage2_input_resolution=spatial_resolution,
                        warp_offset_range=warp_offset_range,
                        vcs_plane_heights=vcs_plane_heights,
                    ),
                )
            ),
            json.dumps(
                dict(
                    task="beve2e_trackernet",
                    **dict(
                        output_name="beve2e_trackernet_input_feature1",
                        bev_stage2_input_resolution=spatial_resolution,
                        warp_offset_range=warp_offset_range,
                        vcs_plane_heights=vcs_plane_heights,
                    ),
                )
            ),
            json.dumps(
                dict(
                    task="beve2e_trackernet",
                    **dict(
                        output_name="beve2e_trackernet_input_feature2",
                        bev_stage2_input_resolution=spatial_resolution,
                        warp_offset_range=warp_offset_range,
                        vcs_plane_heights=vcs_plane_heights,
                    ),
                )
            ),
            json.dumps(
                dict(
                    task="beve2e_trackernet",
                    **dict(
                        output_name="beve2e_trackernet_input_feature3",
                        bev_stage2_input_resolution=spatial_resolution,
                        warp_offset_range=warp_offset_range,
                        vcs_plane_heights=vcs_plane_heights,
                    ),
                )
            ),
            json.dumps(
                dict(
                    task="beve2e_trackernet",
                    **dict(
                        output_name="beve2e_trackernet_input_feature4",
                        bev_stage2_input_resolution=spatial_resolution,
                        warp_offset_range=warp_offset_range,
                        vcs_plane_heights=vcs_plane_heights,
                    ),
                )
            ),
        ],
    ),
    prefix="bev_stage2_neck_feat",
    node_name="bev_stage2_neck_feat",
)

num_det_queries = (
    e2e_query_kwargs["num_veh_queries"] + e2e_query_kwargs["num_vru_queries"]
)

bev_common_desc = dict(
    vcs_origin_coord=vcs_origin_coord,
    bev_stage2_input_resolution=spatial_resolution,
    bev_stage2_output_resolution=bevfusion_spatial_resolution,
    visible_range=warp_offset_range,
    detect_query_num=num_det_queries,
    veh_query_num=e2e_query_kwargs["num_veh_queries"],
    vru_query_num=e2e_query_kwargs["num_vru_queries"],
    track_query_num=e2e_query_kwargs["num_track_queries"],
    time_delta=0.0,  # 默认使用平滑的 time_delta 设置，time_delta 根据实际的时间间隔进行平滑
    time_delta_alpha=0.1,
    mem_bank_len=memory_bank["memory_bank_len"],
)


def get_track_manager_desc(e2e_trk_kwargs):
    # nms-setting to desc
    nms_setting = e2e_trk_kwargs["nms_setting"]
    if "nms" in nms_setting:
        desc_dict = dict(
            e2e_nms_thresh=nms_setting["nms"]["nms_thresh"], e2e_nms_type=1
        )
    elif "let_nms" in nms_setting:
        let_nms = nms_setting["let_nms"]
        desc_dict = {"use_ct_nms": [0] * len(let_nms), "e2e_nms_type": 2}
        for idx, cur_letnms in enumerate(let_nms):
            for key, val in cur_letnms.items():
                if "ct_nms" not in key:
                    if key not in desc_dict:
                        desc_dict[key] = []
                    desc_dict[key].append(val)
                else:
                    desc_dict["use_ct_nms"][idx] = 1
                    for k1, vv in val.items():  # ct_nms_param
                        if k1 not in desc_dict:
                            desc_dict[k1] = [0] * len(let_nms)
                        desc_dict[k1][idx] = (
                            int(vv) if isinstance(vv, bool) else vv
                        )
    else:
        raise ValueError(
            f"nms_setting is not supported yet in compile, {nms_setting}"
        )

    desc_dict["agnostic"] = 1 if nms_setting.get("agnostic", True) else 0

    # valid-filter to desc
    desc_dict["do_valid_range_filter"] = 1
    desc_dict["valid_range"] = [
        e2e_trk_kwargs["valid_range"][2],
        e2e_trk_kwargs["valid_range"][0],
        e2e_trk_kwargs["valid_range"][3],
        e2e_trk_kwargs["valid_range"][1],
    ]  # (top, bottom, left, right)

    desc_dict["miss_tolerance"] = e2e_trk_kwargs["miss_tolerance"]
    desc_dict["traj_thickness"] = e2e_trk_kwargs["traj_thickness"]

    score_thresh = [
        val["score_threshold"]
        for _, val in e2e_trk_kwargs["cls_score_thr"].items()
    ]
    filter_score_thresh = [
        val["filter_score_thresh"]
        for _, val in e2e_trk_kwargs["cls_score_thr"].items()
    ]
    desc_dict["score_thresh"] = score_thresh
    desc_dict["filter_score_thresh"] = filter_score_thresh
    return desc_dict


trk_manager_desc = get_track_manager_desc(e2e_track_manager_kwargs)
trk_manager_desc["kalman_filter_thresh"] = 4


def get_bev_desc(
    use_psc_rot=False,
    N_steps_PSC_rot=None,
):

    if use_psc_rot:
        beve2e_trackernet_outputs_yaws_desc = {
            "task": "beve2e_trackernet",
            "output_name": "beve2e_trackernet_outputs_yaws",
            "channel_labels": [
                f"cos_idx{idx}" for idx in range(N_steps_PSC_rot)
            ],
            "rot_mod_threshold": 0.0001,
            "use_norm_rot": use_norm_rot,
            **bev_common_desc,
        }
    else:
        beve2e_trackernet_outputs_yaws_desc = {
            "task": "beve2e_trackernet",
            "output_name": "beve2e_trackernet_outputs_yaws",
            "channel_labels": ["cos", "sin"],
            "use_norm_rot": 0,
            **bev_common_desc,
        }

    per_tensor_desc = [
        {
            "task": "beve2e_trackernet",
            "output_name": "beve2e_trackernet_outputs_query_pos",
            **bev_common_desc,
        },
        {
            "task": "beve2e_trackernet",
            "output_name": "beve2e_trackernet_outputs_class",
            **bev_common_desc,
        },
        {
            "task": "beve2e_trackernet",
            "output_name": "beve2e_trackernet_outputs_coord_xy",
            **bev_common_desc,
            **trk_manager_desc,
        },
        {
            "task": "beve2e_trackernet",
            "output_name": "beve2e_trackernet_outputs_coord_wh",
            "wh_anchor": sum(
                [value[:2] for _, value in wlh_anchor_size.items()], []
            ),
            **bev_common_desc,
        },
        beve2e_trackernet_outputs_yaws_desc,
        {
            "task": "beve2e_trackernet",
            "output_name": "beve2e_trackernet_outputs_zheights",
            "h_anchor": [value[2] for _, value in wlh_anchor_size.items()],
            "zh_vcs_range": [-3.0, 5.0],
            **bev_common_desc,
        },
        {
            "task": "beve2e_trackernet",
            "output_name": "beve2e_trackernet_output_embedding_for_mem",
            **bev_common_desc,
        },
        {
            "task": "beve2e_trackernet",
            "output_name": "beve2e_trackernet_output_embedding_for_qim",
            **bev_common_desc,
        },
        {
            "task": "beve2e_trackernet",
            "output_name": "beve2e_trackernet_output_track_scores",
            **bev_common_desc,
        },
        {
            "task": "beve2e_trackernet",
            "output_name": "beve2e_trackernet_outputs_velocities",
            **bev_common_desc,
        },
        {
            "task": "beve2e_trackernet",
            "output_name": "beve2e_trackernet_outputs_K_mat",
            **bev_common_desc,
        },
        {
            "task": "beve2e_trackernet",
            "output_name": "beve2e_trackernet_outputs_trajRegs",
            **bev_common_desc,
        },
        {
            "task": "beve2e_trackernet",
            "output_name": "beve2e_trackernet_outputs_trajLOgits",
            **bev_common_desc,
        },
    ]
    per_tensor_desc = [json.dumps(i) for i in per_tensor_desc]
    return per_tensor_desc


def get_model(mode):
    veh_postprocess = copy.deepcopy(vehicle_postprocess)
    veh_postprocess[-1]["nms_setting"] = None
    veh_postprocess[-1]["use_maxpool"] = True
    veh_postprocess[-1]["max_pool_kernel"] = 3
    veh_postprocess[-1]["score_threshold"] = e2e_query_kwargs[
        "veh_filter_scores"
    ]
    veh_postprocess[-1]["node_name"] += "_e2e"
    veh_head = dict(
        type="OutputModule",
        head=vehicle_task_head,
        head_parser=None,
        loss=None,
        postprocess=veh_postprocess,
        target=None,
        convert_to_dict=True,
        trace_convert=True,
        prefix="bev_stage2_3d_vehicle_head",
    )
    vru_postprocess = copy.deepcopy(vrumerge_postprocess)
    vru_postprocess[-1]["nms_setting"] = None
    vru_postprocess[-1]["use_maxpool"] = True
    vru_postprocess[-1]["max_pool_kernel"] = 3
    vru_postprocess[-1]["score_threshold"] = e2e_query_kwargs[
        "vru_filter_scores"
    ]
    vru_postprocess[-1]["node_name"] += "_e2e"
    vru_head = dict(
        type="OutputModule",
        head=vrumerge_task_head,
        head_parser=None,
        loss=None,
        postprocess=vru_postprocess,
        target=None,
        convert_to_dict=True,
        trace_convert=True,
        prefix="bev_stage2_3d_vrumerge_head",
    )
    bev_head = dict(
        type="E2EPlusBEV3DHead",
        veh_head=veh_head,
        vru_head=vru_head,
        e2e_dynamic_head=e2e_head if mode != "deploy" else neck_identity_head,
        veh_roi_resizes=vehicle_roi_resize,
        vru_roi_resizes=vrumerge_roi_resize,
        compile_mode=mode == "deploy",
    )

    if mode == "train":
        bev_head["veh_head"]["loss"] = dict(
            type="ANCBEV3DLoss",
            loss_weights=dict(bev3d_hm=1.0, bev3d_ct_offset=1.5),
            gamma=1,
            beta=0.33,
            node_name="bev_stage2_e2e_veh_loss",
        )
        bev_head["vru_head"]["loss"] = dict(
            type="ANCBEV3DLoss",
            loss_weights=dict(bev3d_hm=1.0, bev3d_ct_offset=1.5),
            gamma=1,
            beta=0.33,
            node_name="bev_stage2_e2e_vru_loss",
        )
        bev_head["e2e_dynamic_head"]["loss"] = dict(
            type="E2EDynamicLoss",
            map_label={"car": 0, "cyclist": 1, "pedestrain": 2},
            loss_weights=loss_weight,
            iou_loss_type="ciou",
            calc_velocity_loss=e2e_out_velocity,
            calc_trajectory_loss=e2e_out_trajectory,
            use_psc_rot=use_psc_rot,
            N_steps_PSC_rot=N_steps_PSC_rot,
        )
    elif mode == "val":
        bev_head["e2e_dynamic_head"]["head"]["use_auxiliary_loss"] = False
        bev_head["e2e_dynamic_head"]["loss"] = None
        bev_head["e2e_dynamic_head"]["head"]["track_manager"] = track_manager

    if training_step == "calibration":
        bev_head["vru_head"]["loss"] = None
        bev_head["veh_head"]["loss"] = None
        bev_head["e2e_dynamic_head"]["loss"] = None
        bev_head["e2e_dynamic_head"]["head"]["track_manager"] = track_manager

    multi_view_module = dict(
        img=dict(
            type="BEVStageOneModule",
            backbone=backbone,
            neck=pafpn_neck,
            head=head,
        ),
        side_img=dict(
            type="BEVStageOneModule",
            backbone=side_backbone,
            neck=side_pafpn_neck,
            head=side_head,
        ),
        narrow_img=dict(
            type="BEVStageOneModule",
            backbone=narrow_backbone,
            neck=narrow_pafpn_neck,
            head=narrow_head,
        ),
    )
    bevfusion_pick_keys = None
    if mode == "deploy":
        multi_view_module = dict()
        for key in camera_view_names:
            if key in front_camera_view_names:
                multi_view_module[key] = dict(
                    type="BEVStageOneModule",
                    backbone=backbone,
                    neck=pafpn_neck,
                    head=deploy_head,
                )
            elif key in side_camera_view_names:
                multi_view_module[key] = dict(
                    type="BEVStageOneModule",
                    backbone=side_backbone,
                    neck=side_pafpn_neck,
                    head=deploy_side_head,
                )
            elif key in narrow_camera_view_names:
                multi_view_module[key] = dict(
                    type="BEVStageOneModule",
                    backbone=narrow_backbone,
                    neck=narrow_pafpn_neck,
                    head=deploy_narrow_head,
                )
            else:
                raise TypeError
        bevfusion_pick_keys = deploy_stage2_inputs_key

    model = dict(
        type="MultiViewTwoStageBEVModule",
        # view_img_key to module
        multi_view_module=multi_view_module,
        multi_view_collect=multi_view_collect,
        bevfusion_pick_keys=bevfusion_pick_keys,
        bev_fusion_module=dict(
            type="BEVStageTwoModule",
            bevfusion=bev_fusion[mode],
            temporal_fusion=temporal_fusion[mode],
            bev_fusion_upsample=bev_fusion_upsampling,
            backbone=bev_backbone,
            neck=bev_neck,
            head=bev_head,
        ),
    )

    return model


e2e_tracker_head = copy.deepcopy(e2e_head)
e2e_tracker_head["head"]["compile_model"] = True
e2e_tracker_head["head"]["feature_name"] = "bev_head_input_frame"
e2e_tracker_head["postprocess"] = dict(
    type="MultiInputSequential",
    modules=[
        # AddDesc should be the last module
        dict(
            type="AddDesc",
            per_tensor_desc=get_bev_desc(
                use_psc_rot=use_psc_rot,
                N_steps_PSC_rot=N_steps_PSC_rot,
            ),
        ),
    ],
)

# -------------------------- SOLVER --------------------------
# ---------------------- Training --------------------
e2e_dynamic_inputs = dict(
    motr_targets=[
        {
            "bev_tracking": [
                {
                    "obj_idxes": torch.ones(100, dtype=torch.int64) * -99,
                    "labels": torch.zeros(100, dtype=torch.int64),
                    "boxes": torch.rand((100, 4), dtype=torch.float32),
                    "yaws": torch.rand((100, 3), dtype=torch.float32),
                    "bev_loc_z": torch.rand(100, dtype=torch.float32),
                    "heights": torch.rand(100, dtype=torch.float32),
                    "scores": torch.rand(100, dtype=torch.float32),
                }
            ]
        }
    ],
    veh_gt={
        "gt_bev_3d": {
            "bev3d_hm": torch.randn(1, 1, *ipm_output_size),
            "bev3d_ignore_mask": torch.randn(1, 1, *ipm_output_size),
            "bev3d_weight_hm": torch.zeros(1, 1, *ipm_output_size),
            "bev3d_background_weight": torch.zeros(1, 1, *ipm_output_size),
            "bev3d_roi_weight": torch.zeros(1, 1, *ipm_output_size),
            "bev3d_ct_offset": torch.randn(1, 2, *ipm_output_size),
        }
    },
    vru_gt={
        "gt_bev_3d": {
            "bev3d_hm": torch.randn(1, 2, *vru_ipm_output_size),
            "bev3d_ignore_mask": torch.randn(1, 2, *vru_ipm_output_size),
            "bev3d_weight_hm": torch.zeros(1, 2, *vru_ipm_output_size),
            "bev3d_background_weight": torch.zeros(1, 2, *vru_ipm_output_size),
            "bev3d_roi_weight": torch.zeros(1, 2, *vru_ipm_output_size),
            "bev3d_ct_offset": torch.randn(1, 2, *vru_ipm_output_size),
        }
    },
    temporal_clr_flag=0,
    odo_info=torch.rand(1, 12, 3),
    timestamp=[torch.zeros(1, 1)],
)


inputs = dict(
    train=e2e_dynamic_inputs,
    val={},
    deploy={},
)

num_all_queries = e2e_query_kwargs["num_track_queries"] + num_det_queries

deploy_stage2_trackernet_inputs = dict(
    bev_head_input_frame=[], beve2e_trackernet_inputs=[]
)
track_query = torch.rand(
    1,
    1,
    e2e_query_kwargs["num_track_queries"],
    e2e_transformer_params["embedding_dim"] * 2,
)
output_embedding = torch.rand(
    (
        1,
        1,
        e2e_query_kwargs["num_track_queries"],
        e2e_transformer_params["embedding_dim"],
    ),
    dtype=torch.float32,
)
all_ref_pts = torch.rand(1, 1, num_all_queries, 2)

mem_bank = torch.zeros(
    (
        1,
        num_all_queries,
        memory_bank["memory_bank_len"],
        e2e_transformer_params["embedding_dim"],
    ),
    dtype=torch.float32,
)
mem_padding_mask = torch.zeros(
    (1, num_all_queries, memory_bank["memory_bank_len"], 1),
    dtype=torch.float32,
)

odo_input = torch.rand(1, 1, memory_bank["memory_bank_len"] + 1, 4)
odo_input[..., 2] = torch.sin(odo_input[..., 2])
odo_input[..., 3] = torch.cos(odo_input[..., 2])

active_mask_for_fix_track_query = torch.ones(
    (1, 1, e2e_query_kwargs["num_track_queries"], 1), dtype=torch.float32
)
# fps queue使用的场景包含几种：
# 1、在memory bank中对历史embedding（长度为memory_bank_len）与当前帧embedding（长度为1）
# 进行编码，需要历史memory_bank_len + 1帧的fps信息。
# 2、根据自车与障碍物的位置计算插值得到的速度，历史位置长度为memory_bank_len，两两帧计算
# 速度，需要memory_bank_len - 1帧的fps信息。
# 综上，fps queue的长度应该为memory_bank_len + 1，满足各个场景需求。
fps_queue = 0.1 * torch.ones(
    (1, memory_bank["memory_bank_len"] + 1, 1, 1), dtype=torch.float32
)
hisxy_2_lcf = torch.ones(
    (1, num_all_queries, memory_bank["memory_bank_len"], 2),
    dtype=torch.float32,
)

heatmap_sampling_grids = torch.rand(1, 1, num_det_queries, 2)
ref_pts_sigmoid = torch.rand(1, 1, num_det_queries, 2)
det_active_mask = torch.rand(1, 1, num_det_queries, 1)

for idx in range(5):
    deploy_stage2_trackernet_inputs["bev_head_input_frame"].append(
        torch.rand(
            1,
            bev_neck["out_stride2channels"][bev_neck["out_strides"][idx]],
            bevfusion_output_size[0] // bev_neck["out_strides"][idx],
            bevfusion_output_size[1] // bev_neck["out_strides"][idx],
        )
    )

deploy_stage2_trackernet_inputs["beve2e_trackernet_inputs"] = [
    track_query,  # (1, 1, 60, 512)
    output_embedding,  # (1, 1, 60, 256)
    active_mask_for_fix_track_query,  # (1, 1, 60, 1)
    all_ref_pts,  # (1, 1, 360, 2)
    mem_bank,  # (1, 360, 10, 256)
    mem_padding_mask,  # (1, 360, 10, 1)
    odo_input,  # (1, 1, 11, 4)
    fps_queue,  # [1, 11, 1, 1]
    hisxy_2_lcf,  # [1, 360, 10, 2],
    heatmap_sampling_grids,  # [1, 1, 300, 2]
    det_active_mask,  # [1, 1, 300, 1]
]


# -------------------------- TRAIN METRIC --------------------------
def get_metrics_patterns(task_name):
    metrics = [
        dict(type="LossShow", name=f"{task_name}_loss_ce"),
        dict(type="LossShow", name=f"{task_name}_loss_xy"),
        dict(type="LossShow", name=f"{task_name}_loss_wh"),
        dict(type="LossShow", name=f"{task_name}_loss_ciou"),
        dict(type="LossShow", name=f"{task_name}_loss_yaw"),
        dict(type="LossShow", name=f"{task_name}_loss_zheight"),
        dict(type="LossShow", name=f"{task_name}_vru_loss_hm"),
        dict(type="LossShow", name=f"{task_name}_veh_loss_hm"),
        dict(type="LossShow", name=f"{task_name}_vru_loss_ct_offset"),
        dict(type="LossShow", name=f"{task_name}_veh_loss_ct_offset"),
    ]

    per_metric_patterns = [  # corresponding to metrics
        dict(label_pattern=None, pred_pattern=f"^.*{task_name}.*_loss_ce$"),
        dict(label_pattern=None, pred_pattern=f"^.*{task_name}.*loss_xy$"),
        dict(label_pattern=None, pred_pattern=f"^.*{task_name}.*loss_wh$"),
        dict(label_pattern=None, pred_pattern=f"^.*{task_name}.*loss_ciou$"),
        dict(label_pattern=None, pred_pattern=f"^.*{task_name}.*loss_yaw$"),
        dict(
            label_pattern=None, pred_pattern=f"^.*{task_name}.*loss_zheight$"
        ),
        dict(label_pattern=None, pred_pattern=".*e2e.*vru.*_hm_loss$"),
        dict(label_pattern=None, pred_pattern=".*e2e.*vehicle.*_hm_loss$"),
        dict(label_pattern=None, pred_pattern=".*e2e.*vru.*_ct_offset_loss$"),
        dict(
            label_pattern=None, pred_pattern=".*e2e.*vehicle.*_ct_offset_loss$"
        ),
    ]
    if e2e_out_velocity:
        metrics.extend(
            [
                dict(type="LossShow", name=f"{task_name}_loss_velocity"),
                dict(type="LossShow", name=f"{task_name}_trackloss_ce"),
                dict(
                    type="LossShow",
                    name=f"{task_name}_loss_velo_yaw_consistency",
                ),
            ]
        )
        per_metric_patterns.extend(
            [
                dict(
                    label_pattern=None,
                    pred_pattern=f"^.*{task_name}.*_loss_velocity$",
                ),
                dict(
                    label_pattern=None,
                    pred_pattern=f"^.*{task_name}.*_trackloss_ce$",
                ),
                dict(
                    label_pattern=None,
                    pred_pattern=f"^.*{task_name}.*_loss_velo_yaw_consistency$",  # noqa
                ),
            ]
        )
    if e2e_out_trajectory:
        metrics.extend(
            [
                dict(type="LossShow", name=f"{task_name}_loss_trajprob"),
                dict(type="LossShow", name=f"{task_name}_loss_trajreg"),
            ]
        )
        per_metric_patterns.extend(
            [
                dict(
                    label_pattern=None,
                    pred_pattern=f"^.*{task_name}.*_loss_trajprob$",
                ),
                dict(
                    label_pattern=None,
                    pred_pattern=f"^.*{task_name}.*_loss_trajreg$",
                ),
            ]
        )

    return metrics, per_metric_patterns


metrics, per_metric_patterns = get_metrics_patterns(task_name)
metric_updater = get_metric_updater(metrics, per_metric_patterns, task_name)

# ------------------------- VALIDATION SETTING-----------------------
vis_setting["camera_view_names"] = camera_view_names

val_metrics = ["dx", "dxp", "dy", "dyp", "drot", "drot_evs", "dl", "dw", "dh"]
if e2e_out_velocity:
    # dvx, dvy: 所有匹配上的目标的速度误差
    # dvx_0s_evs, dvy_0s_evs: 第一帧匹配上的目标的速度误差
    # dvx_3s_evs, dvy_3s_evs: 前三秒匹配上的目标的速度误差
    val_metrics += [
        "dvx",
        "dvy",
        "dvx_0s_evs",
        "dvx_3s_evs",
        "dvy_0s_evs",
        "dvy_3s_evs",
    ]
val_metric_updater = dict(
    type="MetricUpdater",
    metrics=[
        dict(
            type="BEVE2Evalv2",
            result_prefix=task_name,
            eval_category_ids=eval_setting["e2e_dynamic"]["eval_category_ids"],
            score_threshold=eval_setting["e2e_dynamic"]["score_threshold"],
            iou_threshold=eval_setting["e2e_dynamic"]["iou_threshold"],
            gt_max_depth=eval_setting["e2e_dynamic"]["gt_max_depth"],
            metrics=val_metrics,
            classes=["car", "bicycle", "pedestrian"],
            depth_intervals=eval_setting["e2e_dynamic"]["depth_intervals"],
            eval_mode=eval_setting["e2e_dynamic"]["eval_mode"],
            let_iou_param=eval_setting["e2e_dynamic"]["let_iou_param"],
            save_vis_dir=save_vis_dir,
            save_path=save_eval_results,
            e2e_eval_tracking=e2e_eval_tracking,
            e2e_eval_velocity=e2e_out_velocity,
            e2e_eval_trajectory=e2e_out_trajectory,
            e2e_submit_evs=e2e_submit_evs,
            vcs_range=vcs_range,
            e2e_predefined_classes=e2e_predefined_classes,
            prcurv_save_path=save_eval_results,
            vis_setting=vis_setting,
            evs_setting=evs_setting,
            filter_vcs_range=filter_vcs_range,
            task_name=task_name,
            decode_rot_setting=decode_rot_setting,
            time_delta=TIME_DELTA,
        )
    ],
    metric_update_func=val_metric_update_func,
    step_log_freq=-1,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)
val_metric_updater_list = [val_metric_updater]

# --------------------------- PACK VIS ----------------------------
visualize_callbacks = [
    dict(
        type="ANCBevE2EVisualizeV2",
        output_dir=os.path.join(save_prefix, "visualize", "e2e_dynamic"),
        prefix="bev_stage2_e2e_dynamic_head_predict",
        bev_size=ipm_output_size,
        vcs_range=vcs_range,
        extra_img=dict(
            type="ANCBEVImgStitcher",
            per_extra_img_size=(960, 512),
            camera_view_names=camera_view_names,
            camera_layouts=None,
            is_bev_horizon=True,
        ),
        vis_velocity=True,
        vis_trajectory=True,
        camera_view_names=camera_view_names,
    )
]
