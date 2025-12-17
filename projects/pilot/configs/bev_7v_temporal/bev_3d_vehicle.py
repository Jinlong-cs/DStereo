import copy
import json
import os
from collections import OrderedDict, defaultdict

import numpy as np

from hat.utils.apply_func import _as_list
from hat.utils.config import Config
from hat.utils.filesystem import join_path
from projects.pilot.configs.bev_7v_temporal.base import (
    convert_to_split_dataloader,
    get_bev_3d_transforms,
    get_data_dict,
    get_dataloader,
    get_dataloader_list,
    get_dataset_list,
    get_datasets,
    get_homo_transforms,
    get_roi_resize_cfg,
    get_template_dataset,
    get_update_bev_dataset_func,
    get_update_metric_func,
    reformat_compile_vcs_range,
    set_data_dict_value,
    transform_bool2int,
)
from projects.pilot.configs.bev_7v_temporal.bev_3d_base import (
    bev_3d_common_transforms,
    bev_3d_out_size,
    bev_3d_stage2_output_resolution,
    get_bev3d_tb_update_func,
    get_inputs,
    get_metric_updater,
    get_metrics_patterns,
    get_val_metrics,
    load_data_types,
    max_objs,
    update_dataset_for_return_all,
    update_val_dataloaders_for_return_all,
    use_distorted_offset,
)
from projects.pilot.configs.bev_7v_temporal.common import (
    H_persp_view_scale,
    backbone,
    bev_backbone,
    bev_fusion,
    bev_fusion_upsampling,
    bev_neck,
    bevfusion_output_size,
    bn_kwargs,
    bucket_root,
    cal_homo_offset_on_gpu,
    camera_view_names,
    deploy_head,
    deploy_mode,
    deploy_narrow_head,
    deploy_side_head,
    deploy_stage2_inputs_key,
    ego_ignore_range,
    front_camera_view_names,
    head,
    img_ori_size,
    img_resize_wh_size,
    length_of_clip,
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
    train_batch_size_per_gpu,
    train_global_sample_interval,
    train_num_frames_per_iter,
    train_num_workers,
    training_step,
    use_split_dataloader,
    val_batch_size_per_gpu,
    val_num_frames_per_iter,
    val_num_workers,
    vcs_origin_coord,
    vcs_plane_heights,
    vcs_range,
)

cfg_dir = os.path.dirname(__file__)
# -------------------------- Output -------------------------
val_return_latest_flag = False
val_use_clip_sampler = True
train_return_latest_flag = False
# -------------------------- TASK ---------------------------
task_name = "bev_3d_vehicle"
object_type = "vehicle"

task_out_size = (352, 256)

bev_batch_size = train_batch_size_per_gpu
enable_tensorboard = True
use_ignore_mask = True
enable_vehicle_cls = True
use_category_decouple = True
use_category_bce = True

if use_category_decouple:
    assert enable_vehicle_cls is True
if use_category_bce:
    assert use_category_decouple is True

category2id_map = {
    "Vehicle": 0,
    "Car": 0,
    "Bus": 0,
    "Truck": 0,
    "Tricycle": 0,
    "Construction": 0,
    "Special_vehicle": 0,
    "Tiny_car": 0,
    "Lorry": 0,
    "MiniVan": 0,
    "Sedan_Car": 0,
    "SUV": 0,
    "BigTruck": 0,
    "Motor-Tricycle": 0,
    "Flatbed_Trucks": 0,
    "Car_transporter": 0,
    "Tank_truck": 0,
    "Garbage_truck": 0,
    "Digger": 0,
    "Loader": 0,
    "Blur": 0,
    "Pedestrian": -99,
    "Cyclist": -99,
    "Other": -99,
}

cls_hm_kernel = {0: 13}
roi_label_seq = []
enable_roi_label_seq = []
if enable_vehicle_cls:
    # 车型识别类别：小中型轿车、大巴、卡车、三轮车、工程车、特种车辆、迷你轿车、小货车、SUV型轿车
    category2id_map = {
        "Vehicle": 0,
        "Car": 0,
        "Bus": 1,
        "Truck": 2,
        "Tricycle": 3,
        "Construction": 4,
        "Special_vehicle": 4,
        "Tiny_car": 5,
        "Lorry": 6,
        "MiniVan": 0,
        "Small_Medium_Car": 0,
        # 下面是4D GT链路生产的数据中新增的更多详细的车型类别，包括：
        # 小轿车、SUV、大卡车、电动三轮车、平板货车、汽车运输车、油罐车、垃圾车、挖掘机、装载机
        "Sedan_Car": 0,
        "SUV": 0,
        "BigTruck": 2,
        "Motor-Tricycle": 3,
        "Flatbed_Trucks": 2,
        "Car_transporter": 2,
        "Tank_truck": 4,
        "Garbage_truck": 4,
        "Digger": 4,
        "Loader": 4,
        "Pedestrian": -99,
        "Cyclist": -99,
        "Other": -99,
    }
    # cid: category id
    id2label = {
        0: "Car",
        1: "Bus",
        2: "Truck",
        3: "Tricycle",
        4: "SpecialVehicle",
        5: "TinyCar",
        6: "Lorry",
        "all": "vehicle",
    }
    category_class_weight = {
        0: 1.0,
        1: 1.0,
        2: 1.0,
        3: 1.0,
        4: 1.5,
        5: 1.5,
        6: 1.5,
    }
    length_limitation = {
        0: (0.0, np.inf),
        1: (0.0, np.inf),
        2: (6.0, np.inf),
        3: (0.0, np.inf),
        4: (0.0, np.inf),
        5: (0.0, 4.0),
        6: (0.0, 5.8),
    }

    roi_label_seq = [
        "Car",
        "Bus",
        "Truck",
        "Tricycle",
        "Construction",
        "Special_vehicle",
        "Tiny_car",
        "Lorry",
        "MiniVan",
        "Small_Medium_Car",
        "Sedan_Car",
        "SUV",
        "BigTruck",
        "Motor-Tricycle",
        "Flatbed_Trucks",
        "Car_transporter",
        "Tank_truck",
        "Garbage_truck",
        "Digger",
        "Loader",
    ]

    enable_roi_label_seq = [k for k, v in category2id_map.items() if v != -99]
    cls_hm_kernel = {
        0: 13,
        1: 13,
        2: 13,
        3: 13,
        4: 13,
        5: 13,
        6: 13,
    }


# ------------------ DATASET SETTING -------------------
# data from file, copy from SD
val_extra_config = {
    "v0_3_bev_3d_7v_all_range_main_temporal_val_sampled": {
        "length_of_clip": 128,
        "reverse_select": True,
    },
    "v1_1_bev_3d_all_range_aeb_temporal_val_rm_ovlp_sampled": {
        "length_of_clip": 128,
        "reverse_select": True,
    },
    "v3_0_bev_3d_7v_all_range_main_seq_val_sampled": {
        "length_of_clip": 128,
        "reverse_select": True,
    },
    "v3_0_bev_3d_7v_all_range_aeb_seq_val_sampled": {
        "length_of_clip": 128,
        "reverse_select": True,
    },
    "v3_2_0_aeb_test_site_val": {
        "length_of_clip": 16,
        "reverse_select": True,
    },
    "v3_0_bev_3d_7v_all_range_main_seq_val_sampled_egomotion": {
        "length_of_clip": 128,
        "reverse_select": True,
    },
    "v3_0_bev_3d_7v_all_range_aeb_seq_val_sampled_egomotion": {
        "length_of_clip": 128,
        "reverse_select": True,
    },
    "v2_0_1_bev_3d_7v_all_range_cutin_seq_val_sampled_egomotion": {
        "length_of_clip": 128,
        "reverse_select": True,
    },
}
train_data_version = "v3_2_0_bev_3d_7v_wide_main_seq"
val_data_version = [
    # "v0_3_bev_3d_7v_all_range_main_temporal_val_sampled",
    # "v1_1_bev_3d_all_range_aeb_temporal_val_rm_ovlp_sampled",
    "v3_0_bev_3d_7v_all_range_main_seq_val_sampled_egomotion",
    "v3_0_bev_3d_7v_all_range_aeb_seq_val_sampled_egomotion",
    "v2_0_1_bev_3d_7v_all_range_cutin_seq_val_sampled_egomotion",
    "v3_2_0_aeb_test_site_val",
]
if pipeline_test:
    train_data_version = "pipeline_test"
    val_data_version = "pipeline_test"

dataset_dir = os.path.join(
    f"{os.path.dirname(__file__)}", "..", "datasets", "bev_3d"
)

# transform
num_classes = 7
roi_vcs_range = copy.deepcopy(vcs_range)

cls_dimension = np.array([[3.535518, 2.735504, 11.745526]])  # Vehicle
cls_dimension = np.tile(cls_dimension, (num_classes, 1))
# cls_dimension = np.array(
#     [
#         [1.6433004, 1.7866468, 4.390046],
#         [3.1429198, 2.6314278, 10.085228],
#         [3.535518, 2.735504, 11.745526],
#         [1.5879761, 1.258485, 3.0413513],
#         [3.024004, 2.3344283, 8.748539],
#         [1.6563833, 1.2918538, 2.7280383],
#         [1.9004365, 1.7583425, 4.556895],
#     ]
# )

use_occlusion_attribute = True
if use_occlusion_attribute:
    occlusion_attribute_dict = {
        "full_visible": 0,
        "occluded": 1,
        "heavily_occluded": 2,
        "invisible": 3,
    }
occlusion_ignore_id = -99

use_psc_rot = True
use_ignore_mask_img = True

N_steps_PSC_rot = 3
roi_background_weight_cfg = {
    #  [bottom, right, top, left]
    "roi_range": [vcs_range[0], -10, vcs_range[2], 15],
    "weight": 1.5,
}
background_reweight_cfg = {"cyclist": {"kernel": 7, "weight": 10}}
roi_weight_cfg = {
    #  [bottom, right, top, left]
    "roi_range": [-12, -12, 16, 12],
    "weight": 1.5,
}

bev3d_rpy_transforms = None
rpy_pers_aug = True
if rpy_pers_aug:
    bev3d_rpy_transforms = dict(
        type="ANCSetRPYAugParam",
        prob=0.4,
        aug_degree=1,
    )

# bigobj_hm_kernel_cfg = None
bigobj_hm_kernel_cfg = {
    "length_thresh": 9.0,
    "kernel_size": 13,
    "weight": 2.0,
    "rot_reweight": True,
}
bigobj_length_thresh_cfg = {
    1: 0.0,
    2: 0.0,
}

bev3d_target = dict(
    type="ANCBev3dTargetGenerator",
    num_classes=num_classes,
    max_objs=max_objs,
    bev_size=bev_3d_out_size,
    vcs_range=roi_vcs_range,
    cls_dimension=cls_dimension,
    cls_hm_kernel=cls_hm_kernel,
    bigobj_hm_kernel_cfg=bigobj_hm_kernel_cfg,
    category2id_map=category2id_map,
    enable_ignore=use_ignore_mask,
    ego_ignore_range=ego_ignore_range,
    use_category_decouple=use_category_decouple,
    roi_label_seq=roi_label_seq,
    enable_roi_label_seq=enable_roi_label_seq,
    use_occlusion_attribute=use_occlusion_attribute,
    occlusion_attribute_seq=list(occlusion_attribute_dict.keys())
    if use_occlusion_attribute
    else None,
    occlusion_ignore_id=occlusion_ignore_id,
    category_class_weight=category_class_weight,
    length_limitation=length_limitation,
    use_psc_rot=use_psc_rot,
    N_steps_PSC_rot=N_steps_PSC_rot,
    roi_background_weight_cfg=roi_background_weight_cfg,
    roi_weight_cfg=roi_weight_cfg,
    background_reweight_cfg=background_reweight_cfg,
    bigobj_length_thresh_cfg=bigobj_length_thresh_cfg,
)

bev_3d_common_transforms = copy.deepcopy(bev_3d_common_transforms)
bev_3d_transforms = get_bev_3d_transforms(
    [bev3d_target],
    bev_3d_common_transforms,
    load_data_types=load_data_types,
    task_name=task_name,
    bev3d_rpy_transforms=bev3d_rpy_transforms,
    use_occlusion_attribute=use_occlusion_attribute,
    occlusion_attribute_dict=occlusion_attribute_dict
    if use_occlusion_attribute
    else None,
    use_ignore_mask_img=use_ignore_mask_img,
    temporal_bev=True,
)

for transform in bev_3d_transforms:
    if transform["type"] == "ANCCollect3DV":
        transform["load_data_types"].append("bev3d_vehicle_pose")
        break

train_homo_noise = None
set_homo_noise = False
if set_homo_noise:
    train_homo_noise = {
        "noise_value": (0.2, 0.2, 0.2, 0.0, 0.0, 0.04),
        # It is the exact value of generated noise when the noise type is
        # 'specific_cam'. When the noise type is 'random_cam' or 'random_vcs',
        # it is the upper bound value of generated random noises. The format is
        # (roll, pitch, yaw, x, y, z), the units are degrees and meters.
        "noise_type": "random_cam",
        "noise_view_names": camera_view_names,
    }

homo_transforms = get_homo_transforms(
    transforms_list=bev_3d_transforms, camera_view_names=camera_view_names
)

template_dataset = get_template_dataset(
    img_load_size=list(img_resize_wh_size.values()),
    camera_view_names=camera_view_names,
    per_view_shape=img_ori_size,
    transforms=bev_3d_transforms,
    homo_transforms=homo_transforms,
    spatial_resolution=spatial_resolution,
    H_persp_view_scale=H_persp_view_scale,
    vcs_range=vcs_range,
    use_distorted_offset=use_distorted_offset,
    vcs_plane_heights=vcs_plane_heights,
    cal_homo_offset_on_gpu=cal_homo_offset_on_gpu,
    offset_save_path=offset_save_path,
    temporal_bev=True,
    length_of_clip=length_of_clip,
    train_num_frames_per_iter=train_num_frames_per_iter,
)

update_bev_dataset = get_update_bev_dataset_func(
    added_info=[
        "bev_3d_lmdb_path",
        "multi_view_lmdb_path",
        "homo_path",
        "calib_path",
        "camera_module_type",
        "homo_noise",
        "tag_info_lmdb_path",
        "odometry_file_path",
        "num_max_frames",
        "reverse_select",
        "object_tag_infos_lmdb_path",
    ],
    update_trans_info={
        "ANCCollect3DV": "fill_fake_temporal_data",
        "ANCBev3dTargetGenerator": ["filter_vcs_range", "ignore_obj_cls"],
    },
)

val_homo_noise = None
set_val_homo_noise = False
if set_val_homo_noise:
    val_homo_noise = {
        "noise_value": (0.5, 0.5, 0.5, 0.0, 0.0, 0.04),
        # It is the exact value of generated noise when the noise type is
        # 'specific_cam'. When the noise type is 'random_cam' or 'random_vcs',
        # it is the upper bound value of generated random noises. The format is
        # (roll, pitch, yaw, x, y, z), the units are degrees and meters.
        "noise_type": "random_cam",
        "noise_view_names": camera_view_names,
    }

val_template_dataset = copy.deepcopy(template_dataset)
for transform in val_template_dataset["transforms"]:
    if transform["type"] == "ANCApplyMaskOnImg":
        val_template_dataset["transforms"].remove(transform)
    if transform["type"] == "ANCMultiViewTargetGenerator":
        transform["use_ignore_mask_img"] = False

val_template_dataset["num_frames_per_iter"] = val_num_frames_per_iter
val_template_dataset["transforms"][0]["img_idxs"] = list(
    range(val_num_frames_per_iter)
)
val_template_dataset["transforms"][0]["pose_idxs"] = list(
    range(val_num_frames_per_iter + 1)
)
if not val_return_latest_flag:
    update_dataset_for_return_all(
        val_template_dataset, val_num_frames_per_iter
    )
if not train_return_latest_flag:
    update_dataset_for_return_all(template_dataset, train_num_frames_per_iter)
# ----------------------- DATALODER ---------------------------
data_yaml = os.path.join(dataset_dir, f"{task_name}_dataset.yaml")


def get_train_dataloader():
    url = os.path.join(dataset_dir, f"{task_name}_train_version.py")
    train_dataset_info = Config.fromfile(url)
    train_dataset_dict = train_dataset_info[train_data_version]

    data_dict = get_data_dict(data_yaml)
    data_dict = join_path(
        bucket_root, data_dict, ["camera_module_type", "camera_view_names"]
    )
    set_data_dict_value(data_dict, "num_max_frames", length_of_clip)
    set_data_dict_value(data_dict, "reverse_select", False)
    train_datasets = get_datasets(
        train_dataset_dict,
        None,
        template_dataset,
        data_dict,
        update_bev_dataset,
        homo_noise=train_homo_noise,
        global_sample_interval=train_global_sample_interval,
    )
    data_loader = get_dataloader(
        train_datasets,
        train_num_workers,
        train_batch_size_per_gpu,
        shuffle=True,
        persistent_workers=train_num_workers > 0,
    )

    if use_split_dataloader:
        data_loader = convert_to_split_dataloader(
            data_loader, temporal_bev=True
        )
    return data_loader


def get_val_dataloader():
    val_url = os.path.join(dataset_dir, f"{task_name}_val_version.yaml")
    val_data_version_dict = get_data_dict(val_url)
    val_dataset_dict = {
        val_version: val_data_version_dict[val_version]
        for val_version in _as_list(val_data_version)
    }
    data_dict = get_data_dict(data_yaml)
    data_dict = join_path(
        bucket_root, data_dict, ["camera_module_type", "camera_view_names"]
    )
    set_data_dict_value(data_dict, "num_max_frames", length_of_clip)
    set_data_dict_value(data_dict, "reverse_select", False)
    for version in val_extra_config:
        if version in val_dataset_dict:
            dataset_dict = val_dataset_dict[version]
            set_data_dict_value(
                dataset_dict,
                "num_max_frames",
                val_extra_config[version].get(
                    "length_of_clip", length_of_clip
                ),
            )
            set_data_dict_value(
                dataset_dict,
                "reverse_select",
                val_extra_config[version].get("reverse_select", False),
            )
    val_dataset_list = get_dataset_list(
        list(val_dataset_dict.values()),
        None,
        val_template_dataset,
        data_dict,
        update_bev_dataset,
        homo_noise=val_homo_noise if set_val_homo_noise else None,
        repeat_dataset_times=3 if set_val_homo_noise else 1,
    )
    val_data_loader_list = get_dataloader_list(
        val_dataset_list, val_num_workers, val_batch_size_per_gpu, False, False
    )
    if not val_return_latest_flag or val_use_clip_sampler:
        all_length_of_clip = [
            val_extra_config.get(version, {}).get(
                "length_of_clip", length_of_clip
            )
            for version in val_data_version
        ]
        update_val_dataloaders_for_return_all(
            val_data_loader_list,
            val_batch_size_per_gpu,
            val_num_frames_per_iter,
            all_length_of_clip,
        )

    return val_data_loader_list


# -------------------------- MODEL --------------------------
train_inputs, val_inputs, deploy_inputs = get_inputs(
    task_out_size, num_classes=num_classes
)
inputs = dict(
    train=train_inputs,
    val=val_inputs,
    deploy=deploy_inputs,
)

head_channels = OrderedDict(
    bev3d_hm=num_classes,
    bev3d_dim=3,  # h, w, l
    bev3d_rot=N_steps_PSC_rot if use_psc_rot else 2,
    bev3d_ct_offset=2,  # x, y
    bev3d_loc_z=1,  # vcs z axis
)

if use_category_decouple:
    head_channels["bev3d_hm"] = 1
    head_channels["bev3d_cls_hm"] = num_classes

if use_occlusion_attribute:
    head_channels["bev3d_occlusion_hm"] = 4


roi_resize_cfg = None
roi_resize = None

# example, refer if need roi resize
# roi_vcs_range = (-31.6, -32.0, 51.6, 32.0)
# roi_out_size = (208, 160)
# roi_resize_cfg = dict(
#     in_stride=2,
#     roi_vcs_range=roi_vcs_range,
#     out_size=(208, 160),
# )
if roi_resize_cfg is not None:
    roi_resize_cfg = get_roi_resize_cfg(
        input_size=bevfusion_output_size,
        in_stride=roi_resize_cfg["in_stride"],
        output_size=roi_resize_cfg["out_size"],
        ori_vcs_range=vcs_range,
    )
    roi_resize = dict(
        type="RoiResize",
        in_strides=[2, 4, 8, 16, 32],
        roi_resize_cfgs=roi_resize_cfg,
    )

eps_channels = None
quant_config = {}
if enable_vehicle_cls:
    eps_channels = 1
    if use_category_decouple:
        quant_config = {"bev3d_cls_hm": "qint16"}
    else:
        quant_config = {"bev3d_hm": "qint16"}
if use_occlusion_attribute:
    quant_config.update({"bev3d_occlusion_hm": "qint16"})

use_maxpool = True
use_norm_rot = True
max_pool_kernel = 3

# defalut val for eval
score_threshold = 0.0
roi_score_threshold = [0.0] * num_classes

# defined val
score_threshold = model_thresh.get(task_name, {}).get(
    "score_threshold", score_threshold
)
roi_score_threshold = model_thresh.get(task_name, {}).get(
    "roi_score_threshold", roi_score_threshold
)

roi_filter_info = None
if deploy_mode:
    roi_filter_vcs_range = (-20.0, -8.0, 50.0, 8.0)
    roi_filter_info = dict(
        roi_vcs_range=reformat_compile_vcs_range(roi_filter_vcs_range),
        roi_score_threshold=roi_score_threshold,
    )


nms_setting = {
    "letnms": [
        {
            "p_t": 0.35,
            "min_t": 4.0,
            "max_t": 8.0,
            "radius": 1.6,
            "e_loc_threshold": 0.3,
            "angle_threshold": 0.3,
            "area_threshold": 0.4,
            "ct_nms_param": {
                "scale_l": 0.6,
                "scale_w": 0.6,
                "use_yaw_filter": False,
                "yaw_threshold": 0.1,
                "use_mutual_ctnms": True,
            },
        },
        {
            "p_t": 0.35,
            "min_t": 4.0,
            "max_t": 12.0,
            "radius": 3.0,
            "e_loc_threshold": 0.4,
            "angle_threshold": 0.3,
            "area_threshold": 0.4,
            "ct_nms_param": {
                "scale_l": 0.6,
                "scale_w": 0.6,
                "use_yaw_filter": False,
                "yaw_threshold": 0.1,
                "use_mutual_ctnms": True,
            },
        },
        {
            "p_t": 0.35,
            "min_t": 4.0,
            "max_t": 12.0,
            "radius": 3.0,
            "e_loc_threshold": 0.4,
            "angle_threshold": 0.3,
            "area_threshold": 0.4,
            "ct_nms_param": {
                "scale_l": 0.6,
                "scale_w": 0.6,
                "use_yaw_filter": False,
                "yaw_threshold": 0.1,
                "use_mutual_ctnms": True,
            },
        },
        {
            "p_t": 0.35,
            "min_t": 4.0,
            "max_t": 8.0,
            "radius": 1.6,
            "e_loc_threshold": 0.3,
            "angle_threshold": 0.3,
            "area_threshold": 0.4,
            "ct_nms_param": {
                "scale_l": 0.6,
                "scale_w": 0.6,
                "use_yaw_filter": False,
                "yaw_threshold": 0.1,
                "use_mutual_ctnms": True,
            },
        },
        {
            "p_t": 0.35,
            "min_t": 4.0,
            "max_t": 12.0,
            "radius": 3.0,
            "e_loc_threshold": 0.4,
            "angle_threshold": 0.3,
            "area_threshold": 0.4,
            "ct_nms_param": {
                "scale_l": 0.6,
                "scale_w": 0.6,
                "use_yaw_filter": False,
                "yaw_threshold": 0.1,
                "use_mutual_ctnms": True,
            },
        },
        {
            "p_t": 0.35,
            "min_t": 4.0,
            "max_t": 8.0,
            "radius": 1.6,
            "e_loc_threshold": 0.3,
            "angle_threshold": 0.3,
            "area_threshold": 0.4,
            "ct_nms_param": {
                "scale_l": 0.6,
                "scale_w": 0.6,
                "use_yaw_filter": False,
                "yaw_threshold": 0.1,
                "use_mutual_ctnms": True,
            },
        },
        {
            "p_t": 0.35,
            "min_t": 4.0,
            "max_t": 12.0,
            "radius": 3.0,
            "e_loc_threshold": 0.4,
            "angle_threshold": 0.3,
            "area_threshold": 0.4,
            "ct_nms_param": {
                "scale_l": 0.6,
                "scale_w": 0.6,
                "use_yaw_filter": False,
                "yaw_threshold": 0.1,
                "use_mutual_ctnms": True,
            },
        },
    ],
    "agnostic": True,
    "topk_deploy": 300,
}

apply_nms_setting = (
    model_thresh.get(task_name, {}).get("nms_setting", nms_setting)
    if model_thresh
    else None
)

loss_weights = {
    "bev3d_hm": 4.0,
    "bev3d_dim": 1.0,
    "bev3d_rot": 1.7,
    "bev3d_ct_offset": 1.5,
    "bev3d_loc_z": 1.5,
    "bev3d_corner": 1.0,
}

if use_category_decouple:
    loss_weights["bev3d_cls_hm"] = 2.0
if use_occlusion_attribute:
    loss_weights["bev3d_occlusion_hm"] = 1.0

# ----------------------------- DESC ---------------------------
bev_common_desc = dict(
    vcs_origin_coord=vcs_origin_coord,
    bev_stage2_input_resolution=spatial_resolution,
    bev_stage2_output_resolution=bev_3d_stage2_output_resolution,
    visible_range=reformat_compile_vcs_range(roi_vcs_range),
    warp_offset_range=reformat_compile_vcs_range(vcs_range),
    vcs_plane_heights=vcs_plane_heights,
)


def get_desc(
    enable_vehicle_cls,
    use_occlusion_attribute,
    use_category_decouple=True,
    use_psc_rot=False,
    N_steps_PSC_rot=None,
    score_thresh=None,
    bev_common_desc=None,
    use_maxpool=True,
    max_pool_kernel=None,
    nms_setting=None,
    roi_filter_info=None,
    use_norm_rot=False,
):
    apply_score_thresh = score_thresh
    apply_common_desc = copy.deepcopy(bev_common_desc)
    apply_use_maxpool = use_maxpool
    apply_max_pool_kernel = max_pool_kernel
    apply_nms_setting = nms_setting
    apply_roi_filter_info = roi_filter_info
    postprocess_desc = {
        "task": "bev_3d_vehicle",
        "output_name": "bev_3d_vehicle_heatmap_output",
        "score_threshold": apply_score_thresh,
        "properties": [{"channel_labels": ["vehicle"]}],
        **apply_common_desc,
    }
    if apply_use_maxpool:
        postprocess_desc["max_pool_kernel"] = apply_max_pool_kernel
    if apply_nms_setting is not None:
        if "letnms" in apply_nms_setting:
            postprocess_desc["use_let_nms"] = 1
            postprocess_desc["topk"] = apply_nms_setting.get(
                "topk_deploy", 120
            )
            letnms_setting = defaultdict(list)
            for setting in apply_nms_setting["letnms"]:
                for param_name, param_value in setting.items():
                    letnms_setting[param_name].append(
                        transform_bool2int(param_value)
                    )
            postprocess_desc.update(letnms_setting)
    if use_psc_rot and N_steps_PSC_rot is not None:
        channel_labels = [f"cos_idx{idx}" for idx in range(N_steps_PSC_rot)]
        postprocess_rot_desc = {
            "task": "bev_3d_vehicle",
            "output_name": "bev_3d_vehicle_rot_output",
            "properties": [{"channel_labels": channel_labels}],
            "rot_mod_threshold": 0.0001,
            "use_norm_rot": use_norm_rot,
            **apply_common_desc,
        }
    else:
        postprocess_rot_desc = {
            "task": "bev_3d_vehicle",
            "output_name": "bev_3d_vehicle_rot_output",
            "properties": [{"channel_labels": ["cos", "sin"]}],
            "use_norm_rot": use_norm_rot,
            **apply_common_desc,
        }
    per_tensor_desc = [
        postprocess_desc,
        {
            "task": "bev_3d_vehicle",
            "output_name": "bev_3d_vehicle_dimension_output",
            "cls_dimension": cls_dimension[0].tolist(),
            "properties": [{"channel_labels": ["height", "width", "length"]}],
            **apply_common_desc,
        },
        postprocess_rot_desc,
        {
            "task": "bev_3d_vehicle",
            "output_name": "bev_3d_vehicle_ct_offset_output",
            "properties": [{"channel_labels": ["offset_x", "offset_y"]}],
            **apply_common_desc,
        },
        {
            "task": "bev_3d_vehicle",
            "output_name": "bev_3d_vehicle_loc_z_output",
            "properties": [{"channel_labels": ["z_value"]}],
            **apply_common_desc,
        },
    ]
    if use_occlusion_attribute:
        per_tensor_desc.append(
            {
                "task": "bev_3d_vehicle",
                "output_name": "bev_3d_vehicle_occlusion_output",
                "properties": [
                    {
                        "channel_labels": [
                            "full_visible",
                            "occluded",
                            "heavily_occluded",
                            "invisible",
                        ]
                    }
                ],
                **apply_common_desc,
            }
        )
    if enable_vehicle_cls:
        per_tensor_desc[0]["properties"] = [{"channel_labels": ["score"]}]
        if not use_category_decouple:
            cls_id_insert_pos = len(per_tensor_desc)
        else:
            cls_id_insert_pos = -1
        per_tensor_desc.insert(
            cls_id_insert_pos,
            {
                "task": "bev_3d_vehicle",
                "output_name": "bev_3d_vehicle_class_id_output",
                "roi_filter_info": apply_roi_filter_info,
                "properties": [
                    {
                        "channel_labels": [
                            "Small_Medium_Car",
                            "Bus",
                            "Trucks",
                            "Motors",
                            "Special_vehicle",
                            "Tiny_car",
                            "Lorry",
                        ]
                    }
                ],
                **apply_common_desc,
            },
        )
    per_tensor_desc = [json.dumps(i) for i in per_tensor_desc]
    return per_tensor_desc


task_head = dict(
    type="ANCBEV3DHead",
    in_strides=[2, 4, 8, 16, 32],
    out_strides=2,
    in_channels=48,
    forward_frame_idx=0,
    head_channels=head_channels,
    use_varg=False,
    use_bias=False,
    bn_kwargs=bn_kwargs,
    dw_with_relu=True,
    pw_with_relu=False,
    factor=2,
    group_base=8,
    merge_block=True,
    dequant_out=True,
    quant_config=quant_config,
    node_name="bev_stage2_3d_vehicle_head",
)

postprocess = []
if enable_vehicle_cls:
    if use_category_decouple:
        cls_hm_argmax = dict(
            type="ArgmaxPostprocess",
            data_name="bev3d_cls_hm",
            dim=1,
            keepdim=True,
            node_name=f"bev_stage2_{task_name}_argmax",
        )
        postprocess.append(cls_hm_argmax)
    else:
        hm_max = dict(
            type="MaxPostProcess",
            data_names=["bev3d_hm"],
            out_names=[["bev3d_hm", "bev3d_cls_hm"]],
            dim=1,
            keepdim=True,
            node_name=f"bev_stage2_{task_name}_max",
        )
        postprocess.append(hm_max)
if use_occlusion_attribute:
    occlusion_hm_argmax = dict(
        type="ArgmaxPostprocess",
        data_name="bev3d_occlusion_hm",
        dim=1,
        keepdim=True,
        node_name=f"bev_stage2_{task_name}_occlusion_argmax",
    )
    postprocess.append(occlusion_hm_argmax)
postprocess.append(
    dict(
        type="ANCBEV3Decoder",
        topk=300,
        use_maxpool=use_maxpool,
        max_pool_kernel=max_pool_kernel,
        cls_dimension=cls_dimension,
        vcs_range=roi_vcs_range,
        bev_3d_out_size=bev_3d_out_size,
        num_classes=num_classes,
        add_hm_eps=False,
        eps_channels=eps_channels,
        nms_setting=nms_setting,
        score_threshold=score_threshold,
        roi_filter_info=roi_filter_info,
        use_category_decouple=use_category_decouple,
        use_psc_rot=use_psc_rot,
        N_steps_PSC_rot=N_steps_PSC_rot,
        use_norm_rot=True,
        node_name=f"bev_stage2_{task_name}_decoder",
    )
)


def get_model(mode):
    bev_head = dict(
        type="OutputModule",
        head=task_head,
        head_parser=None,
        loss=None,
        postprocess=None,
        target=None,
        convert_to_dict=True,
        trace_convert=True,
        prefix="bev_stage2_3d_vehicle_head",
    )
    if mode == "train":
        bev_head["loss"] = dict(
            type="ANCBEV3DLoss",
            loss_weights=loss_weights,
            class_weight=None,
            gamma=1,
            beta=0.33,
            use_category_bce=use_category_bce,
            use_norm_rot=True,
            gt_name="gt_bev_3d",
            use_rot_wing_loss=True,
            node_name=f"bev_stage2_{task_name}_loss",
            return_latest_flag=train_return_latest_flag,
            vcs_range=vcs_range,
            use_psc_rot=use_psc_rot,
            N_steps_PSC_rot=N_steps_PSC_rot,
            cls_dimension=cls_dimension,
        )
    elif mode == "val":
        bev_head["postprocess"] = postprocess
    else:
        bev3d_desc = get_desc(
            enable_vehicle_cls=enable_vehicle_cls,
            use_occlusion_attribute=use_occlusion_attribute,
            use_category_decouple=use_category_decouple,
            use_psc_rot=use_psc_rot,
            N_steps_PSC_rot=N_steps_PSC_rot,
            score_thresh=score_threshold,
            bev_common_desc=bev_common_desc,
            use_maxpool=use_maxpool,
            use_norm_rot=int(use_norm_rot),
            max_pool_kernel=max_pool_kernel,
            nms_setting=apply_nms_setting,
            roi_filter_info=roi_filter_info,
        )
        bev_head["head"]["dequant_out"] = False
        bev_head["convert_to_dict"] = False
        postprocess_modules = []
        if enable_vehicle_cls:
            if use_category_decouple:
                postprocess_modules.append(cls_hm_argmax)
            else:
                postprocess_modules.append(hm_max)
        if use_occlusion_attribute:
            postprocess_modules.append(occlusion_hm_argmax)
        if use_occlusion_attribute or enable_vehicle_cls:
            postprocess_modules.append(
                dict(
                    type="DequantModule",
                    data_names=[
                        "bev3d_hm",
                        "bev3d_dim",
                        "bev3d_rot",
                        "bev3d_ct_offset",
                        "bev3d_loc_z",
                    ],
                    node_name="bev_stage2_3d_vehicle_dequant",
                )
            )
        postprocess_modules.append(
            dict(
                type="AddDesc",
                per_tensor_desc=bev3d_desc,
                node_name="bev_stage2_3d_vehicle_add_desc",
            )
        )
        bev_head["postprocess"] = dict(
            type="MultiInputSequential", modules=postprocess_modules
        )

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


# -------------------------- TRAIN METRIC --------------------------
metrics, per_metric_patterns = get_metrics_patterns(
    task_name,
    use_category_decouple=True,
    use_occlusion_attribute=True,
    use_corner_loss="bev3d_corner" in loss_weights,
)
metric_updater = get_metric_updater(metrics, per_metric_patterns, task_name)

# ------------------------- VALIDATION SETTING-----------------------
visibility_intervals = None
add_category_confusion_eval = True

vis_setting = dict(
    camera_view_names=camera_view_names,
    bev_size=bevfusion_output_size,
    vcs_range=vcs_range,
    score_threshold=0.2,
    project_bbox_to_cameras=True,
    anno_show=True,
    concat_imgs=True,
    draw_lidar=False,
    multi_views_type="bev_7v",
    per_extra_img_size=(960, 512),
)


base_taggers = {
    "YAW |": {
        "gt_key": "vcs_rot_z_",
        "pred_key": "bev3d_rot",
        "mapper": {
            "YAW |": [
                [0, 22.5],
                [360 - 22.5, 360],
                [180 - 22.5, 180 + 22.5],
            ]
        },
        "tagger_fn": "tag_by_range",
        "transforms": lambda rad: np.rad2deg(rad) % 360.0,
    },
    "YAW --": {
        "gt_key": "vcs_rot_z_",
        "pred_key": "bev3d_rot",
        "mapper": {
            "YAW --": [[90 - 22.5, 90 + 22.5], [270 - 22.5, 270 + 22.5]]
        },
        "tagger_fn": "tag_by_range",
        "transforms": lambda rad: np.rad2deg(rad) % 360.0,
    },
    "YAW \\": {
        "gt_key": "vcs_rot_z_",
        "pred_key": "bev3d_rot",
        "mapper": {"YAW \\": [[22.5, 90 - 22.5], [180 + 22.5, 270 - 22.5]]},
        "tagger_fn": "tag_by_range",
        "transforms": lambda rad: np.rad2deg(rad) % 360.0,
    },
    "YAW /": {
        "gt_key": "vcs_rot_z_",
        "pred_key": "bev3d_rot",
        "mapper": {
            "YAW /": [[90 + 22.5, 180 - 22.5], [270 + 22.5, 360 - 22.5]]
        },
        "tagger_fn": "tag_by_range",
        "transforms": lambda rad: np.rad2deg(rad) % 360.0,
    },
    "urban": {
        "gt_key": "scene",
        "pred_key": "scene",
        "mapper": {"ids": ["urban"]},
        "tagger_fn": "tag_by_category",
    },
    "highway": {
        "gt_key": "scene",
        "pred_key": "scene",
        "mapper": {"ids": ["highway"]},
        "tagger_fn": "tag_by_category",
    },
    "rural": {
        "gt_key": "scene",
        "pred_key": "scene",
        "mapper": {"ids": ["rural"]},
        "tagger_fn": "tag_by_category",
    },
    "tunnel": {
        "gt_key": "scene",
        "pred_key": "scene",
        "mapper": {"ids": ["tunnel"]},
        "tagger_fn": "tag_by_category",
    },
    "toll_station": {
        "gt_key": "scene",
        "pred_key": "scene",
        "mapper": {"ids": ["toll_station"]},
        "tagger_fn": "tag_by_category",
    },
    "charge_station": {
        "gt_key": "scene",
        "pred_key": "scene",
        "mapper": {"ids": ["charge_station"]},
        "tagger_fn": "tag_by_category",
    },
    "day": {
        "gt_key": "time",
        "pred_key": "time",
        "mapper": {"ids": ["day"]},
        "tagger_fn": "tag_by_category",
    },
    "night": {
        "gt_key": "time",
        "pred_key": "time",
        "mapper": {"ids": ["night"]},
        "tagger_fn": "tag_by_category",
    },
    "L>10m": {
        "gt_key": "vcs_dim_",
        "pred_key": "bev3d_dim",
        "mapper": {"ids": [[10, 100]]},
        "tagger_fn": "tag_by_range",
        "transforms": lambda dim: dim[2],
    },
    "vcs(-5.0, -5.0, 10.0, 5.0)": {
        "gt_key": "vcs_loc_",
        "pred_key": "bev3d_ct",
        "mapper": {"ids": [[-5.0, -5.0, 10.0, 5.0]]},
        "tagger_fn": "tag_by_vcs",
    },
    "cutin": {
        "gt_key": "tag_cutin",
        "pred_key": "",
        "mapper": {"ids": [True]},
        "tagger_fn": "tag_by_category",
    },
    "xy(0,10)": {
        "gt_key": "vcs_loc_",
        "pred_key": "bev3d_ct",
        "mapper": {
            "range": [
                [0, 10],
            ]
        },
        "tagger_fn": "tag_by_range",
        "transforms": lambda loc: np.linalg.norm(loc[:2]),
    },
    "xy(10,40)": {
        "gt_key": "vcs_loc_",
        "pred_key": "bev3d_ct",
        "mapper": {
            "range": [
                [10, 40],
            ]
        },
        "tagger_fn": "tag_by_range",
        "transforms": lambda loc: np.linalg.norm(loc[:2]),
    },
    "xy(40,100)": {
        "gt_key": "vcs_loc_",
        "pred_key": "bev3d_ct",
        "mapper": {
            "range": [
                [40, 100],
            ]
        },
        "tagger_fn": "tag_by_range",
        "transforms": lambda loc: np.linalg.norm(loc[:2]),
    },
    "cutin_range": {
        "gt_key": "vcs_loc_",
        "pred_key": "bev3d_ct",
        "mapper": {"ids": [[-5.0, -5.0, 15.0, -1.5], [-5.0, 1.5, 15.0, 5.0]]},
        "tagger_fn": "tag_by_vcs",
    },
    "cipv_range": {
        "gt_key": "vcs_loc_",
        "pred_key": "bev3d_ct",
        "mapper": {"ids": [[-5.0, -1.5, 30, 1.5]]},
        "tagger_fn": "tag_by_vcs",
    },
    "Yaw0": {
        "gt_key": "vcs_rot_z_",
        "pred_key": "bev3d_rot",
        "mapper": {
            "YAW |": [
                [0, 5],
                [360 - 5, 360],
                [180 - 5, 180 + 5],
            ]
        },
        "tagger_fn": "tag_by_range",
        "transforms": lambda rad: np.rad2deg(rad) % 360.0,
    },
}
taggers = {
    "YAW |": {"base_tags": ["YAW |"]},
    "YAW --": {"base_tags": ["YAW --"]},
    "YAW /": {"base_tags": ["YAW /"]},
    "YAW \\": {"base_tags": ["YAW \\"]},
    "urban": {"base_tags": ["urban"]},
    "highway": {"base_tags": ["highway"]},
    "rural": {"base_tags": ["rural"]},
    "tunnel": {"base_tags": ["tunnel"]},
    "toll_station": {"base_tags": ["toll_station"]},
    "charge_station": {"base_tags": ["charge_station"]},
    "day": {"base_tags": ["day"]},
    "night": {"base_tags": ["night"]},
    "Close_Long": {"base_tags": ["L>10m", "vcs(-5.0, -5.0, 10.0, 5.0)"]},
    "cutin_range": {"base_tags": ["cutin_range", "YAW |"]},
    "cutin_range_Long": {"base_tags": ["L>10m", "cutin_range", "YAW |"]},
    "CIPV_Long": {"base_tags": ["cipv_range", "Yaw0", "L>10m"]},
}
eval_setting = {
    "eval_category_ids": list(range(num_classes)) + ["all"]
    if enable_vehicle_cls
    else (0,),
    "score_threshold": 0.1,
    "iou_threshold": 0.2,
    "gt_max_depth": 110,
    "depth_intervals": (3, 6, 12, 25, 50),
    "eval_vcs_range": (-30.0, -50.0, 100.0, 50.0),
    "ego_ignore_range": ego_ignore_range,
    "eval_mode": ["bev_iou", "let_iou"],
    "let_iou_param": {"p_t": 0.35, "min_t": 4.0, "max_t": 8.0},
    "base_taggers": base_taggers,
    "taggers": taggers,
    "category_wise_threshold": False,
    "metric_key": (
        "Recall",
        "Precision",
        "dx",
        "dy",
        "drot",
        "dl",
        "dw",
        "cc_dx",  # cc means cutin_corner
        "cc_dy",
        "cc_dx%95",
        "cc_dy%95",
        "cipv_dx",
        "cipv_dy",
        "cipv_dx%95",
        "cipv_dy%95",
    ),
    "auto_threshold": True,
    "distance_wise_mode": "xy",
}
cutin_taggers = copy.deepcopy(taggers)
cutin_taggers.pop("cutin_range")
cutin_taggers.pop("cutin_range_Long")
cutin_taggers["CutIn"] = {"base_tags": ["cutin"]}
cutin_taggers["CutIn_xy(0,10)"] = {"base_tags": ["cutin", "xy(0,10)"]}
cutin_taggers["CutIn_xy(10,40)"] = {"base_tags": ["cutin", "xy(10,40)"]}
cutin_taggers["CutIn_xy(40,100)"] = {"base_tags": ["cutin", "xy(40,100)"]}
cutin_taggers["CutIn_Long"] = {"base_tags": ["cutin", "L>10m"]}
cutin_taggers["CutIn_Long_xy(0,10)"] = {
    "base_tags": ["cutin", "L>10m", "xy(0,10)"]
}
cutin_taggers["CutIn_Long_xy(10,40)"] = {
    "base_tags": ["cutin", "L>10m", "xy(10,40)"]
}
cutin_taggers["CutIn_Long_xy(40,100)"] = {
    "base_tags": ["cutin", "L>10m", "xy(40,100)"]
}
cutin_eval_setting = copy.deepcopy(eval_setting)
cutin_eval_setting.update(
    {
        "taggers": cutin_taggers,
    }
)
val_metric_updater_list = []
for dataset_key in _as_list(val_data_version):
    save_dir = os.path.join(save_prefix, "eval", dataset_key)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_metric_path = os.path.join(
        save_dir, f"{task_name}_{training_step}_eval_metric.json"
    )
    confusion_save_path = None
    # confusion_save_path = save_dir

    # set save_vis_path to open visualize
    save_vis_dir = None
    # save_vis_dir = os.path.join(save_dir, "vis")
    # if os.path.exists(save_vis_dir):
    #     shutil.rmtree(save_vis_dir)
    # os.makedirs(save_vis_dir)

    save_pred = None
    # save_prcuv = None
    # save_pred = os.path.join(save_dir, f"{task_name}_pred")
    save_prcuv = os.path.join(save_dir, f"{task_name}_aps")
    _eval_setting = (
        eval_setting if "cutin" not in dataset_key else cutin_eval_setting
    )
    val_metrics = get_val_metrics(
        task_name,
        **_eval_setting,
        name=f"{task_name}_{dataset_key}_BEV3D",
        save_path=save_pred,
        prcurv_save_path=save_prcuv,
        save_metric_path=save_metric_path,
        use_ignore_mask=use_ignore_mask,
        save_vis_dir=save_vis_dir,
        vis_setting=vis_setting,
        visibility_intervals=visibility_intervals,
        id2label=id2label,
        eval_occlusion=use_occlusion_attribute,
        occlusion_ignore_id=occlusion_ignore_id,
        eval_category_confusion=add_category_confusion_eval,
        confusion_save_path=confusion_save_path,
        num_classes=num_classes,
        result_prefix="wide",
        metric_save_dir=os.path.join(
            save_prefix, "eval", dataset_key, task_name
        ),
    )

    val_metric_patterns = [
        {
            "bev3d_ct": "^.*predict_bev3d_ct",
            "bev3d_cls_id": "^.*predict_bev3d_cls_id",
            "bev3d_score": "^.*predict_bev3d_score",
            "bev3d_rot": "^.*predict_bev3d_rot",
            "bev3d_dim": "^.*predict_bev3d_dim",
            "bev3d_loc_z": "^.*predict_bev3d_loc_z",
            "bev3d_occlusion_id": "^.*predict_bev3d_occlusion_id",  # noqa
        }
    ]
    if add_category_confusion_eval:
        val_confusion_metric_patterns = copy.deepcopy(val_metric_patterns[0])
        val_confusion_metric_patterns.pop("bev3d_occlusion_id")
        val_metric_patterns.append(val_confusion_metric_patterns)

    val_metric_updater = dict(
        type="MetricUpdater",
        metrics=val_metrics,
        metric_update_func=get_update_metric_func(
            "val", task_name, val_metric_patterns
        ),
        step_log_freq=-1,
        epoch_log_freq=1,
        log_prefix=f"Validation_{task_name}_{dataset_key}",
    )
    val_metric_updater_list.append(val_metric_updater)


# -------------------------- TENSORBOAED --------------------------
if enable_tensorboard:
    tb_update_func = get_bev3d_tb_update_func

# --------------------------- PACK VIS ----------------------------
visualize_callbacks = [
    dict(
        type="ANCBev3DVisualizeV2",
        output_dir=os.path.join(save_prefix, "visualize", "bev_3d_vehicle"),
        task="bev_3d_vehicle",
        prefix="bev_stage2_3d_vehicle_head_predict",
        score_threshold=score_threshold,
        bev_size=bevfusion_output_size,
        vcs_range=vcs_range,
        camera_view_names=camera_view_names,
        extra_img=dict(
            type="ANCBEVImgStitcher",
            per_extra_img_size=(960, 512),
            camera_view_names=camera_view_names,
            camera_layouts=None,
        ),
    )
]
