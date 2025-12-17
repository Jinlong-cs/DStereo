import copy
import json
import os
from collections import OrderedDict, defaultdict

import numpy as np

from hat.utils.apply_func import _as_list
from hat.utils.config import Config
from hat.utils.filesystem import join_path
from projects.pilot.configs.bev_7v.base import (
    convert_to_split_dataloader,
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
)
from projects.pilot.configs.bev_7v.bev_3d_base import (
    bev_3d_stage2_output_resolution,
    bev_common_transforms,
    get_bev3d_tb_update_func,
    get_bev_3d_transforms,
    get_inputs,
    get_metric_updater,
    get_metrics_patterns,
    get_val_metrics,
    load_data_types,
    max_objs,
    use_distorted_offset,
)
from projects.pilot.configs.bev_7v.common import (
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
    deploy_homo_offset_key,
    deploy_mode,
    deploy_narrow_head,
    deploy_side_head,
    ego_ignore_range,
    front_camera_view_names,
    head,
    img_ori_size,
    img_resize_wh_size,
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
    train_batch_size_per_gpu,
    train_global_sample_interval,
    train_num_workers,
    training_step,
    use_split_dataloader,
    val_batch_size_per_gpu,
    val_num_workers,
    vcs_origin_coord,
    vcs_plane_heights,
    vcs_range,
)

cfg_dir = os.path.dirname(__file__)

# -------------------------- TASK ---------------------------
task_name = "bev_3d_vrumerge"

enable_tensorboard = True

roi_vcs_range = (-32.0, -32.0, 52.8, 32.0)
task_in_stride = 2
task_out_size = (424, 320)

# ------------------ DATASET SETTING -------------------
dataset_repeat_config = {"AEB": 2, "AEB_Test_Site": 10 - 1}
val_convert_mf_to_sf_config = {
    "v1_1_bev_3d_all_range_aeb_temporal_val_rm_ovlp_sampled": {
        "enable": True,
        "length_of_clip": 128,
        "reverse_select": True,
    },
    "v0_3_bev_3d_7v_all_range_main_temporal_val_sampled": {
        "enable": True,
        "length_of_clip": 128,
        "reverse_select": True,
    },
    "v3_0_bev_3d_7v_all_range_main_seq_val_sampled_egomotion": {
        "enable": True,
        "length_of_clip": 128,
        "reverse_select": True,
    },
    "v3_0_bev_3d_7v_all_range_aeb_seq_val_sampled_egomotion": {
        "enable": True,
        "length_of_clip": 128,
        "reverse_select": True,
    },
    "v3_0_1_aeb_test_site_val": {
        "enable": True,
        "length_of_clip": 16,
        "reverse_select": True,
    },
}
train_data_version = "v3_0_2_bev_3d_7v_wide_main"
val_data_version = [
    # "v0_3_bev_3d_7v_all_range_main_temporal_val_sampled",
    # "v1_1_bev_3d_all_range_aeb_temporal_val_rm_ovlp_sampled",
    "v3_0_bev_3d_7v_all_range_main_seq_val_sampled_egomotion",
    "v3_0_bev_3d_7v_all_range_aeb_seq_val_sampled_egomotion",
    "v3_0_1_aeb_test_site_val",
]
if pipeline_test:
    train_data_version = "pipeline_test"
    val_data_version = "pipeline_test"
dataset_dir = os.path.join(
    f"{os.path.dirname(__file__)}", "..", "datasets", "bev_3d"
)

# transform
num_classes = 2
use_ignore_mask = False
use_ignore_mask_img = True
cls_dimension = np.array(
    [
        [1.5518998, 0.73560804, 1.7083853],
        [1.6383535, 0.60629284, 0.5058226],
    ]
)  # vrumerge
cls_hm_kernel = {
    0: 9,
    1: 9,
}
category2id_map = {
    "Pedestrian": 1,
    "Car": -99,
    "Cyclist": 0,
    "Bus": -99,
    "Truck": -99,
    "Tricycle": -99,
    "Blur": -99,
    "Construction": -99,
    "Other": -99,
}
id2label = {
    0: "Cyclist",
    1: "Pedestrian",
}

background_reweight_cfg = {"tricycle": {"kernel": 7, "weight": 10}}

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
N_steps_PSC_rot = 3

bev3d_rpy_transforms = None

bev3d_target = dict(
    type="ANCBev3dTargetGenerator",
    num_classes=num_classes,
    max_objs=max_objs,
    bev_size=task_out_size,
    vcs_range=roi_vcs_range,
    cls_dimension=cls_dimension,
    cls_hm_kernel=cls_hm_kernel,
    category2id_map=category2id_map,
    enable_ignore=use_ignore_mask,
    ego_ignore_range=ego_ignore_range,
    use_occlusion_attribute=use_occlusion_attribute,
    occlusion_attribute_seq=list(occlusion_attribute_dict.keys())
    if use_occlusion_attribute
    else None,
    occlusion_ignore_id=occlusion_ignore_id,
    use_psc_rot=use_psc_rot,
    N_steps_PSC_rot=N_steps_PSC_rot,
    background_reweight_cfg=background_reweight_cfg,
)


bev_3d_transforms = get_bev_3d_transforms(
    [bev3d_target],
    copy.deepcopy(bev_common_transforms),
    load_data_types=load_data_types,
    bev3d_rpy_transforms=bev3d_rpy_transforms,
    use_occlusion_attribute=use_occlusion_attribute,
    occlusion_attribute_dict=occlusion_attribute_dict
    if use_occlusion_attribute
    else None,
    use_ignore_mask_img=use_ignore_mask_img,
)

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
    temporal_bev=False,
    length_of_clip=None,
    train_num_frames_per_iter=None,
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
        "convert_mf_to_sf",
        "num_max_frames",
        "num_frames_per_iter",
        "reverse_select",
    ],
    update_trans_info={
        "ANCCollect3DV": "fill_fake_temporal_data",
        "ANCBev3dTargetGenerator": "filter_vcs_range",
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
    set_data_dict_value(data_dict, "convert_mf_to_sf", False)
    set_data_dict_value(data_dict, "num_max_frames", None)
    set_data_dict_value(data_dict, "num_frames_per_iter", None)
    set_data_dict_value(data_dict, "reverse_select", None)
    train_datasets = get_datasets(
        train_dataset_dict,
        None,
        template_dataset,
        data_dict,
        update_bev_dataset,
        homo_noise=train_homo_noise,
        global_sample_interval=train_global_sample_interval,
        repeat_dataset_config=dataset_repeat_config,
    )
    data_loader = get_dataloader(
        train_datasets,
        train_num_workers,
        train_batch_size_per_gpu,
        shuffle=True,
        persistent_workers=train_num_workers > 0,
    )

    if use_split_dataloader:
        data_loader = convert_to_split_dataloader(data_loader)
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
    set_data_dict_value(data_dict, "convert_mf_to_sf", False)
    set_data_dict_value(data_dict, "num_max_frames", None)
    set_data_dict_value(data_dict, "num_frames_per_iter", None)
    set_data_dict_value(data_dict, "reverse_select", None)
    for version in val_convert_mf_to_sf_config:
        if version in val_dataset_dict:
            if val_convert_mf_to_sf_config[version].get("enable", False):
                dataset_dict = val_dataset_dict[version]
                set_data_dict_value(dataset_dict, "convert_mf_to_sf", True)
                set_data_dict_value(dataset_dict, "num_frames_per_iter", 1)
                set_data_dict_value(
                    dataset_dict,
                    "num_max_frames",
                    val_convert_mf_to_sf_config[version]["length_of_clip"],
                )
                set_data_dict_value(
                    dataset_dict,
                    "reverse_select",
                    val_convert_mf_to_sf_config[version]["reverse_select"],
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

if use_occlusion_attribute:
    head_channels["bev3d_occlusion_hm"] = 4


roi_resize_cfg = get_roi_resize_cfg(
    input_size=bevfusion_output_size,
    in_stride=task_in_stride,
    output_size=task_out_size,
    ori_vcs_range=vcs_range,
    roi_vcs_range=roi_vcs_range,
)
vrumerge_roi_resize = dict(
    type="RoiCropResize",
    in_strides=[2, 4, 8, 16, 32],
    target_stride=roi_resize_cfg["in_stride"],
    output_size=roi_resize_cfg["output_size"],
    roi_box=roi_resize_cfg["roi_box"],
    node_name="bev_3d_vrumerge_roi_resize",
)


eps_channels = 1
quant_config = {"bev3d_hm": "qint16"}

if use_occlusion_attribute:
    quant_config.update({"bev3d_occlusion_hm": "qint16"})

use_maxpool = False
use_norm_rot = True
max_pool_kernel = 3


score_threshold = 0.0

score_threshold = model_thresh.get(task_name, {}).get(
    "score_threshold", score_threshold
)

roi_filter_info = None
if deploy_mode:
    roi_filter_vcs_range = (-10.0, -8.0, 25.0, 4.0)
    roi_score_threshold = model_thresh.get(task_name, {}).get(
        "roi_score_threshold"
    )
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
            "radius": 0.6,
            "e_loc_threshold": 0.4,
            "angle_threshold": 3.2,
            "area_threshold": 0.1,
        },
        {
            "p_t": 0.35,
            "min_t": 4.0,
            "max_t": 8.0,
            "radius": 0.4,
            "e_loc_threshold": 0.3,
            "angle_threshold": 3.2,
            "area_threshold": 0.1,
        },
    ],
    "agnostic": True,
    "topk_deploy": 120,
}

loss_weights = {
    "bev3d_hm": 2.0,
    "bev3d_dim": 0.5,
    "bev3d_rot": 1.0,
    "bev3d_ct_offset": 0.75,
    "bev3d_loc_z": 0.75,
}

if use_occlusion_attribute:
    loss_weights["bev3d_occlusion_hm"] = 0.5

class_weight = [1.0, 2.0]
# aidi_eval = True if need aidi leaderboard evaluation
aidi_eval = False

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
    use_occlusion_attribute,
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
        "task": "bev_3d_vrumerge",
        "output_name": "bev_3d_vrumerge_heatmap_output",
        "score_threshold": apply_score_thresh,
        "properties": [{"channel_labels": ["score"]}],
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
                    letnms_setting[param_name].append(param_value)
            postprocess_desc.update(letnms_setting)
    if use_psc_rot and N_steps_PSC_rot is not None:
        channel_labels = [f"cos_idx{idx}" for idx in range(N_steps_PSC_rot)]
        postprocess_rot_desc = {
            "task": "bev_3d_vrumerge",
            "output_name": "bev_3d_vrumerge_rot_output",
            "properties": [{"channel_labels": channel_labels}],
            "rot_mod_threshold": 0.0001,
            "use_norm_rot": use_norm_rot,
            **apply_common_desc,
        }
    else:
        postprocess_rot_desc = {
            "task": "bev_3d_vrumerge",
            "output_name": "bev_3d_vrumerge_rot_output",
            "properties": [{"channel_labels": ["cos", "sin"]}],
            "use_norm_rot": use_norm_rot,
            **apply_common_desc,
        }
    per_tensor_desc = [
        postprocess_desc,
        {
            "task": "bev_3d_vrumerge",
            "output_name": "bev_3d_vrumerge_dimension_output",
            "cls_dimension": [
                1.5518998,
                0.73560804,
                1.7083853,
                1.6383535,
                0.60629284,
                0.5058226,
            ],
            "properties": [{"channel_labels": ["height", "width", "length"]}],
            **apply_common_desc,
        },
        postprocess_rot_desc,
        {
            "task": "bev_3d_vrumerge",
            "output_name": "bev_3d_vrumerge_ct_offset_output",
            "properties": [{"channel_labels": ["offset_x", "offset_y"]}],
            **apply_common_desc,
        },
        {
            "task": "bev_3d_vrumerge",
            "output_name": "bev_3d_vrumerge_loc_z_output",
            "properties": [{"channel_labels": ["z_value"]}],
            **apply_common_desc,
        },
    ]
    if use_occlusion_attribute:
        per_tensor_desc.append(
            {
                "task": "bev_3d_vrumerge",
                "output_name": "bev_3d_vrumerge_occlusion_output",
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
    per_tensor_desc.append(
        {
            "task": "bev_3d_vrumerge",
            "output_name": "bev_3d_vrumerge_class_id_output",
            "roi_filter_info": apply_roi_filter_info,
            "properties": [{"channel_labels": ["cyclist", "pedestrian"]}],
            **apply_common_desc,
        }
    )
    per_tensor_desc = [json.dumps(i) for i in per_tensor_desc]
    return per_tensor_desc


def get_model(mode):
    bev_head = dict(
        type="OutputModule",
        head=dict(
            type="ANCBEV3DHead",
            in_strides=[2],
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
        ),
        head_parser=None,
        loss=None,
        postprocess=None,
        target=None,
        convert_to_dict=True,
        node_name="bev_stage2_3d_vrumerge_head",
    )
    hm_max = dict(
        type="MaxPostProcess",
        data_names=["bev3d_hm"],
        out_names=[["bev3d_hm", "bev3d_cls_id"]],
        dim=1,
        keepdim=True,
    )
    occlusion_hm_argmax = dict(
        type="ArgmaxPostprocess",
        data_name="bev3d_occlusion_hm",
        dim=1,
        keepdim=True,
    )
    if mode == "train":
        bev_head["loss"] = dict(
            type="ANCBEV3DLoss",
            loss_weights=loss_weights,
            class_weight=class_weight,
            gamma=1,
            beta=0.33,
            use_norm_rot=use_norm_rot,
            gt_name="gt_bev_3d",
            use_rot_wing_loss=False,
        )
    elif mode == "val":
        postprocess = [hm_max]
        if use_occlusion_attribute:
            postprocess.append(occlusion_hm_argmax)
        postprocess.append(
            dict(
                type="ANCBEV3Decoder",
                topk=300,
                use_maxpool=use_maxpool,
                max_pool_kernel=max_pool_kernel,
                cls_dimension=cls_dimension,
                vcs_range=roi_vcs_range,
                bev_3d_out_size=task_out_size,
                num_classes=num_classes,
                add_hm_eps=False,
                eps_channels=eps_channels,
                nms_setting=nms_setting,
                score_threshold=score_threshold,
                roi_filter_info=roi_filter_info,
                use_psc_rot=use_psc_rot,
                N_steps_PSC_rot=N_steps_PSC_rot,
                use_norm_rot=use_norm_rot,
            )
        )
        if aidi_eval:
            bev_head["convert_to_dict"] = False
        bev_head["postprocess"] = postprocess
    else:
        bev3d_desc = get_desc(
            use_occlusion_attribute=use_occlusion_attribute,
            use_psc_rot=use_psc_rot,
            N_steps_PSC_rot=N_steps_PSC_rot,
            score_thresh=score_threshold,
            bev_common_desc=bev_common_desc,
            use_maxpool=use_maxpool,
            use_norm_rot=int(use_norm_rot),
            max_pool_kernel=max_pool_kernel,
            nms_setting=nms_setting,
            roi_filter_info=roi_filter_info,
        )
        bev_head["head"]["dequant_out"] = False
        bev_head["convert_to_dict"] = False
        postprocess_modules = []
        postprocess_modules.append(hm_max)
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
            )
        )
        if use_occlusion_attribute:
            postprocess_modules.append(occlusion_hm_argmax)
        postprocess_modules.append(
            dict(
                type="AddDesc",
                per_tensor_desc=bev3d_desc,
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
        bevfusion_pick_keys = deploy_homo_offset_key

    model = dict(
        type="MultiViewTwoStageBEVModule",
        # view_img_key to module
        multi_view_module=multi_view_module,
        multi_view_collect=multi_view_collect,
        bev_fusion_module=dict(
            type="BEVStageTwoModule",
            bevfusion=bev_fusion[mode],
            bev_fusion_upsample=bev_fusion_upsampling,
            backbone=bev_backbone,
            neck=bev_neck,
            roi_resizes=vrumerge_roi_resize,
            head=bev_head,
        ),
        bevfusion_pick_keys=bevfusion_pick_keys,
    )

    return model


# -------------------------- TRAIN METRIC --------------------------
metrics, per_metric_patterns = get_metrics_patterns(
    task_name, use_occlusion_attribute=True
)
metric_updater = get_metric_updater(metrics, per_metric_patterns, task_name)

# ------------------------- VALIDATION SETTING-----------------------
visibility_intervals = None

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
}

eval_setting = {
    "eval_category_ids": (
        0,
        1,
    ),
    "score_threshold": 0.1,
    "iou_threshold": 0.2,
    "gt_max_depth": 60,
    "depth_intervals": (3, 6, 12, 25),
    "eval_vcs_range": (-30.0, -30.0, 50.0, 30.0),
    "ego_ignore_range": ego_ignore_range,
    "eval_mode": ["bev_iou", "let_iou"],
    "let_iou_param": {"p_t": 0.35, "min_t": 4.0, "max_t": 8.0},
    "base_taggers": base_taggers,
    "metric_key": (
        "Recall",
        "Precision",
        "dxp",
        "dyp",
        "dxyp",
        "drot",
        "rsize",
    ),
    "group_pred_by_cls": True,
    "auto_threshold": True,
    "eval_category_cls": False,
    "distance_wise_mode": "xy",
}

val_metric_updater_list = []
for dataset_key in _as_list(val_data_version):
    save_dir = os.path.join(save_prefix, "eval", dataset_key)
    if not os.path.exists(save_dir):
        # shutil.rmtree(save_dir)
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

    val_metrics = get_val_metrics(
        task_name,
        **eval_setting,
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
        output_dir=os.path.join(save_prefix, "visualize", "bev_3d_vrumerge"),
        task="bev_3d_vrumerge",
        prefix="OutputModule_predict",
        score_threshold=score_threshold[0]
        if isinstance(score_threshold, (list, tuple))
        else score_threshold,
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
