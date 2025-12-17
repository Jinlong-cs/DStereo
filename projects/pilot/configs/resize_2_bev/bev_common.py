# flake8: noqa

import copy
import json
import math

import numpy as np
import torch
from common import (
    backbone,
    batch_size,
    batch_size_bev,
    bn_kwargs,
    fpn_neck,
    front_input_hw,
    front_resize_hw,
    front_roi_region,
    input_hw,
    is_local_train,
    lmdb_data,
    log_freq,
    process_type,
    raw_front_image_hw,
    raw_image_hw,
    resize_hw,
    roi_region,
    split_mode,
    ufpn_seg_neck,
    val_transforms,
    vanishing_point,
    view_num,
    vis_tasks_bev,
    with_cam_standiardization,
)

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.core.proj_spec.descs import get_lane_parsing_desc
from hat.data.collates.collates import collate_3d
from hat.models.backbones.vargnetv2 import get_vargnetv2_stride2channels
from hat.utils.apply_func import _as_list

if view_num == 4:
    raise NotImplementedError("Not supported for 4 view input NOW!")

elif view_num == 5:
    raw_image_hw_list = [raw_image_hw] * 5
    resize_hw_list = [resize_hw] * 5
    input_hw_list = [input_hw] * 5
    roi_region_list = [roi_region] * 5

elif view_num == 6:
    raw_image_hw_list = [raw_front_image_hw] + [raw_image_hw] * 5
    resize_hw_list = [front_resize_hw] + [resize_hw] * 5
    input_hw_list = [front_input_hw] + [input_hw] * 5
    roi_region_list = [front_roi_region] + [roi_region] * 5


# bev loss weight
loss_weight = 1.0
task_loss_weights = {
    "bev_3d_vehicle_cls": 2.0,
    "bev_3d_vehicle": 1.0,
    "bev_3d_pedestrian": 1.0,
    "bev_3d_cyclist_cls": 1.0,
    "bev_3d_cyclist": 1.0,
    "bev_seg": 1.0,
    "bev_om": 5.0,
}
bucket_root = "/horizon-bucket" if is_local_train else "/bucket/input"


camera_view_names = [
    "camera_front_left",
    "camera_front_right",
    "camera_rear_left",
    "camera_rear_right",
    "camera_rear",
]

per_view_shape = {
    "camera_front_left": raw_image_hw,
    "camera_front_right": raw_image_hw,
    "camera_rear_left": raw_image_hw,
    "camera_rear_right": raw_image_hw,
    "camera_rear": raw_image_hw,
}

if view_num == 6:
    camera_view_names = ["camera_front"] + camera_view_names
    per_view_shape["camera_front"] = raw_front_image_hw

if with_cam_standiardization:
    side_cam_calib = {
        # "yaw": 0,
        "pitch": 0,
        "roll": 0,
        "center_u": int(raw_image_hw[1] / 2),
        "center_v": int(raw_image_hw[0] / 2),
        "focal_u": 827.54518127,
        "focal_v": 1012.73890686,
        "distort": [0.0] * 8,
    }

    rear_cam_calib = {
        # "yaw": 0,
        "pitch": 0,
        "roll": 0,
        "center_u": int(raw_image_hw[1] / 2),
        "center_v": int(raw_image_hw[0] / 2),
        "focal_u": 1685.5672607421875,
        "focal_v": 1843.05859375,
        "distort": [0.0] * 8,
    }
    standardized_cam_calibs = {
        "camera_front_left": side_cam_calib,
        "camera_front_right": side_cam_calib,
        "camera_rear_left": side_cam_calib,
        "camera_rear_right": side_cam_calib,
        "camera_rear": rear_cam_calib,
    }
    cam_standardized_transform = dict(
        type="CameraStandardization",
        warping_on_bpu=True,
        meta_key=(),
        enable_inv_trans=False,
        image_width=resize_hw[1],
        image_height=resize_hw[0],
    )
    default_calib = {
        "camera_x": 0,
        "camera_y": 0,
        "camera_z": 1.0,
        "center_u": int(raw_image_hw[1] / 2),
        "center_v": int(raw_image_hw[0] / 2),
        "distort": [0.0] * 8,
        "focal_u": 1149.12744140625,
        "focal_v": 1149.12744140625,
        "fov": 100,
        "image_height": 1280,
        "image_width": 1920,
        "pitch": 0.0,
        "roll": 0.0,
        "yaw": 0.0,
        "vcs": {"rotation": [0, 0, 0], "translation": [0, 0.0, 0]},
    }
else:
    standardized_cam_calibs = None
    cam_standardized_transform = None
    default_calib = None

classidx2name = {
    0: "unknow",
    1: "pedestrian",
    2: "car",
    3: "cyclist",
    4: "bus",
    5: "truck",
    6: "specialcar",
    7: "tricycle",
    8: "dontcare",
}

vcs_range = (-70.0, -50.0, 30.0, 50.0)
# vcs_range = (-50.0, -40.0, 30.0, 40.0)
assert vcs_range[2] - vcs_range[0] == vcs_range[3] - vcs_range[1]

vcs_origin_coord = [362, 256]


bevfusion_output_size = (512, 512)

bev_fusion_input_stride = 4


# used in bev fusion block
ipm_output_size = (256, 256)  # (height, witdh)
assert ipm_output_size in [
    (256, 256),
    (512, 512),
], "ipm outsize only support (256,256) or (512,512)"

upsampling_after_fusion = bevfusion_output_size != ipm_output_size

spatial_resolution = (
    abs(vcs_range[2] - vcs_range[0]) / ipm_output_size[0],
    abs(vcs_range[3] - vcs_range[1]) / ipm_output_size[1],
)  # (height, witdh)
# spatial resolution in bevfusion feature
bevfusion_spatial_resolution = (
    abs(vcs_range[2] - vcs_range[0]) / bevfusion_output_size[0],
    abs(vcs_range[3] - vcs_range[1]) / bevfusion_output_size[1],
)  # (height, witdh)


# code below is settings about block warp
pad_top = int(abs(vcs_range[2] / spatial_resolution[0]))
pad_bottom = int(abs(vcs_range[0] / spatial_resolution[0]))
pad_left = int(abs(vcs_range[3] / spatial_resolution[1]))
pad_right = int(abs(vcs_range[1] / spatial_resolution[1]))

# warp_padding`s order is (left,right,up,bottom)
block_warp_padding = [
    (0, pad_right, 0, 0),
    (pad_left, 0, 0, 0),
    (0, pad_right, 0, 0),
    (pad_left, 0, 0, 0),
    (0, 0, pad_top, 0),
]
if view_num == 6:
    block_warp_padding = [(0, 0, 0, pad_bottom)] + block_warp_padding

warp_sizes = [
    (
        ipm_output_size[0] - pad[2] - pad[3],
        ipm_output_size[1] - pad[0] - pad[1],
    )
    for pad in block_warp_padding
]
# code up is settings about block warp

if view_num == 5:
    organize_data_type = "front"
elif view_num == 6:
    organize_data_type = "front_side"

use_homo_offset = True
use_distorted_offset = True

# class settings for stage1 seg loss
# not used currently in pilot

seg_class = 16

# model


def get_grid_quant_scale(ipm_output_size, view_max_size):
    max_coord = max(
        view_max_size,
        ipm_output_size[0],
        ipm_output_size[1],
    )
    coord_bit_num = math.ceil(math.log(max_coord + 1, 2))
    coord_shift = 15 - coord_bit_num
    coord_shift = max(min(coord_shift, 8), 0)
    grid_quant_scale = 1.0 / (1 << coord_shift)
    return grid_quant_scale


stage2_alpha = 1.5
bev_fusion_input_name = "pred_segs_frame0"
bev_fusion_output_name = "bev_stage2_input"
bev_stage2_feats_name = "bev_stage2_feats"
stage2_stride2channels = get_vargnetv2_stride2channels(stage2_alpha)
feat_channels = 32

neck_strides = [4, 8, 16, 32, 64]
neck_channels = [16, 32, 64, 128, 128]
neck_stride2channels = {s: c for s, c in zip(neck_strides, neck_channels)}

view_max_size = 256
grid_quant_scale = get_grid_quant_scale(ipm_output_size, view_max_size)
use_random_rotation = False

if view_num == 5:
    views = 5
elif view_num == 6:
    views = [1, 5]

seperate_neck_for_bev_tasks = False

if not seperate_neck_for_bev_tasks:
    bev_stage1_neck = fpn_neck
else:
    bev_stage1_neck = copy.deepcopy(fpn_neck)
    bev_stage1_neck["node_name"] = "bev_stage1_fpn_neck"

seperate_stage2_among_bev_tasks = False
seperate_stage2_neck_among_bev_tasks = False

merge_ped_and_cyc = False


bev_stage1_head = dict(
    type="OutputModule",
    keep_name=True,
    head=dict(
        type="PixelHead",
        feature_name="feats",
        in_strides=neck_strides,
        out_strides=[bev_fusion_input_stride],
        stride2channels=neck_stride2channels,
        forward_frame_idxs=[0],
        out_nums=seg_class,
        output_name="pred_segs",
        quanti_last_conv=True,
        dequant_out=False,
        bn_kwargs=bn_kwargs,
        group_base=8,
    ),
    node_name="bev_stage1_head",
)
bev_backbone = dict(
    type="VargNetV2",
    num_classes=1000,
    bn_kwargs=bn_kwargs,
    alpha=stage2_alpha,
    group_base=8,
    factor=2,
    bias=False,
    include_top=False,
    flat_output=False,
    input_channels=seg_class,
    disable_quanti_input=True,
    node_name="bev_stage2_back_bone",
)
bev_neck = dict(
    type="Unet",
    in_strides=[2, 4, 8, 16, 32],
    out_strides=[2, 4, 8, 16, 32],
    stride2channels=stage2_stride2channels,
    factor=2,
    use_bias=False,
    group_base=8,
    node_name="bev_stage2_neck",
)
bev_fusion = dict(
    type="ANCBEVFusionModule",
    grid_quant_scale=grid_quant_scale,
    random_rotation_cfg=dict(
        type="RandomRotation",
        grid_quant_scale=grid_quant_scale,
        height=ipm_output_size[0],
        width=ipm_output_size[1],
        angles=(0, 90, 180, 270),
    )
    if use_random_rotation
    else None,
    drop_view_prob=0.0,
    bev_fusion_input_name=bev_fusion_input_name,
    bev_fusion_out_name=bev_fusion_output_name,
    views=views,
    ipm_output_size=ipm_output_size,
    use_homo_offset=use_homo_offset,
    block_warp_padding=block_warp_padding,
    stacked_input=True,
    compile_model=False,
    node_name="bev_fusion",
)

bev_fusion_test = dict(
    type="ANCBEVFusionModule",
    grid_quant_scale=grid_quant_scale,
    random_rotation_cfg=dict(
        type="RandomRotation",
        grid_quant_scale=grid_quant_scale,
        height=ipm_output_size[0],
        width=ipm_output_size[1],
        angles=(0, 90, 180, 270),
    )
    if use_random_rotation
    else None,
    drop_view_prob=0.0,
    bev_fusion_input_name=bev_fusion_input_name,
    bev_fusion_out_name=bev_fusion_output_name,
    views=views,
    ipm_output_size=ipm_output_size,
    use_homo_offset=use_homo_offset,
    block_warp_padding=block_warp_padding,
    stacked_input=False,
    compile_model=True,
    node_name="bev_fusion",
)

if upsampling_after_fusion:
    bev_fusion_upsample = dict(
        type="ResizeParser",
        data_name=bev_fusion_output_name,
        use_plugin_interpolate=True,
        dequant_out=False,
        resize_kwargs=dict(size=bevfusion_output_size, mode="bilinear"),
        node_name="bev_upsample",
    )
else:
    bev_fusion_upsample = None

bev_fusion_input_desc = dict(
    type="AddDesc",
    per_tensor_desc=[
        json.dumps(
            dict(
                task="bev",
                roi_input=dict(
                    fp_x=resize_hw[1] // 2,
                    fp_y=resize_hw[0] // 2,
                    width=resize_hw[1],
                    height=resize_hw[0],
                ),
                vanishing_point=vanishing_point,
            )
        )
    ],
)
if split_mode:
    bev_stage1_head.update(dict(postprocess=bev_fusion_input_desc))


def get_common_transforms(
    _size,
    _organize_data_type="front",
    _H_persp_view_scale=1 / 4,
    input_sequence_length=1,
    crop_roi_list=None,
    to_yuv=False,
    lazy_yuv=True,
    input_img_sequence=False,
):
    if crop_roi_list is None:
        crop_roi_list = [[0, 0, s[1], s[0]] for s in _size]
    common_transforms = {
        "Collect3DV": dict(
            type="ANCCollect3DV",
            load_data_types=None,
            img_idxs=[0] if input_sequence_length == 1 else [0, 1],
            gt_seg_idxs=[0],
        ),
        "Resize3DV": dict(
            type="ANCResize3DV",
            size=_size,
        ),
        "Crop3DV": dict(
            type="ANCCrop3DV",
            height=[c[3] - c[1] for c in crop_roi_list],
            width=[c[2] - c[0] for c in crop_roi_list],
            top=[c[1] for c in crop_roi_list],
            left=[c[0] for c in crop_roi_list],
        ),
        "ResizeHomo": dict(
            type="ANCResizeHomo",
            H_persp_view_scale=_H_persp_view_scale,
        ),
        "ToTensor3DV": dict(
            type="ANCToTensor3DV",
            to_yuv=to_yuv,
            with_color_imgs=False,
            lazy_yuv=lazy_yuv,
        ),
        "PrepareDataBEV": dict(
            type="ANCPrepareDataBEV",
            organize_data_type=_organize_data_type,
            single_frame=True if input_sequence_length == 1 else False,
            input_img_sequence=input_img_sequence,
        ),
    }
    return common_transforms


def get_template_dataset(
    _img_load_size=None,
    _camera_view_names=camera_view_names,
    _per_view_shape=per_view_shape,
    _transforms=None,
    _homo_transforms=None,
):
    assert isinstance(_transforms, list)
    template_dataset = dict(
        type="ANCAuto3DV",
        transforms=_transforms,
        img_load_size=[hw[::-1] for hw in resize_hw_list]
        if _img_load_size is None
        else _img_load_size,
        camera_view_names=_camera_view_names,
        per_view_shape=_per_view_shape,
        root=bucket_root,
        sync_file=None,
        sync_file_lmdb=None,
        num_samples=None,
        img_rec_path=None,
        seg_rec_path=None,
        bev_seg_rec_path=None,
        bev_3d_lmdb_path=None,
        gt_online_mapping_dir=None,
        multi_view_rec=None,
        homo_gen=dict(
            homo_path=None,
            calib_path=None,
            spatial_resolution=spatial_resolution,
            vcs_range=vcs_range,
            camera_view_names=_camera_view_names,
            per_view_shape=_per_view_shape,
            norm_homo=True,
            use_distorted_offset=use_distorted_offset,
            homo_transforms=_homo_transforms,
        ),
    )
    return template_dataset


def get_homo_transforms(transforms_list, camera_view_names):
    homo_transforms = {}
    for _index, view in enumerate(camera_view_names):
        view_transform = {}
        for _transform in transforms_list:
            if _transform["type"] == "ANCResize3DV":
                view_transform["Resize"] = _transform["size"][_index]
            if _transform["type"] == "ANCPad3DV":
                view_transform["Pad"] = _transform["paddings"][_index]
            if _transform["type"] == "ANCCrop3DV":
                view_transform["Crop"] = (
                    _transform["top"][_index],
                    _transform["left"][_index],
                )
            if _transform["type"] == "ANCResizeHomo":
                view_transform["ResizeHomo"] = _transform["H_persp_view_scale"]
        homo_transforms[view] = view_transform
    return homo_transforms


def get_multi_view_datasets(rec_list, template_dataset, update_bev_dataset):
    ret_datasets = []
    for rec in rec_list:
        cur_info = {}
        cur_info["multi_view_rec"] = rec
        cur_dataset = copy.deepcopy(template_dataset)
        cur_dataset = update_bev_dataset(cur_dataset, cur_info)
        ret_datasets.append(cur_dataset)
    return ret_datasets


def get_update_bev_dataset_func(added_info):
    # the "update_bev_dataset" only contains the common bev_data, for the
    # added info, should add it in the bev_xxx.config, e.g. homo_offset_path,
    # homo_path
    added_info = _as_list(added_info)

    def update_bev_dataset(dataset, info):
        dataset["multi_view_rec"] = info["multi_view_rec"]
        resample_dataset = dict(
            type="ResampleDataset",
            dataset=dataset,
            resample_interval=1,
        )
        return resample_dataset

    return update_bev_dataset


def get_dataloader(datasets, num_workers):
    data_loader = dict(
        type=torch.utils.data.DataLoader,
        collate_fn=collate_3d,
        dataset=dict(type="ConcatDataset", datasets=datasets),
        batch_size=batch_size_bev,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return data_loader
