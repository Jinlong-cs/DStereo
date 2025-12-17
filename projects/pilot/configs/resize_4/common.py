import json
import os
import sys
from copy import deepcopy

import numpy as np
import torch

from hat.core.proj_spec.descs import position_encoding_desc
from hat.utils import Config

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from project_common import (  # noqa
    enable_model_tracking,
    eval_data_setting,
    model_checkpoint,
    model_name_postfix,
    model_setting,
    model_thresh,
    model_version,
    num_machines,
    pipeline_test,
    resume_training,
    tasks,
    training_step,
)

sys.modules.pop("project_common")

# clipping: begin
if training_step == "int_infer":
    os.environ["NO_HDFLOW"] = "1"

from datasets.partitions import parse_by_partition

# clipping: end

with_rle = False
with_angle_augmentation = False

# model info
model_type = "pilot_multitask_resize4"
model_name = "_".join([model_type, model_setting, model_version])
if model_name_postfix:
    model_name = "_".join([model_name, model_name_postfix])

if model_thresh is not None:
    model_thresh = json.loads(model_thresh)

if tasks is not None:
    tasks = json.loads(tasks)
else:
    # tasks
    vehicle_tasks = [
        dict(name="vehicle_detection", important=True),
        dict(name="vehicle_category_classification"),
        dict(name="vehicle_occlusion_classification"),
        dict(name="vehicle_truncation_classification"),
        dict(name="vehicle_wheel_detection", important=True),
        dict(name="vehicle_ground_line"),
        # dict(name="vehicle_flank", important=True),
    ]

    rear_tasks = [
        dict(name="rear_detection", important=True),
        dict(name="rear_plate_detection"),
        dict(name="rear_occlusion_classification"),
        dict(name="rear_part_classification"),
    ]

    person_tasks = [
        dict(name="person_detection", important=True),
        dict(name="person_face_detection"),
        dict(name="person_occlusion_classification"),
        dict(name="person_orientation_classification"),
        dict(name="person_pose_classification"),
    ]

    cyclist_tasks = [
        dict(name="cyclist_detection", important=True),
    ]

    dense_tasks = [
        dict(name="default_segmentation", important=True),
    ]

    if "parking" not in model_setting:
        dense_tasks.append(dict(name="lane_segmentation", important=True))

    cam_3d_tasks = [
        dict(name="vehicle_heatmap_3d_detection", important=True),
        dict(name="ped_cyc_heatmap_3d_detection", important=True),
    ]

    if "sparse_3d" not in training_step and training_step != "int_infer":
        dense_tasks += cam_3d_tasks
    else:
        vehicle_tasks.append(dict(name="vehicle_roi_3d", important=True))
        person_tasks.append(dict(name="person_roi_3d", important=True))
        cyclist_tasks.append(dict(name="cyclist_roi_3d", important=True))

    tasks = (
        vehicle_tasks + rear_tasks + person_tasks + cyclist_tasks + dense_tasks
    )

# dataset
ds_path = os.path.join(
    os.path.dirname(__file__),
    f"../datasets/{model_setting.lower()}_datasets.py",
)
ds_cfg = Config.fromfile(ds_path)
datapaths = ds_cfg.datapaths

is_local_train = not os.path.exists("/running_package")
batch_size_factor = 2
if is_local_train:
    batch_size = 4
    if pipeline_test:
        batch_size = 1
    batch_size = batch_size * batch_size_factor
    log_freq = 5
    save_prefix = "tmp_output"
else:
    batch_size = 16 * batch_size_factor
    log_freq = 25
    save_prefix = "/job_data/models/"

pred_batch_size = 64

lmdb_data = "lmdb" in model_setting.lower()

# clipping: begin
if ds_cfg is not None and datapaths is not None and lmdb_data:
    partitions = ds_cfg.get("partitions", {})
    datapaths = parse_by_partition(
        datapaths,
        partitions,
        training_step,
        os.path.join(save_prefix, model_type, "logs"),
    )
# clipping: end

input_hw = resize_hw = (
    (320, 480)
    if "x3c" in model_setting.lower() or "test" in model_setting.lower()
    else (320, 512)
)
actual_input_hw = (320, 512)
# input_padding (w_left, w_right, h_top, h_bottom)
input_padding = (
    [16, 16, 0, 0]
    if "x3c" in model_setting.lower() or "test" in model_setting.lower()
    else [0, 0, 0, 0]
)
roi_region = (0, 0, *input_hw[::-1])
vanishing_point = (input_hw[1] // 2, input_hw[0] // 2)

bn_kwargs = dict(eps=1e-5, momentum=0.1)

# data_config
inter_method = 10
pixel_center_aligned = False
min_valid_clip_area_ratio = 0
rand_translation_ratio = 0.1

# 3d config
undistort_depth_uv = True

# whether use rotation_iou
use_rotation_iou = False


if with_angle_augmentation:
    # pitch +代表下压， -代表上扬
    pitch_list_N = [-8, -7, -6, -5.5, -5, -4, -3, -2.5, -2, -1.5, -1]  # noqa
    pitch_list_P = [0, 1, 1.5, 2, 2.5, 3, 4, 5, 5.5, 6, 7, 8]  # noqa
    pitch_list = pitch_list_N + pitch_list_P

    standardized_calib_all = {
        # "roll": 0.0,
        "pitch": [x / 180 * np.pi for x in pitch_list],
        # "center_u": float(resize_hw[1]) * 2,
        # "center_v": float(resize_hw[0]) * 2,
        # "focal_u": 560.65365601,
        # "focal_v": 792.67155457,
        # "distort": [0.0] * 8,
        "image_width": resize_hw[1] * 4,
        "image_height": resize_hw[0] * 4,
        "post_resize_scale": 4.0,
        "warping_on_bpu": False,
        "enable_inv_trans": True,
    }
else:
    standardized_calib_all = None

default_calib = torch.tensor(
    [
        [
            [1114.34668, 0, 978.904541, 0],
            [0, 1114.34668, 670.611328, 0],
            [0, 0, 1, 0],
        ]
    ]
    * pred_batch_size
)

default_distCoeffs = torch.tensor(
    [
        [
            -0.586143374,
            0.221308455,
            1.17504729e-04,
            1.71026128e-04,
            7.10545704e-02,
            -0.186895579,
            -9.31752697e-02,
            0.222488135,
        ]
    ]
    * pred_batch_size
)

# roi configs


def get_det_rpn_out_keys(mode):
    if "train" in mode:
        return None
    elif "val" in mode:
        return []
    elif "test" in mode:
        return ["pred_boxes"]
    else:
        raise KeyError(mode)


test_roi_num = 30

# position encoding config
is_with_pe = True if "x3c" in model_setting.lower() else False
is_loss_custom = False

undistort_depth_uv = False if is_with_pe else undistort_depth_uv
pe_config = dict(
    is_with_pe=is_with_pe,
    pe_stride=4,
    pe_c=3,
    pe_h=(input_hw[0] + sum(input_padding[2:])) // 4,
    pe_w=(input_hw[1] + sum(input_padding[:2])) // 4,
    img_resize=4,
    input_hw=input_hw,
    default_intrinsic_mat=default_calib[0].numpy(),
    default_distort=default_distCoeffs[0].numpy(),
    default_pitch=0,
    default_roll=0,
    default_camera_z=1,
    crop_roi_3d=None,
    is_with_relu=False,
    verbose=1,
)
pe_desc = dict(
    type="AddDesc",
    per_tensor_desc=position_encoding_desc(
        task_name="position_encoding",
        shape=(
            1,
            pe_config["pe_h"],
            pe_config["pe_w"],
            pe_config["pe_c"],
        ),
    ),
    node_name="pe_desc",
)

# common structures
backbone = dict(
    type="ZeroPad2DPatcher",
    backbone=dict(
        type="VargNetV2Stage2631",
        num_classes=1000,
        multiplier=0.5,
        group_base=4,
        last_channels=1024,
        stages=(1, 2, 3, 4, 5),
        include_top=False,
        extend_features=True,
        bn_kwargs=bn_kwargs,
        node_name="backbone",
    ),
    input_padding=input_padding,
)

fpn_neck = dict(
    type="FPN",
    in_strides=[2, 4, 8, 16, 32, 64],
    in_channels=[16, 16, 32, 64, 128, 128],
    out_strides=[4, 8, 16, 32, 64],
    out_channels=[16, 32, 64, 128, 128],
    bn_kwargs=bn_kwargs,
    node_name="fpn_neck",
)

fix_channel_neck = dict(
    type="FixChannelNeck",
    in_strides=[4, 8, 16, 32, 64],
    in_channels=[16, 32, 64, 128, 128],
    out_strides=[8, 16, 32, 64],
    out_channel=64,
    bn_kwargs=bn_kwargs,
    node_name="fix_channel_neck",
)

ufpn_seg_neck = dict(
    type="UFPN",
    in_strides=[4, 8, 16, 32, 64],
    in_channels=[16, 32, 64, 128, 128],
    out_channels=[16, 32, 64, 128, 128],
    bn_kwargs=bn_kwargs,
    group_base=4,
    node_name="ufpn_seg_neck",
)

ufpn_3d_neck = deepcopy(ufpn_seg_neck)
ufpn_3d_neck.update(
    dict(
        output_strides=[4],
        node_name="ufpn_3d_neck",
    )
)

if is_with_pe:
    ufpn_3d_neck.update(
        dict(
            is_with_relu=pe_config["is_with_relu"],
            pe_stride=pe_config["pe_stride"],
            pe_channel=pe_config["pe_c"],
        )
    )

val_decoders = {}

val_transforms = [
    dict(type="YUVTurboJPEGDecoder", to_string=True),
    dict(type="BPUPyramidResizer", scale_wh=(0.25, 0.25), pyramid_type="ips"),
    dict(type="CropImgPatch", static_roi=roi_region, is_buf=False),
    dict(type="ToTensor", to_yuv=False),
    dict(type="Normalize", mean=128.0, std=128.0),
]

vis_tasks = []

loss_custom_weight_init = {
    "dense_heatmap": 1,
    "dense_rotation": 1,
    "dense_depth": 1,
    "dense_dimensions": 1,
    "dense_location_offset": 1,
    "dense_box2d_wh": 1,
    "sparse_center_2d": 1,
    "sparse_offset_2d": 1,
    "sparse_offset_3d": 1,
    "sparse_depth": 1,
    "sparse_depth_u": 1,  # if undistort_depth_uv=True
    "sparse_depth_v": 1,  # if undistort_depth_uv=True
    "sparse_dim": 1,
    "sparse_rot": 1,
    "sparse_iou": 1,
}
