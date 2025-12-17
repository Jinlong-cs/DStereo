import json
import os
import sys
from copy import deepcopy

import torch

from hat.core.proj_spec.descs import position_encoding_desc
from hat.models.backbones.mixvargenet import MixVarGENetConfig
from hat.utils import Config

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from project_common import (  # noqa
    cam_frame_to_dict,
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
with_cam_standiardization = False
with_sparse_training_stage = False
with_image_edge_mask = False

# clipping: begin
if "niofy" in model_setting:
    with_cam_standiardization = True
    with_sparse_training_stage = False
    with_image_edge_mask = True
# clipping: end

# model info
model_type = "pilot5_multitask_resize2_rear_bayes"
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
        dict(name="vehicle_flank", important=True),
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
    # clipping: begin
    if "niofy" in model_setting:
        cyclist_tasks += [dict(name="cyclist_classification")]
    # clipping: end
    dense_tasks = [
        dict(name="default_segmentation", important=True),
    ]

    if "parking" not in model_setting:
        dense_tasks.append(dict(name="lane_segmentation", important=True))

    cam_3d_tasks = [
        dict(name="vehicle_heatmap_3d_detection", important=True),
        dict(name="ped_cyc_heatmap_3d_detection", important=True),
    ]

    if with_sparse_training_stage:
        if "sparse_3d" not in training_step and training_step != "int_infer":
            dense_tasks += cam_3d_tasks
        else:
            vehicle_tasks.append(dict(name="vehicle_roi_3d", important=True))
            person_tasks.append(dict(name="person_roi_3d", important=True))
            cyclist_tasks.append(dict(name="cyclist_roi_3d", important=True))
    else:
        if training_step != "int_infer":
            dense_tasks += cam_3d_tasks
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
if not os.path.exists(ds_path):
    # 拆分白天夜晚评测集,但是目前训练集没有拆分,白天夜晚共用同一训练集
    ds_path = os.path.join(
        os.path.dirname(__file__),
        f"../datasets/{model_setting.lower().replace('_day', '').replace('_night', '')}_datasets.py",
    )
if os.path.exists(ds_path):
    ds_cfg = Config.fromfile(ds_path)
    datapaths = ds_cfg.datapaths
else:
    ds_cfg = None
    datapaths = None

lmdb_data = "lmdb" in model_setting.lower()

# training
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
    batch_size = 8 * batch_size_factor
    # clipping: begin
    batch_size = (
        7 * batch_size_factor
        if "niofy" in model_setting
        else 8 * batch_size_factor
    )
    # clipping: end
    log_freq = 25
    save_prefix = "/job_data/models/"

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

pred_batch_size = 32

input_hw = (512, 960)
roi_region = (0, 0, 960, 512)
vanishing_point = (960 // 2, 540 // 2)
resize_hw = (640, 960)  # for training by [1280, 1920] image
decoder3d_input_hw = (540, 960)
# currently only used in real3d tasks
crop_roi_3d = (0, 100, 1920, 1124)  # used in 3d tasks to crop raw image
# used in 3d tasks to mask image edge (ltrb)
image_edge_mask_ranges = ((0, 50), (0, 50), (910, 960), (590, 640))

# clipping: begin
if "niofy" in model_setting:
    input_hw = resize_hw = (640, 960)
    roi_region = (0, 0, input_hw[1], input_hw[0])
    vanishing_point = (int(input_hw[1] / 2), int(input_hw[0] / 2))
    decoder3d_input_hw = (640, 960)
    crop_roi_3d = None
# clipping: end
use_legacy_mx_dataset = True
# currently only used in real3d tasks

bn_kwargs = dict(eps=1e-5, momentum=0.1)

# data_config
inter_method = 10
pixel_center_aligned = False
min_valid_clip_area_ratio = 0
rand_translation_ratio = 0.1

# 3d config
undistort_depth_uv = False

# whether use rotation_iou
use_rotation_iou = False

if with_cam_standiardization:
    standardized_calib_all = {
        "roll": 0.0,
        "pitch": 0.0,
        # "center_u": float(resize_hw[1]),
        # "center_v": float(resize_hw[0]),
        # "focal_u": 560.65365601,
        # "focal_v": 792.67155457,
        # "distort": [0.0] * 8,
        "image_width": resize_hw[1] * 2,
        "image_height": resize_hw[0] * 2,
        "post_resize_scale": 2.0,
        "warping_on_bpu": True,
        "enable_inv_trans": True,
    }
    if crop_roi_3d:  # for galaxy
        standardized_calib_all["crop_region"] = crop_roi_3d
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


test_roi_num = 50

# position encoding config
is_with_pe = True
pe_config = dict(
    is_with_pe=is_with_pe,
    pe_stride=4,
    pe_c=3,
    pe_h=(crop_roi_3d[3] - crop_roi_3d[1]) // 8
    if crop_roi_3d
    else resize_hw[0] // 4,
    pe_w=(crop_roi_3d[2] - crop_roi_3d[0]) // 8
    if crop_roi_3d
    else resize_hw[1] // 4,
    img_resize=2,
    input_hw=resize_hw,
    default_intrinsic_mat=default_calib[0].numpy(),
    default_distort=default_distCoeffs[0].numpy(),
    default_pitch=0,
    default_roll=0,
    default_camera_z=1,
    crop_roi_3d=crop_roi_3d,
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
mixvargenet_config = [
    [
        MixVarGENetConfig(
            in_channels=16,
            out_channels=16,
            head_op="mixvarge_f4_gb16",
            stack_ops=[],
            stack_factor=1,
            stride=1,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # stride 2
    ],
    [
        MixVarGENetConfig(
            in_channels=16,
            out_channels=32,
            head_op="mixvarge_f4_gb16",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=2,
        ),  # stride 4
    ],
    [
        MixVarGENetConfig(
            in_channels=32,
            out_channels=64,
            head_op="mixvarge_f4_gb16",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[4],
            extra_downsample_num=2,
        ),  # stride 8
    ],
    [
        MixVarGENetConfig(
            in_channels=64,
            out_channels=128,
            head_op="mixvarge_f4_gb16",
            stack_ops=[
                "mixvarge_f4_gb16",
                "mixvarge_f4_gb16",
                "mixvarge_f4_gb16",
                "mixvarge_f4_gb16",
                "mixvarge_f4_gb16",
                "mixvarge_f4_gb16",
            ],
            stack_factor=1,
            stride=2,
            fusion_strides=[4, 8],
            extra_downsample_num=2,
        ),  # stride 16
    ],
    [
        MixVarGENetConfig(
            in_channels=128,
            out_channels=256,
            head_op="mixvarge_f2_gb16",
            stack_ops=[
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
            ],
            stack_factor=1,
            stride=2,
            fusion_strides=[8, 16],
            extra_downsample_num=2,
        ),  # stride 32
    ],
    [
        MixVarGENetConfig(
            in_channels=256,
            out_channels=256,
            head_op="mixvarge_f2_gb16",
            stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
            stack_factor=1,
            stride=2,
            fusion_strides=[16, 32],
            extra_downsample_num=0,
        ),  # stride 64
    ],
]

backbone = dict(
    type="MixVarGENet",
    net_config=mixvargenet_config,
    output_list=[0, 1, 2, 3, 4, 5],
    input_channels=3,
    num_classes=1000,
    bn_kwargs=bn_kwargs,
    include_top=False,
    bias=True,
    node_name="backbone",
)
if with_cam_standiardization:
    desc = deepcopy(standardized_calib_all)
    desc.pop("warping_on_bpu")
    desc.pop("enable_inv_trans")
    if "galaxy" in model_setting.lower():
        if "center_v" in desc:
            desc["center_v"] -= 100.0
        desc["image_height"] -= 100.0 * 2
    backbone["warping_module"] = dict(
        type="WarpingModule",
        max_side_length=max(input_hw),
        uv_map_desc=dict(
            type="AddDesc",
            per_tensor_desc=json.dumps(
                dict(
                    task_name="camera_standiardization",
                    shape=(1, *input_hw, 2),
                    **desc,
                )
            ),
            node_name="camera_standiardization",
        ),
    )

fpn_neck = dict(
    type="FPN",
    in_strides=[2, 4, 8, 16, 32, 64],
    in_channels=[16, 32, 64, 128, 256, 256],
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
    out_channel=32,
    bn_kwargs=bn_kwargs,
    node_name="fix_channel_neck",
)

ufpn_seg_neck = dict(
    type="UFPN",
    in_strides=[4, 8, 16, 32, 64],
    in_channels=[16, 32, 64, 128, 128],
    out_channels=[16, 32, 64, 128, 128],
    bn_kwargs=bn_kwargs,
    group_base=8,
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
    dict(type="BgrToYuv444", rgb_input=True),
    dict(
        type="TorchVisionAdapter",
        interface="Normalize",
        mean=128.0,
        std=128.0,
    ),
]
# clipping: begin
val_transforms = [
    dict(type="YUVTurboJPEGDecoder", to_string=True),
    dict(type="BPUPyramidResizer", scale_wh=(0.5, 0.5), pyramid_type="ips"),
    dict(type="ToTensor", to_yuv=False),
    dict(type="Normalize", mean=128.0, std=128.0),
]

if "galaxy" in model_setting.lower() or "test" in model_setting.lower():
    crop_img_path = dict(
        type="CropImgPatch",
        static_roi=(0, 100, 1920, 1124),
    )
    val_transforms.insert(1, crop_img_path)

if with_cam_standiardization:
    cam_stand_transform = dict(
        type="CameraStandardization",
        meta_key=(),
        **standardized_calib_all,
    )
    val_transforms.insert(0, cam_stand_transform)
# clipping: end

vis_tasks = []
