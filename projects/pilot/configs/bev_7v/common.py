import json
import os
import sys
from copy import deepcopy

import torch

from hat.models.backbones.mixvargenet import MixVarGENetConfig
from projects.pilot.configs.bev_7v.base import (
    STAGE1_SEG_DICT,
    ViewDomainFactory,
    get_block_warping,
    get_common_transforms,
    get_grid_quant_scale,
    get_narrow_block_warping,
    get_warp_sizes,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from project_common import (  # noqa
    enable_model_tracking,
    model_checkpoint,
    model_name_postfix,
    model_setting,
    model_thresh,
    model_version,
    num_machines,
    pack_infer_consist,
    pack_infer_vis,
    pipeline_test,
    resume_training,
    split_flag,
    tasks,
    training_step,
)

# model info
model_type = "pilot_multitask_bev_7v"
model_name = "_".join([model_type, model_setting, model_version])
if model_name_postfix:
    model_name = "_".join([model_name, model_name_postfix])

model_thresh = json.loads(model_thresh) if model_thresh else {}

if training_step == "pack_infer" and not model_thresh:
    raise ValueError("Must provide model_thresh for Pack infer.")

deploy_mode = training_step in ["int_infer", "pack_infer"]

if tasks is not None:
    tasks = json.loads(tasks)
else:
    bev_tasks = [
        dict(name="bev_3d_vehicle", important=True),
        dict(name="bev_3d_vrumerge", important=True),
        dict(name="online_mapping", important=True),
        dict(name="bev_arrow", important=True),
        dict(name="bev_junction", important=True),
        dict(name="bev_roadmarking", important=True),
        dict(name="bev_vismask", important=True),
        dict(name="bev_freespace", important=True),
    ]

    if not deploy_mode:
        # just for training
        bev_tasks.append(dict(name="bev_seg", important=True))
    else:
        # just for compile or pack_infer
        bev_tasks.append(dict(name="bev_crosspoint", important=True))
        if training_step == "pack_infer":
            # save common feature for consistence
            bev_tasks.append(dict(name="bev_each_view_feat", important=True))

    tasks = bev_tasks

bev_task_names = [t["name"] for t in tasks if "bev" in t["name"]]

is_local_train = not os.path.exists("/running_package")
bucket_root = "/horizon-bucket" if is_local_train else "/bucket/input"

batch_size_factor = 1

if is_local_train or pipeline_test:
    # batch size of each view input imgs
    train_batch_size_per_gpu = 1
    val_batch_size_per_gpu = 8
    train_num_workers = 1
    val_num_workers = 1
else:
    train_batch_size_per_gpu = int(2 * batch_size_factor)
    val_batch_size_per_gpu = 16
    train_num_workers = 1
    val_num_workers = 1

if is_local_train:
    log_freq = 5
    save_prefix = "tmp_output"
else:
    log_freq = 25
    save_prefix = "/job_data/models/"

# common config variable
do_val_visualize = False

# all support view camera
front_camera_view_names = ["camera_front"]
side_camera_view_names = [
    "camera_front_left",
    "camera_front_right",
    "camera_rear_left",
    "camera_rear_right",
    "camera_rear",
]
fisheye_camera_view_names = [
    "fisheye_front",
    "fisheye_rear",
    "fisheye_left",
    "fisheye_right",
]
narrow_camera_view_names = ["camera_front_30fov"]

# bev_7v view and shape
camera_view_names = (
    front_camera_view_names + side_camera_view_names + narrow_camera_view_names
)
num_views = len(camera_view_names)


# NOTE: The following information is for reference only,
# each dataset will replace some meta info.
# HxW
img_ori_size = {
    "camera_front": (2160, 3840),
    "camera_front_left": (1280, 1920),
    "camera_front_right": (1280, 1920),
    "camera_rear_left": (1280, 1920),
    "camera_rear_right": (1280, 1920),
    "camera_rear": (1280, 1920),
    "camera_front_30fov": (2160, 3840),
}

img_resize_scale = {
    "camera_front": 1 / 4,
    "camera_front_left": 1 / 2,
    "camera_front_right": 1 / 2,
    "camera_rear_left": 1 / 2,
    "camera_rear_right": 1 / 2,
    "camera_rear": 1 / 2,
    "camera_front_30fov": 1 / 4,
}

img_resize_size = {
    "camera_front": (540, 960),
    "camera_front_left": (640, 960),
    "camera_front_right": (640, 960),
    "camera_rear_left": (640, 960),
    "camera_rear_right": (640, 960),
    "camera_rear": (640, 960),
    "camera_front_30fov": (540, 960),
}

views_dist = [
    sum([name in front_camera_view_names for name in camera_view_names]),
    sum([name in side_camera_view_names for name in camera_view_names]),
    sum([name in fisheye_camera_view_names for name in camera_view_names]),
    sum([name in narrow_camera_view_names for name in camera_view_names]),
]

# WxH
img_resize_wh_size = dict()
for k, v in img_resize_size.items():
    img_resize_wh_size[k] = v[::-1]

vcs_range = (-35.2, -51.2, 105.6, 51.2)  # (bottom, right, top, left)
ego_ignore_range = (-0.6, -0.5, 2.0, 0.5)
ipm_output_size = (352, 256)
bevfusion_output_size = (704, 512)

# vcs_origin_coord , the number must be int
vcs_origin_coord = [
    int(
        vcs_range[2] / (vcs_range[2] - vcs_range[0]) * bevfusion_output_size[0]
    ),
    int(
        vcs_range[3] / (vcs_range[3] - vcs_range[1]) * bevfusion_output_size[1]
    ),
]

# spatial resolution in ipm feature
spatial_resolution = (
    abs(vcs_range[2] - vcs_range[0]) / ipm_output_size[0],
    abs(vcs_range[3] - vcs_range[1]) / ipm_output_size[1],
)  # (height, width)
# spatial resolution in bevfusion feature
bevfusion_spatial_resolution = (
    abs(vcs_range[2] - vcs_range[0]) / bevfusion_output_size[0],
    abs(vcs_range[3] - vcs_range[1]) / bevfusion_output_size[1],
)  # (height, width)

# WxH, 30fov same as front (960, 512)
front_input_size = (960, 512)
side_input_size = (960, 640)
narrow_input_size = (960, 512)

crop_roi = {
    "camera_front": (0, 0) + front_input_size,
    "camera_front_left": (0, 0) + side_input_size,
    "camera_front_right": (0, 0) + side_input_size,
    "camera_rear_left": (0, 0) + side_input_size,
    "camera_rear_right": (0, 0) + side_input_size,
    "camera_rear": (0, 0) + side_input_size,
    "camera_front_30fov": (0, 0) + narrow_input_size,
}

stage1_out_stride = 4
H_persp_view_scale = 1 / stage1_out_stride
vcs_plane_heights = (0.0, 0.5, 1.0, 1.5)

# train setting
use_stage1_loss = False
use_split_dataloader = True
store_homo_offset = True
offset_save_path = None
cal_homo_offset_on_gpu = True
train_global_sample_interval = 1

homo_offset_info_keys = [
    "T_vcs2cam",
    "intrinsics",
    "distort_coeffs",
    "transformats",
    "ipm_img_sizes",
    "img_shape",
    "fake_homo_flag",
    "aug_flag",
]

if store_homo_offset:
    if offset_save_path is None:
        offset_save_path = os.path.join(
            save_prefix, f"{model_name}/bev_homo_offset"
        )
    if cal_homo_offset_on_gpu:
        homo_offset_info_keys.append("homo_offset")

seg_class = 4
remap_dict = (
    dict(gt_seg=STAGE1_SEG_DICT[f"33_{str(seg_class)}"])
    if use_stage1_loss
    else {}
)

drop_view_prob = 0.0
block_warp_padding = get_block_warping(vcs_range, spatial_resolution)
narrow_block_warp_padding = get_narrow_block_warping(
    vcs_range, spatial_resolution
)

warp_sizes = get_warp_sizes(ipm_output_size, block_warp_padding)
narrow_warp_sizes = get_warp_sizes(ipm_output_size, narrow_block_warp_padding)

camera_view_names = ViewDomainFactory.get_valid_camera_view_names(
    camera_view_names
)
views_domain2nums = ViewDomainFactory.get_views_domain2nums(camera_view_names)

common_transforms = get_common_transforms(
    size=list(img_resize_size.values()),
    remap_dict=remap_dict,
    crop_roi=list(crop_roi.values()),
    views_domain2nums=views_domain2nums,
    train_num_frames_per_iter=None,
)

# ---------------------------- inputs ------------------------
inputs = dict(
    img=torch.zeros((1, 3, *front_input_size[::-1])),
    side_img=torch.zeros((5, 3, *side_input_size[::-1])),
    narrow_img=torch.zeros((1, 3, *narrow_input_size[::-1])),
)

if cal_homo_offset_on_gpu:
    inputs.update(
        dict(
            meta_info=dict(
                T_vcs2cam=[torch.zeros(1, 1, 4, 4, dtype=torch.float32)]
                * num_views,
                intrinsics=[torch.zeros(1, 1, 3, 3, dtype=torch.float32)]
                * num_views,
                distort_coeffs=[torch.zeros(1, 8, dtype=torch.float32)]
                * num_views,
                transformats=[torch.zeros(1, 1, 3, 3, dtype=torch.float32)]
                * num_views,
                ipm_img_sizes=[torch.zeros(1, 2, dtype=torch.float32)]
                * num_views,
                img_shape=[torch.zeros(1, 1, 2, dtype=torch.float32)]
                * num_views,
                fake_homo_flag=torch.zeros(
                    1, len(vcs_plane_heights) * num_views, dtype=torch.float32
                ),
                aug_flag=[False],
                homography=torch.randn(
                    (1, num_views * len(vcs_plane_heights), 3, 3)
                ),
                homo_offset=torch.randn(
                    (num_views * len(vcs_plane_heights),)
                    + ipm_output_size
                    + (2,)
                ),
            ),
            view=views_domain2nums,
            temporal_info=dict(
                num_frames_per_iter=[
                    1,
                ]
            ),
        )
    )
else:
    inputs.update(
        dict(
            meta_info=dict(
                homo_offset=torch.randn(
                    (num_views * len(vcs_plane_heights),)
                    + ipm_output_size
                    + (2,)
                ),
                homography=torch.randn(
                    (1, num_views * len(vcs_plane_heights), 3, 3)
                ),
            ),
        )
    )

# torch.jit.trace should consistent type, e.g.,
# dict(
#   img=tensor,
#   offset=tensor,
# )
# or
# dict(
#   img=List[tensor],
#   offset=List[tensor],
# )
deploy_stage1_inputs = dict()
deploy_stage1_outputs = dict()

for i, view_key in enumerate(camera_view_names):
    stage1_out_key = f"OutputModule{i + 1}"
    deploy_stage1_outputs[stage1_out_key] = torch.zeros((1, 4, 128, 240))
    if view_key in front_camera_view_names:
        deploy_stage1_inputs[view_key] = torch.zeros(
            (1, 3, *front_input_size[::-1])
        )
    elif view_key in side_camera_view_names:
        deploy_stage1_inputs[view_key] = torch.zeros(
            (1, 3, *side_input_size[::-1])
        )
    elif view_key in narrow_camera_view_names:
        deploy_stage1_inputs[view_key] = torch.zeros(
            (1, 3, *narrow_input_size[::-1])
        )
    else:
        raise TypeError

deploy_stage2_inputs = dict()
deploy_homo_offset_key = []
idx = 0
for s in warp_sizes + narrow_warp_sizes:
    for _ in range(len(vcs_plane_heights)):
        offset_key = f"homo_offset_{idx}"
        deploy_stage2_inputs[offset_key] = torch.randn((1, s[0], s[1], 2))
        deploy_homo_offset_key.append(offset_key)
        idx += 1

deploy_inputs = dict()
deploy_inputs.update(deploy_stage1_inputs)
deploy_inputs.update(deploy_stage2_inputs)
deploy_stage2_inputs.update(deploy_stage1_outputs)

opt_inputs = dict(
    train={},
    val={},
    deploy={},
)

# for torch.jit.trace
trace_inputs = deepcopy(deploy_inputs)


# ----------------------------- model ------------------------
bn_kwargs = dict(eps=1e-5, momentum=0.1)

# backbone & neck structures
mixvargenet_config = [
    [
        MixVarGENetConfig(
            in_channels=32,
            out_channels=32,
            head_op="mixvarge_f2",
            stack_ops=[],
            stack_factor=1,
            stride=1,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 2
    [
        MixVarGENetConfig(
            in_channels=32,
            out_channels=32,
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 4
    [
        MixVarGENetConfig(
            in_channels=32,
            out_channels=64,
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 8
    [
        MixVarGENetConfig(
            in_channels=64,
            out_channels=96,
            head_op="mixvarge_f2_gb16",
            stack_ops=[
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
            ],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 16
    [
        MixVarGENetConfig(
            in_channels=96,
            out_channels=160,
            head_op="mixvarge_f2_gb16",
            stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 32
]


backbone = dict(
    type="MixVarGENet",
    net_config=mixvargenet_config,
    output_list=[0, 1, 2, 3, 4],
    input_channels=3,
    num_classes=1000,
    bn_kwargs=bn_kwargs,
    include_top=False,
    bias=True,
    disable_quanti_input=False,
    node_name="backbone",
)

side_backbone = dict(
    type="MixVarGENet",
    net_config=mixvargenet_config,
    output_list=[0, 1, 2, 3, 4],
    input_channels=3,
    num_classes=1000,
    bn_kwargs=bn_kwargs,
    include_top=False,
    bias=True,
    disable_quanti_input=False,
    node_name="side_backbone",
)

narrow_backbone = dict(
    type="MixVarGENet",
    net_config=mixvargenet_config,
    output_list=[0, 1, 2, 3, 4],
    input_channels=3,
    num_classes=1000,
    bn_kwargs=bn_kwargs,
    include_top=False,
    bias=True,
    disable_quanti_input=False,
    node_name="narrow_backbone",
)

pafpn_neck = dict(
    type="PAFPN",
    in_channels=[32, 32, 64, 96, 160],
    out_channels={4: 32, 8: 64, 16: 96, 32: 160, 64: 320},
    out_strides=[4, 8, 16, 32, 64],
    start_level=1,
    add_extra_convs="on_output",  # use P5
    num_outs=5,
    relu_before_extra_convs=True,
    node_name="bifpn_neck",
)

side_pafpn_neck = dict(
    type="PAFPN",
    in_channels=[32, 32, 64, 96, 160],
    out_channels={4: 32, 8: 64, 16: 96, 32: 160, 64: 320},
    out_strides=[4, 8, 16, 32, 64],
    start_level=1,
    add_extra_convs="on_output",  # use P5
    num_outs=5,
    relu_before_extra_convs=True,
    node_name="side_bifpn_neck",
)

narrow_pafpn_neck = dict(
    type="PAFPN",
    in_channels=[32, 32, 64, 96, 160],
    out_channels={4: 32, 8: 64, 16: 96, 32: 160, 64: 320},
    out_strides=[4, 8, 16, 32, 64],
    start_level=1,
    add_extra_convs="on_output",  # use P5
    num_outs=5,
    relu_before_extra_convs=True,
    node_name="narrow_bifpn_neck",
)

head = dict(
    type="OutputModule",
    convert_to_dict=False,
    head=dict(
        type="PixelHead",
        # feature_name="feats",
        in_strides=[4, 8, 16, 32, 64],
        out_strides=[4],
        stride2channels={4: 32, 8: 64, 16: 96, 32: 160, 64: 320},  # noqa
        forward_frame_idxs=[0],
        out_nums=seg_class,
        output_name="front_pred_segs",
        quanti_last_conv=True,
        dequant_out=False,
        bn_kwargs=bn_kwargs,
        group_base=8,
        batchnorm_output=True,
    ),
    node_name="front_bev_stage1_head",
)
deploy_head = deepcopy(head)
deploy_head["postprocess"] = dict(
    type="AddDesc",
    per_tensor_desc=json.dumps(
        dict(
            task="bev",
            roi_input=dict(
                fp_x=960 // 2,
                fp_y=540 // 2,
                width=960,
                height=512,
            ),
            vanishing_point=[960 // 2, 540 // 2],
        )
    ),
)
deploy_head["head"]["dequant_out"] = True

side_head = dict(
    type="OutputModule",
    convert_to_dict=False,
    head=dict(
        type="PixelHead",
        # feature_name="feats",
        in_strides=[4, 8, 16, 32, 64],
        out_strides=[4],
        stride2channels={4: 32, 8: 64, 16: 96, 32: 160, 64: 320},  # noqa
        forward_frame_idxs=[0],
        out_nums=seg_class,
        output_name="side_pred_segs",
        quanti_last_conv=True,
        dequant_out=False,
        bn_kwargs=bn_kwargs,
        group_base=8,
        batchnorm_output=True,
    ),
    node_name="side_bev_stage1_head",
)
deploy_side_head = deepcopy(side_head)
deploy_side_head["postprocess"] = dict(
    type="AddDesc",
    per_tensor_desc=json.dumps(
        dict(
            task="bev",
            roi_input=dict(
                fp_x=960 // 2,
                fp_y=640 // 2,
                width=960,
                height=640,
            ),
            vanishing_point=[960 // 2, 640 // 2],
        )
    ),
)
deploy_side_head["head"]["dequant_out"] = True

narrow_head = dict(
    type="OutputModule",
    convert_to_dict=False,
    head=dict(
        type="PixelHead",
        in_strides=[4, 8, 16, 32, 64],
        out_strides=[4],
        stride2channels={4: 32, 8: 64, 16: 96, 32: 160, 64: 320},  # noqa
        forward_frame_idxs=[0],
        out_nums=seg_class,
        output_name="narrow_pred_segs",
        quanti_last_conv=True,
        dequant_out=False,
        bn_kwargs=bn_kwargs,
        group_base=8,
        batchnorm_output=True,
    ),
    node_name="narrow_bev_stage1_head",
)
deploy_narrow_head = deepcopy(narrow_head)
deploy_narrow_head["postprocess"] = dict(
    type="AddDesc",
    per_tensor_desc=json.dumps(
        dict(
            task="bev",
            roi_input=dict(
                fp_x=960 // 2,
                fp_y=540 // 2,
                width=960,
                height=512,
            ),
            vanishing_point=[960 // 2, 540 // 2],
        )
    ),
)
deploy_narrow_head["head"]["dequant_out"] = True

multi_view_collect = dict(
    type="FlattenCollect",
    node_name="flatten_collect",
)

# bev fusiong & upsampling & roi resize module
view_max_size = 256
grid_quant_scale = get_grid_quant_scale(max(ipm_output_size), view_max_size)

use_random_rotation = False
random_rotation_cfg = None
if use_random_rotation:
    random_rotation_cfg = dict(
        type="RandomRotation",
        grid_quant_scale=grid_quant_scale,
        height=ipm_output_size[0],
        width=ipm_output_size[1],
        angles=(0, 90, 180, 270),
    )

if cal_homo_offset_on_gpu:
    generate_offset_module = dict(
        type="ANCGenerateBEVHomOffset",
        vcs_range=vcs_range,
        spatial_resolution=spatial_resolution,
        vcs_plane_heights=vcs_plane_heights,
    )

model_valid_views = []
# remove zeros in bev fusion module.
for view in views_dist:
    if view > 0:
        model_valid_views.append(view)

bev_fusion = dict(
    train=dict(
        type="ANCBEVFusionModule",
        grid_quant_scale=grid_quant_scale,
        random_rotation_cfg=random_rotation_cfg,
        drop_view_prob=drop_view_prob,
        views=model_valid_views,
        ipm_output_size=ipm_output_size,
        use_homo_offset=True,
        block_warp_padding=block_warp_padding + narrow_block_warp_padding,
        vcs_plane_nums=len(vcs_plane_heights),
        generate_offset_module=generate_offset_module
        if cal_homo_offset_on_gpu
        else None,
        homo_offset_info_keys=homo_offset_info_keys
        if cal_homo_offset_on_gpu
        else None,
        node_name="bev_fusion",
    ),
    val=dict(
        type="ANCBEVFusionModule",
        grid_quant_scale=grid_quant_scale,
        random_rotation_cfg=random_rotation_cfg,
        drop_view_prob=drop_view_prob,
        views=model_valid_views,
        ipm_output_size=ipm_output_size,
        use_homo_offset=True,
        block_warp_padding=block_warp_padding + narrow_block_warp_padding,
        vcs_plane_nums=len(vcs_plane_heights),
        generate_offset_module=None,
        homo_offset_info_keys=None,
        node_name="bev_fusion",
    ),
    deploy=dict(
        type="ANCBEVFusionModule",
        grid_quant_scale=grid_quant_scale,
        random_rotation_cfg=random_rotation_cfg,
        drop_view_prob=drop_view_prob,
        views=model_valid_views,
        ipm_output_size=ipm_output_size,
        use_homo_offset=True,
        block_warp_padding=block_warp_padding + narrow_block_warp_padding,
        vcs_plane_nums=len(vcs_plane_heights),
        generate_offset_module=None,
        homo_offset_info_keys=None,
        compile_model=True,
        node_name="bev_fusion",
    ),
)
bev_fusion_upsampling = dict(
    type="ResizeParser",
    use_plugin_interpolate=True,
    dequant_out=False,
    resize_kwargs=dict(size=bevfusion_output_size, mode="bilinear"),
    node_name="bev_fusion_upsampling",
)

# bev backbone neck
mixvargenet_2x_config = [
    [
        MixVarGENetConfig(
            in_channels=32,
            out_channels=64,
            head_op="mixvarge_f2",
            stack_ops=[],
            stack_factor=1,
            stride=1,
            fusion_strides=[],
            extra_downsample_num=0,
        )
    ],  # stride 2
    [
        MixVarGENetConfig(
            in_channels=64,
            out_channels=64,
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        )
    ],  # stride 4
    [
        MixVarGENetConfig(
            in_channels=64,
            out_channels=128,
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        )
    ],  # stride 8
    [
        MixVarGENetConfig(
            in_channels=128,
            out_channels=256,
            head_op="mixvarge_f2_gb16",
            stack_ops=[
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
            ],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        )
    ],  # stride 16
    [
        MixVarGENetConfig(
            in_channels=256,
            out_channels=256,
            head_op="mixvarge_f2_gb16",
            stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        )
    ],  # stride 32
]

bev_backbone = dict(
    type="MixVarGENet",
    net_config=mixvargenet_2x_config,
    output_list=[0, 1, 2, 3, 4],
    input_channels=seg_class * len(vcs_plane_heights),
    input_sequence_length=1,
    input_resize_scale=None,
    num_classes=1000,
    include_top=False,
    bn_kwargs=bn_kwargs,
    bias=True,
    disable_quanti_input=True,
    node_name="bev_stage2_backbone",
)

bev_neck = dict(
    type="Unet",
    in_strides=[2, 4, 8, 16, 32],
    out_strides=[2, 4, 8, 16, 32],
    stride2channels={
        2: 64,
        4: 64,
        8: 128,
        16: 256,
        32: 256,
        64: 256,
        128: 256,
        256: 256,
    },
    out_stride2channels={
        2: 48,
        4: 48,
        8: 96,
        16: 192,
        32: 192,
        64: 192,
        128: 192,
        256: 192,
    },
    factor=2,
    use_bias=False,
    group_base=8,
    node_name="bev_stage2_neck",
)
