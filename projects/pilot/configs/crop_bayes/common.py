import json
import os
import sys

from hat.models.backbones.mixvargenet import MixVarGENetConfig
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

# model info
model_type = "pilot5_multitask_crop_bayes"
model_name = "_".join([model_type, model_setting, model_version])

if model_name_postfix:
    model_name = "_".join([model_name, model_name_postfix])

if model_thresh is not None:
    model_thresh = json.loads(model_thresh)

input_scale = (1, 1)

# clipping: begin
if "niofy" in model_setting.lower():
    input_scale = (2, 2)  # (h_scale, w_scale), must be tuple
# clipping: end

with_tracking_feature = True
if tasks is not None:
    tasks = json.loads(tasks)
else:
    # tasks
    tasks = [
        # vehicle
        dict(name="vehicle_detection", important=True),
        dict(name="vehicle_category_classification", important=True),
        dict(name="vehicle_occlusion_classification", important=True),
        dict(name="vehicle_wheel_detection", important=True),
        # dict(name="vehicle_ground_line", important=True),
        dict(name="vehicle_flank", important=True),
        # rear
        dict(name="rear_detection", important=True),
        dict(name="rear_occlusion_classification", important=True),
        dict(name="rear_part_classification", important=True),
        # person
        dict(name="person_detection", important=True),
        dict(name="person_occlusion_classification", important=True),
        dict(name="person_orientation_classification", important=True),
        dict(name="person_pose_classification", important=True),
        # cyclist
        dict(name="cyclist_detection", important=True),
    ]
    # clipping: begin
    if "hc23" in model_setting.lower():
        new_tasks = [dict(name="cyclist_classification", important=True)]
        deprecated_tasks = [
            "vehicle_wheel_detection",
            "vehicle_flank",
        ]
        tasks.extend(new_tasks)
        tasks = [t for t in tasks if t["name"] not in deprecated_tasks]
    elif "niofy" in model_setting.lower():
        tasks += [dict(name="cyclist_classification", important=True)]


# task setting
with_tracking_feature = "hc23" not in model_setting.lower()
# clipping: end

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
    # default during eval
    ds_path = os.path.join(
        os.path.dirname(__file__),
        "../datasets/galaxy_0233_rear_lmdb_datasets.py",
    )
    ds_cfg = Config.fromfile(ds_path)
    datapaths = ds_cfg.datapaths


lmdb_data = "lmdb" in model_setting.lower()

# training
is_local_train = not os.path.exists("/running_package")
if "niofy" in model_setting.lower():
    batch_size_factor = 1.5
else:
    batch_size_factor = 2

if is_local_train:
    batch_size = 12
    if pipeline_test:
        batch_size = 1
    batch_size = int(batch_size * batch_size_factor)
    log_freq = 5
    save_prefix = "tmp_output"
else:
    batch_size = int(16 * batch_size_factor)
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

pred_batch_size = 64

# now only train with rear input_hw,
# and the model will be used on rear and side.

input_hw = resize_hw = (192, 512)
# set from roi_region = (704, 568, 1216, 760) by crop top 100
roi_region = (704, 568 - 100, 1216, 760 - 100)
vanishing_point = (int(1920 / 2), int(1080 / 2))
origin_img_hw = (1080, 1920)

# clipping: begin
if "galaxy_0233_rear" in model_setting.lower():
    input_hw = resize_hw = (192, 512)
    # set from roi_region = (704, 568, 1216, 760) by crop top 100
    roi_region = (704, 568 - 100, 1216, 760 - 100)
    vanishing_point = (int(1920 / 2), int(1080 / 2))
    origin_img_hw = (1080, 1920)
elif "galaxy_x3c_side" in model_setting.lower():
    input_hw = resize_hw = (192, 384)
    vanishing_point = (int(1920 / 2), int(1280 / 2))
    roi_region = (1536, 432, 1920, 624)  # right rear
    origin_img_hw = (1280, 1920)
    # roi_region = (0, 431, 384, 623)  # left rear
elif "hc23" in model_setting.lower():
    input_hw = resize_hw = (192, 512)
    origin_img_hw = (1280, 1920)
    roi_region = (704, 624, 1216, 816)
    vanishing_point = (int(1920 / 2), int(1280 / 2))

elif "niofy" in model_setting.lower() and "x3c_rear" in model_setting.lower():
    input_hw = resize_hw = (192, 512)
    roi_region = (704, 568, 1216, 760)
    vanishing_point = (int(1920 / 2), int(1280 / 2))
    origin_img_hw = (1280, 1920)
elif "niofy" in model_setting.lower() and "x3c_side" in model_setting.lower():
    input_hw = resize_hw = (192, 384)
    vanishing_point = (int(1920 / 2), int(1280 / 2))
    roi_region = (1536, 528, 1920, 720)  # right rear
    origin_img_hw = (1280, 1920)
    # roi_region = (0, 528, 384, 720)  # left rear
# clipping: end

bn_kwargs = dict(eps=1e-5, momentum=0.1)

# data_config
inter_method = 10
pixel_center_aligned = False
min_valid_clip_area_ratio = 0.5
rand_translation_ratio = 0.1

rand_translation_ratio = 0.0
rand_crop_scale_range = (0.8, 1.0 / 0.8)
center_crop_scale_range = (0.8, 1.0 / 0.8)
norm_scale = 1.0


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

# # common structures

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
            stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
            stack_factor=1,
            stride=2,
            fusion_strides=[8, 16],
            extra_downsample_num=0,
        ),  # stride 32
    ],
]

backbone = dict(
    type="ResizePatcher",
    backbone=dict(
        type="MixVarGENet",
        net_config=mixvargenet_config,
        output_list=[0, 1, 2, 3, 4],
        input_channels=3,
        num_classes=1000,
        bn_kwargs=bn_kwargs,
        include_top=False,
        bias=True,
        node_name="backbone",
    ),
    input_scale=input_scale,
)

fpn_neck = dict(
    type="FPN",
    in_strides=[2, 4, 8, 16, 32],
    in_channels=[16, 32, 64, 128, 256],
    out_strides=[4, 8, 16, 32],
    out_channels=[16, 32, 64, 128],
    bn_kwargs=bn_kwargs,
    node_name="fpn_neck",
)

fix_channel_neck = dict(
    type="FixChannelNeck",
    in_strides=[4, 8, 16, 32],
    in_channels=[16, 32, 64, 128],
    out_strides=[8, 16, 32],
    out_channel=32,
    bn_kwargs=bn_kwargs,
    node_name="fix_channel_neck",
)

fix_downscale_neck = (
    dict(
        type="ConvDownscaleNeck",
        in_strides=[8, 16, 32],
        in_channels=[32, 32, 32],
        out_strides=[8 * 2, 16 * 2, 32 * 2],
        out_channels=[32, 32, 32],
        bn_kwargs=bn_kwargs,
        node_name="fix_downscale_neck",
    )
    if "niofy" in model_setting.lower()
    else None
)


val_decoders = {}

val_transforms = [
    dict(type="YUVTurboJPEGDecoder", to_string=True),
    dict(type="CropImgPatch", static_roi=roi_region),
    dict(type="ImgBufToYUV444"),
    dict(type="ToTensor", to_yuv=False),
    dict(type="Normalize", mean=128.0, std=128.0),
]

vis_tasks = []
