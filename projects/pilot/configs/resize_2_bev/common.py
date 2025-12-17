import json
import os
import sys
from copy import deepcopy

from hat.models.backbones.mixvargenet import MixVarGENetConfig
from hat.utils import Config

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from project_common import (  # noqa
    enable_model_tracking,
    model_checkpoint,
    model_name_postfix,
    model_setting,
    model_thresh,
    model_version,
    num_machines,
    pipeline_test,
    resume_training,
    split_mode,
    tasks,
    training_step,
)

sys.modules.pop("project_common")

# model info
model_type = "pilot_multitask_resize2_bev"
model_name = "_".join([model_type, model_setting, model_version])
if model_name_postfix:
    model_name = "_".join([model_name, model_name_postfix])

if model_thresh is not None:
    model_thresh = json.loads(model_thresh)

with_cam_standiardization = False

if tasks is not None:
    tasks = json.loads(tasks)
else:
    # tasks
    dense_tasks = [
        dict(name="default_segmentation", important=True),
    ]

    if "parking" not in model_setting:
        dense_tasks.append(dict(name="lane_segmentation", important=True))

    om_task = [
        dict(name="bev_om", important=True),
    ]
    bev_tasks = [
        # dict(name="bev_om", important=True),
        # dict(name="bev_3d_vehicle", important=True),
        dict(name="bev_3d_vehicle_cls", important=True),
        dict(name="bev_3d_pedestrian", important=True),
        # dict(name="bev_3d_cyclist", important=True),
        dict(name="bev_3d_cyclist_cls", important=True),
    ]
    default_parsing = [
        dict(name="default_segmentation", important=True),
    ]
    lane_parsing = [
        dict(name="lane_segmentation", important=True),
    ]
    desensetization_tasks = [
        dict(name="face_detection", important=True),
        dict(name="vehicle_plate_detection", important=True),
    ]

    # tasks = bev_tasks
    tasks = bev_tasks + default_parsing + lane_parsing + desensetization_tasks

bev_task_names = [t["name"] for t in tasks if "bev_3d" in t["name"]]

# currrently we do not seperate bev tasks to multi stages
# which require all inputs need to multiply view_num

# dataset
ds_path = os.path.join(
    os.path.dirname(__file__),
    f"../datasets/{model_setting.lower()}_datasets.py",
)
if os.path.exists(ds_path):
    datapaths = Config.fromfile(ds_path).datapaths

is_local_train = not os.path.exists("/running_package")
merged_bev3d = True  # set True if bev3d tasks involved for faster training

if is_local_train:
    batch_size = 4
    batch_size_bev = 2
    log_freq = 5
    save_prefix = "tmp_output"
else:
    batch_size = 10
    # TODO (zihan.qiu): set batch to 6 when MultiStageBatchProcessor ready
    batch_size_bev = 5 if merged_bev3d else 6
    log_freq = 25
    save_prefix = "/job_data/models/"

pred_batch_size = 28
pred_batch_size_bev = 2

lmdb_data = "lmdb" in model_setting.lower()

# bev view and shape
# currrently we do not seperate bev tasks to multi stages
# which require all inputs need to multiply view_num

view_num = 5 if any(["bev" in t["name"] for t in tasks]) else 1
# set view_num = 6 to support front view
# view_num = 6
assert view_num in [4, 5, 6]

resize_factor = 0.5
raw_image_hw = (1280, 1920) if "x3c" in model_setting.lower() else (1280, 2048)

# # # hard code image hw for compile
# raw_image_hw = (1280, 1920)


input_hw = resize_hw = [int(x * resize_factor) for x in raw_image_hw]
roi_region = (0, 0, input_hw[1], input_hw[0])
vanishing_point = (int(input_hw[1] / 2), int(input_hw[0] / 2))

# front image info
front_resize_factor = 0.25
raw_front_image_hw = (2160, 3840)
front_resize_hw = [int(x * front_resize_factor) for x in raw_front_image_hw]
front_roi_region = (0, 0, front_resize_hw[1], front_resize_hw[0] - 28)
front_input_hw = [
    front_roi_region[3] - front_roi_region[1],
    front_roi_region[2] - front_roi_region[0],
]


bn_kwargs = dict(eps=1e-5, momentum=0.1)
# data_config
inter_method = 10
pixel_center_aligned = False
min_valid_clip_area_ratio = 0.5
rand_translation_ratio = 0.1

# input type config
process_type = "cat"
input_cat = dict(
    type="ListInputPreprocess",
    need_quant=True,
    need_cat=True,
    node_name="input_cat",
)
# 3d config


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


test_roi_num = 100

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
            stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
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
    disable_quanti_input=True
    if process_type == "cat" and not split_mode
    else False,
    node_name="backbone",
)
if with_cam_standiardization:
    backbone["warping_module"] = dict(
        type="WarpingModule", max_side_length=max(input_hw)
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
val_decoders = {}

val_transforms = [
    dict(type="YUVTurboJPEGDecoder", to_string=True),
    dict(
        type="BPUPyramidResizer",
        scale_wh=(resize_factor, resize_factor),
        pyramid_type="ips",
    ),
    dict(type="ToTensor", to_yuv=False),
    dict(type="Normalize", mean=128.0, std=128.0),
]

# val_transforms = [
#     # dict(type="BgrToYuv444", rgb_input=True),
#     dict(
#         type="TorchVisionAdapter",
#         interface="Normalize",
#         mean=128.0,
#         std=128.0,
#     ),
# ]


vis_tasks_2d = []
vis_tasks_bev = []
