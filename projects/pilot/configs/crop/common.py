import json
import os
import sys

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
model_type = "pilot_multitask_crop"
model_name = "_".join([model_type, model_setting, model_version])

if model_name_postfix:
    model_name = "_".join([model_name, model_name_postfix])

if model_thresh is not None:
    model_thresh = json.loads(model_thresh)

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
        dict(name="vehicle_ground_line", important=True),
        # dict(name="vehicle_flank", important=True),
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

# dataset
ds_path = os.path.join(
    os.path.dirname(__file__),
    f"../datasets/{model_setting.lower()}_datasets.py",
)
assert os.path.exists(ds_path)
ds_cfg = Config.fromfile(ds_path)
datapaths = ds_cfg.datapaths

is_local_train = not os.path.exists("/running_package")
batch_size_factor = 1
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

input_hw = resize_hw = (192, 512)
if "x3c" in model_setting.lower():
    if "cc02" in model_setting.lower():
        roi_region = (704, 568, 1216, 760)

    elif "sedan" in model_setting.lower():
        roi_region = (704, 528, 1216, 720)
    elif "suv" in model_setting.lower():
        roi_region = (704, 576, 1216, 768)
    elif "c385" in model_setting.lower():
        roi_region = (704, 528, 1216, 720)
    elif "ev52" in model_setting.lower():
        roi_region = (704, 576, 1216, 768)
    else:
        raise NotImplementedError
    vanishing_point = (960, 640)
    origin_img_hw = (1280, 1920)
else:
    roi_region = (768, 568, 1280, 760)
    vanishing_point = (int(2048 / 2), int(1280 / 2))
    origin_img_hw = (1280, 2048)

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


test_roi_num = 50

# common structures
backbone = dict(
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

val_decoders = {}

val_transforms = [
    dict(type="YUVTurboJPEGDecoder", to_string=True),
    dict(type="CropImgPatch", static_roi=roi_region),
    dict(type="ImgBufToYUV444"),
    dict(type="ToTensor", to_yuv=False),
    dict(type="Normalize", mean=128.0, std=128.0),
]

vis_tasks = []
