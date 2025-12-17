import datetime
import json
import os
from pathlib import Path

import torch
from hatbc.filestream.bucket.client import get_bucket_mount_root
from horizon_plugin_pytorch.quantization import March

from hat.core.proj_spec.classification import (
    get_mono_classification_desc,
    get_rcnn_classification_desc,
)

job_name = "TrafficSignMultitask2PE-types258-occ4"
model_version = "v1.1.9"
ctimestr = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
job_name = f"{job_name}_{model_version}"
task_name = "mono_multitask_2pe_traffic_sign"
location = "CN"

# ---------------------------------------------------------
bucket2mount_root = get_bucket_mount_root()

# mount bucket to local manually: dmp mount exec mono ./mono
buckets = [
    "mono",
]
# use mount_root to construct rec, json path
mount_root = bucket2mount_root.get(buckets[0], None)
if mount_root is None:
    raise FileNotFoundError("mono bucket")

pipeline_test = os.environ.get("HAT_PIPELINE_TEST", "0") == "1"
is_local_train = not os.path.exists("/running_package")
march = March.BERNOULLI2


config_file = "projects/" + os.path.abspath(__file__).split("projects/")[-1]
hat_root = os.path.abspath(__file__).split("projects/")[0]
config_file_root = os.path.dirname(config_file)
abs_config_file_root = os.path.abspath(config_file_root)

save_prefix = "tmp_output" if is_local_train else "/job_data/models/"
ckpt_dir = Path(save_prefix) / f"{task_name}_{model_version}"
log_dir = ckpt_dir / "logs"

# ------------------------------------------------------
# train
# ------------------------------------------------------
num_machines = 1
num_gpus_per_machine = 8
device_ids = (
    [0, 1, 2, 3] if is_local_train else list(range(num_gpus_per_machine))
)
log_freq = 5 if is_local_train else 25
float_steps = 10 if pipeline_test else 50000
qat_steps = 10 if pipeline_test else 5000
warmup_steps = 0 if pipeline_test else 500

float_lr = 0.01
qat_lr = 0.00005

save_interval = 50 if pipeline_test else 1000
interval_by = "step"
pretrain_checkpoint = None
# -------------------------------------------------------
# input
# -------------------------------------------------------
input_hw = (64, 64)
deploy_input_hw = (64, 64)
resize_hw = None
inter_method = 10
pixel_center_aligned = False
min_valid_clip_area_ratio = 0.5
rand_translation_ratio = 0.0


# -------------------------------------------------------
# dataloader
# -------------------------------------------------------
roi_point_id = 4
lt_id = 0
rb_id = 2

norm_length = 64
norm_method = "resize"  # change as norm_point_id1/2 change
padding_context = None  # useful when norm_method = resize

# -------------------------------------------------------
# multitask shared model
# ------------------------------------------------------
sync_bn = True
bn_kwargs = dict(eps=1e-5, momentum=0.1)
backbone = dict(
    type="TinyVargNetV2",
    num_classes=1000,
    bn_kwargs=bn_kwargs,
    alpha=0.5,
    group_base=8,
    include_top=False,
    extend_features=False,
    input_resize_scale=None,
    channel_list=[32, 32, 64, 128, 256],
    node_name="backbone",
)


def get_stride2channels():
    pc = [int(c * backbone["alpha"]) for c in backbone["channel_list"]]
    s2c = {2 ** (idx + 1): c for idx, c in enumerate(pc)}
    return s2c, pc


stride2channels, backbone_channels = get_stride2channels()


# -------------------------------------------------------
# tracking_feature
# ------------------------------------------------------
last_stride = 2 ** len(backbone["channel_list"])
tracking_feature_desc = {
    "task": "tracking_feature",
    "size": [
        1,
        input_hw[0] // last_stride,
        input_hw[1] // last_stride,
        int(backbone["channel_list"][-1] * backbone["alpha"]),
    ],
}
add_tracking_feature_desc = dict(
    type="AddDesc",
    strict=True,
    per_tensor_desc=[json.dumps(tracking_feature_desc)],
    node_name="traffic_sign_tracking_feature_desc",
)
tracking_module = dict(
    type="TrafficSignClassifierMultitask",
    backbone=backbone,
    backbone_extra=torch.nn.Identity(),
    prediction_head=dict(
        type="ClassificationTrackHead",
        node_name="traffic_sign_tracking_feature_head",
    ),
    losses=None,
    desc=add_tracking_feature_desc,
)


# classification specific
def get_classification_model_desc(
    task_name,
    output_name,
    class_name,
    desc_id,
    prediction_return_cnt=1,
):
    if get_train_step() != "int_infer":
        per_tensor_desc = get_rcnn_classification_desc(
            "frcnn_classification", output_name, class_name, desc_id
        )
        per_tensor_desc = json.loads(per_tensor_desc)
        per_tensor_desc.update(
            crop_desc=dict(
                norm_len=norm_length,
                norm_method=norm_method,
                image_size=input_hw,
                padding=None,
            ),
        )
        classification_model_desc = dict(
            type="AddDesc",
            strict=True,
            per_tensor_desc=[
                json.dumps(per_tensor_desc)
                for _ in range(prediction_return_cnt)
            ],
            node_name=f"{output_name}_classification_desc",
        )
        return classification_model_desc
    else:
        per_tensor_desc = get_mono_classification_desc(
            task_name, output_name, class_name, desc_id
        )
        per_tensor_desc = json.loads(per_tensor_desc)
        per_tensor_desc.update(
            dict(
                location=location,
                image_size=input_hw,
                norm_method=norm_method,
                padding=[0.0, 0.0, 0.0, 0.0],
            ),
        )
        classification_model_desc = dict(
            type="AddDesc",
            strict=True,
            per_tensor_desc=[
                json.dumps(per_tensor_desc)
                for _ in range(prediction_return_cnt)
            ],
            node_name=f"{output_name}_classification_desc",
        )
        return classification_model_desc


# -------------------------------------------------------
# common_func
# ------------------------------------------------------


def get_train_step():
    return os.environ.get("HAT_TRAINING_STEP", "float")


def get_task_model(mode, task_configs):
    task_model = dict(
        type="MultitaskGraphModel",
        inputs=dict(img=torch.zeros((1, 3, *input_hw))),
        task_inputs={T.task_name: T.inputs[mode] for T in task_configs},
        task_modules={T.task_name: T.get_model(mode) for T in task_configs},
        lazy_forward=False,
    )
    if get_train_step() == "int_infer" and mode == "test":
        task_model["task_inputs"].update(tracking_feature=dict())
        task_model["task_modules"].update(tracking_feature=tracking_module)
    return task_model
