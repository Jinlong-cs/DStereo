import json
import os
import sys

from hat.core.proj_spec.descs import get_rcnn_classification_desc
from hat.utils import Config

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from project_common import (  # noqa
    batch_size_factor,
    enable_model_tracking,
    input_size,
    march,
    model_checkpoint,
    model_name_postfix,
    model_setting,
    model_version,
    num_machines,
    num_worker,
    pipeline_test,
    pred_batch_size,
    resume_training,
    tasks,
    training_step,
)

sys.modules.pop("project_common")

# model info
model_type = "work_condition_multitask"
model_name = "_".join([model_type, model_setting])
compile_model = False
if model_name_postfix:
    model_name = "_".join([model_type, model_name_postfix])

if tasks is not None:
    tasks = json.loads(tasks)
else:
    tasks = [
        dict(name="scene_classification"),
        dict(name="weather_classification"),
        dict(name="illumination_classification"),
        dict(name="time_classification"),
    ]

# dataset
ds_path = os.path.join(
    os.path.dirname(__file__), "../datasets/train_datasets.py"
)
if os.path.exists(ds_path):
    datapaths = Config.fromfile(ds_path).datapaths

is_local_train = not os.path.exists("/running_package")

if is_local_train:
    batch_size = 4 * batch_size_factor
    log_freq = 5
    save_prefix = "tmp_output"
else:
    batch_size = 12 * batch_size_factor
    log_freq = 100
    save_prefix = "/job_data/models/"

pred_batch_size = pred_batch_size

# model_config
bn_kwargs = dict(eps=1e-5, momentum=0.1)
cls_pooling_padding = 0

# data_config
inter_method = 10
pixel_center_aligned = False
min_valid_clip_area_ratio = 0.5
random_roi_ratio = 0.0
norm_length = 456
norm_method = "width"
padding_context = None

backbone = dict(
    type="TinyVargNetV2",
    num_classes=1000,
    bn_kwargs=bn_kwargs,
    alpha=0.75,
    group_base=8,
    include_top=False,
    extend_features=True,
    input_resize_scale=None,
    channel_list=[32, 32, 64, 128, 256, 256],
    node_name="backbone",
)


# common structures
def get_stride2channels():
    pc = [
        int(c * backbone.get("alpha", 1)) for c in [32, 32, 64, 128, 256, 256]
    ]
    s2c = {2 ** (idx + 1): c for idx, c in enumerate(pc)}
    return s2c, pc


stride2channels, backbone_channels = get_stride2channels()


# classification specific
def get_classification_model_desc(
    task_name,
    output_name,
    class_name,
    desc_id,
    prediction_return_cnt=1,
):
    per_tensor_desc = get_rcnn_classification_desc(
        task_name, output_name, class_name, desc_id
    )
    per_tensor_desc = json.loads(per_tensor_desc)
    per_tensor_desc.update(
        crop_desc=dict(
            image_size=input_size,
            norm_method="resize",
            padding=[0.0, 0.0, 0.0, 0.0],
            input_padding=[0, 0, 0, 0],
            roi_input={"left": 16, "top": 0, "width": 456, "height": 234},
        ),
    )
    classification_model_desc = dict(
        type="AddDesc",
        strict=True,
        per_tensor_desc=[
            json.dumps(per_tensor_desc) for _ in range(prediction_return_cnt)
        ],
        node_name=f"{output_name}_classification_desc",
    )
    return classification_model_desc


val_decoders = {}

val_transforms = [
    dict(
        type="RoiTransformer",
        roi_crop_parm=dict(
            norm_len=norm_length,
            norm_method=norm_method,
            output_wh=input_size[::-1],
            input_wh=None,
            min_crop_scale=1.0,
            max_crop_scale=1.0,
            max_coord_jitter_ratio=0.0,
            img_min_scale=0.01,
            img_max_scale=100,
            padd_val=0.0,
            random_roi_ratio=0.0,
            restrict_roi_in_center=False,
            flip_ratio=0,
        ),
        img_crop_parm=dict(
            target_wh=input_size[::-1],
            inter_method=2,
            use_pyramid=True,
            pyramid_min_step=0.8,
            pyramid_max_step=0.8,
            pixel_center_aligned=True,
        ),
        bbox_ts_parm=dict(
            clip=False,
            min_valid_area=100,
            min_valid_clip_area_ratio=0,
            min_edge_size=10,
            label_type="classification",
        ),
    ),
    dict(
        type="ToTensor",
    ),
]

vis_tasks = []
