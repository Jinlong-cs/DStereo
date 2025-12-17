import json
import os
import re
from pathlib import Path

import torch
from hatbc.filestream.bucket.client import get_bucket_mount_root
from horizon_plugin_pytorch.quantization import March

from hat.core.proj_spec.detection import (
    get_class_names_used_in_desc,
    get_det_default_merge_fn_type_and_params,
)

# ---------------------------------------------------------
job_name = "TrafficSignAssistDetection"
compile_task_name = "traffic_sign_assist"
desc_task_type = "traffic_sign_assist_detection"

aidi_model_name = "hat_traffic_sign_assist"

exp_root = os.path.dirname(__file__)
model_id = os.path.basename(exp_root)

model_version = model_id.split("_")[0]
assert re.match(r"^v\d+\.\d+\.\d+", str(model_version)), (
    "illegal version name %s, example: v1.0.0" % model_version
)

basic_versions = None
readme_str = None
readme_file = os.path.join(exp_root, "README.md")
if os.path.exists(readme_file):
    with open(readme_file, "r", encoding="utf-8") as fin:
        readme_str = fin.read()
        found = re.findall(r"基于(.*)\n", readme_str)
        if len(found) > 0:
            assert len(found) == 1, len(found)
            basic_versions = found[0].split(",")

            if basic_versions[0] in ["None", "none", ""]:
                basic_versions = None

assert (
    len(readme_str) < 999
), "commit message shall be within 999 characters but get {}".format(
    len(readme_str)
)

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
# march = March.BAYES
march = March.BERNOULLI2  # j3
hpflow_infer = bool(os.environ.get("HPFLOW_INFER", False) == "True")

use_calibration_step = march


config_file = "projects/" + os.path.abspath(__file__).split("projects/")[-1]
config_file_root = os.path.dirname(config_file)
hat_root = os.path.abspath(__file__).split("projects/")[0]
abs_config_file_root = os.path.abspath(config_file_root)

save_prefix = "tmp_output" if is_local_train else "/job_data/models/"
ckpt_dir = Path(save_prefix) / f"{job_name}_{model_version}"
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
float_steps = 1000 if pipeline_test else 50000
qat_steps = 1000 if pipeline_test else 3000
warmup_steps = 100 if pipeline_test else 500

float_lr = 0.01
qat_lr = 0.0002

save_interval = 50 if pipeline_test else 1000
interval_by = "step"
# -------------------------------------------------------
# input
# -------------------------------------------------------
input_hw = (128, 128)  # img_crop_hw, also model_input_hw
deploy_input_hw = (128, 128)
img_channel = 3
resize_hw = None  # None means no resize
inter_method = 10
pixel_center_aligned = False
min_valid_clip_area_ratio = 0.9
rand_translation_ratio = 0.0
padding_context = None  # useful when norm_method = resize
pretrain_checkpoint = "http://fm-hao-chen.ucloudtrain.hogpu.cc/plat_gpu/TinyVarGNetV2_CLS-HAT-pretrain-20220223-113926/output/models/TinyVarGNetV2_CLS/float-checkpoint-best-9b796482.pth.tar"  # noqa

# data loader
# --------------------------------------------------
roi_point_id = 4
norm_point_id1 = 1
norm_point_id2 = 2
lt_id = 10
rb_id = 12

norm_len = 110
norm_method = "height"  # change as norm_point_id1/2 change
padding_context = None  # useful when norm_method = resize

norm_len_ratio_list = [0.7, 0.85, 1.0]
norm_len_list = [int(norm_len * ratio) for ratio in norm_len_ratio_list]
sample_weight_list = [2.0, 2.0, 2.0]


bbox_reg_type = "frcnn"
# legacy_bbox = True
legacy_bbox = False
# -------------------------------------------------------
# multitask shared model
# ------------------------------------------------------

bn_kwargs = dict(eps=1e-5, momentum=0.1)
backbone = dict(
    type="TinyVargNetV2",
    num_classes=1000,
    bn_kwargs=bn_kwargs,
    alpha=0.5,
    group_base=8,
    include_top=False,
    extend_features=True,
    input_resize_scale=None,
    channel_list=[32, 32, 64, 128, 256, 512],
    node_name="backbone",
)


def get_stride2channels():
    pc = [int(c * backbone["alpha"]) for c in backbone["channel_list"]]
    # s2c = {2 ** (idx + 1): c for idx, c in enumerate(pc)}
    s2c = {2 ** i: c for i, c in enumerate(pc, 1)}
    return s2c, pc


stride2channels, backbone_channels = get_stride2channels()


# detection specific
def get_unet_neck(
    in_strides=(2, 4, 8, 16, 32, 64),
    out_strides=(4,),
):
    unet_neck = dict(
        type="Unet",
        in_strides=in_strides,
        out_strides=out_strides,
        stride2channels=stride2channels,
        fusion_block_name="onepath",
        node_name="unet_neck",
    )
    return unet_neck


def get_anchor_generator(anchor_args, data_args, task_name):
    anchor_generator = dict(
        type="AnchorGenerator",
        feat_strides=anchor_args["feat_strides"],
        anchor_wh_groups=anchor_args["anchor_wh_groups"],
        legacy_bbox=data_args["legacy_bbox"],
        node_name=f"{task_name}_anchor",
    )
    return anchor_generator


def get_anchor_head(
    in_channels,
    num_channels,
    anchor_args,
    task_name,
    *,
    is_dim_match=False,
    factor=2,
    group_base=4,
):
    anchor_head = dict(
        type="RPNVarGNetHead",
        in_channels=in_channels,
        num_channels=num_channels,
        num_classes=anchor_args["num_fg_classes"],
        num_anchors=[len(_) for _ in anchor_args["anchor_wh_groups"]],
        feat_strides=anchor_args["feat_strides"],
        is_dim_match=is_dim_match,
        bn_kwargs=bn_kwargs,
        factor=factor,
        group_base=group_base,
        output_shift=4,
        node_name=f"{task_name}_anchor_head",
    )
    return anchor_head


def get_anchor_post_process_cfg(
    anchor_args,
    task_name,
    *,
    use_clippings=True,
    nms_iou_threshold=0.5,
    box_filter_threshold=0.05,
    pre_nms_top_k=2000,
    post_nms_top_k=100,
    nms_margin=0.0,
    nms_padding_mode="pad_zero",
    bbox_min_hw=(1, 1),
    march=March.BERNOULLI2,
):
    if march == March.BERNOULLI2:
        anchor_pred = dict(
            type="AnchorPostProcess",
            num_classes=anchor_args["num_fg_classes"],
            class_offsets=[0] * len(anchor_args["feat_strides"]),
            use_clippings=use_clippings,
            image_hw=input_hw,
            nms_iou_threshold=nms_iou_threshold,
            pre_nms_top_k=pre_nms_top_k,
            post_nms_top_k=post_nms_top_k,
            nms_margin=nms_margin,
            input_key="rpn_head_out",
            box_filter_threshold=box_filter_threshold,
            nms_padding_mode=nms_padding_mode,
            bbox_min_hw=bbox_min_hw,
            node_name=f"{task_name}_anchor_pred",
        )
    elif march == March.BAYES:
        anchor_pred = dict(
            type="RPNTrafficSignFilter",
            threshold=float(box_filter_threshold),
            strides=[1],  # not used
            idx_range=None,
            anchor_args=anchor_args,
            node_name=f"{task_name}_anchor_pred",
        )
    else:
        raise ValueError(f"Unknown march {march} !")
    return anchor_pred


def get_anchor_model_desc(
    classnames,
    data_args,
    task_name,
    mode,
    anchor_wh_groups,
    roi_region=None,
    prediction_return_cnt=4,
    vanishing_point=None,
):
    merge_fn_type, merge_fn_params = get_det_default_merge_fn_type_and_params()
    if march == March.BERNOULLI2:
        anchor_model_desc = dict(
            type="AddDesc",
            strict=True,
            per_tensor_desc=[
                json.dumps(
                    dict(
                        task="frcnn_detection"
                        if hpflow_infer
                        else desc_task_type,
                        anchor_wh_pair=sum(anchor_wh_groups[0], []),
                        linear_a=4,
                        linear_b=2.0,
                        class_name=get_class_names_used_in_desc(classnames),
                        reg_type="rcnn" if hpflow_infer else "frcnn",
                        legacy_bbox=int(
                            data_args["legacy_bbox"]
                        ),  # use 0/1 instead of False/True
                        pixel_center_align=0,
                        score_threshold=[0.25, 0.25],
                        image_size=input_hw,
                        norm_method=norm_method,
                        norm_len=norm_len,
                    )
                )
                for _ in range(prediction_return_cnt)
            ],
            node_name=f"{task_name}_anchor_desc",
        )
    else:
        anchor_model_desc = dict(
            type="AddDesc",
            strict=True,
            per_tensor_desc=[
                json.dumps(
                    dict(
                        task="frcnn_detection"
                        if hpflow_infer
                        else desc_task_type,
                        anchor_wh_pair=sum(anchor_wh_groups[0], []),
                        linear_a=4,
                        linear_b=2,
                        class_name=get_class_names_used_in_desc(classnames),
                        reg_type="rcnn" if hpflow_infer else "frcnn",
                        legacy_bbox=int(
                            data_args["legacy_bbox"]
                        ),  # use 0/1 instead of False/True
                        pixel_center_align=0,
                        score_threshold=[0.25, 0.25],
                        nms_threshold=0.5,
                        norm_len=norm_len,
                        norm_method=norm_method,
                        image_size=input_hw,
                        padding=[0.0, 0.0, 0.0, 0.0],
                    )
                )
                for _ in range(prediction_return_cnt)
            ],
            node_name=f"{task_name}_anchor_desc",
        )

    return anchor_model_desc


# -------------------------------------------------------
# common_func
# ------------------------------------------------------


def get_train_step():
    return os.environ.get("HAT_TRAINING_STEP", "float")


def get_task_model(mode, task_configs):
    return dict(
        type="MultitaskGraphModel",
        inputs=dict(img=torch.zeros((1, 3, *input_hw))),
        task_inputs={T.task_name: T.inputs[mode] for T in task_configs},
        task_modules={T.task_name: T.get_model(mode) for T in task_configs},
        lazy_forward=False,
    )
