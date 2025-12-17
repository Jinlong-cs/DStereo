import json
import os
from pathlib import Path

import torch
from hatbc.filestream.bucket.client import get_bucket_mount_root
from horizon_plugin_pytorch.quantization import March

from hat.core.proj_spec.detection import (
    get_class_names_used_in_desc,
    get_det_default_merge_fn_type_and_params,
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
multitask_task_name = "mono_multitask_2pe_person"
# march = March.BERNOULLI2  # j3
march = March.BAYES  # j5

# input_shift = 6 # j3
input_shift = 12  # j5

use_calibration_step = march

save_prefix = "tmp_output" if is_local_train else "/job_data/models/"
ckpt_dir = Path(save_prefix) / multitask_task_name
log_dir = ckpt_dir / "logs"

# ------------------------------------------------------
# train
# ------------------------------------------------------
num_machines = 1
num_gpus_per_machine = 2
device_ids = [0] if is_local_train else list(range(num_gpus_per_machine))
log_freq = 5 if is_local_train else 25
float_steps = 23000 * 4
qat_steps = 10000 * 4
warmup_steps = 500

float_lr = 0.01
qat_lr = 0.0005

save_interval = 50 if pipeline_test else 5000
interval_by = "step"
# -------------------------------------------------------
# input
# -------------------------------------------------------
input_hw = (192, 192)  # img_crop_hw, also model_input_hw
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
lt_id = 0
rb_id = 2

norm_len = 112
norm_method = "height"  # change as norm_point_id1/2 change
padding_context = None  # useful when norm_method = resize

norm_len_ratio_list = [0.7, 0.85, 1.0]
norm_len_list = [int(norm_len * ratio) for ratio in norm_len_ratio_list]
sample_weight_list = [2.0, 2.0, 2.0]


bbox_reg_type = "frcnn"
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
    extend_features=False,
    input_resize_scale=None,
    channel_list=[32, 32, 64, 128, 256],
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
    in_strides=(2, 4, 8, 16, 32),
    out_strides=(4,),
):
    unet_neck = dict(
        type="Unet",
        in_strides=in_strides,
        out_strides=out_strides,
        stride2channels=stride2channels,
        fusion_block_name="default",
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
    if march == March.BERNOULLI2:  # j3
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
            output_shift=input_shift,
            node_name=f"{task_name}_anchor_head",
        )
    else:  # j5
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
            output_shift=input_shift,
            node_name=f"{task_name}_anchor_head",
            qat_dtype="int16",
        )
    return anchor_head


def get_anchor_post_process_cfg(
    mode,
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
):
    task_name = "person_head"
    if mode == "deploy" and march == March.BAYES:
        anchor_pred = dict(
            type="PersonDetFilterV2",
            input_key="rpn_head_out",
            num_anchor=[len(_) for _ in anchor_args["anchor_wh_groups"]][0],
            threshold=0.05,
            node_name=f"{task_name}_anchor_pred",
        )
    elif mode == "deploy":  # j3 only
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
    else:
        anchor_pred = dict(
            type="PersonDetAnchorPostProcess",
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
            input_shift=input_shift,
        )

    return anchor_pred


def get_anchor_model_desc(
    classnames,
    data_args,
    anchor_args,
    task_name,
    mode,
    roi_region=None,
    prediction_return_cnt=4,
    vanishing_point=None,
    det_task_name="person_2pe_detection",
):
    prediction_return_cnt = (
        3 if mode == "deploy" and march == March.BAYES else 4
    )
    merge_fn_type, merge_fn_params = get_det_default_merge_fn_type_and_params()
    if mode == "deploy" and march == March.BERNOULLI2:
        anchor_model_desc = dict(
            type="AddDesc",
            strict=True,
            per_tensor_desc=[
                json.dumps(
                    dict(
                        task=det_task_name,
                        class_name=get_class_names_used_in_desc(classnames),
                        reg_type="frcnn",
                        legacy_bbox=int(
                            data_args["legacy_bbox"]
                        ),  # use 0/1 instead of False/True
                        anchor_wh_pair=[40, 92, 92, 92],
                        score_threshold=[0.5, 0.5, 0.5, 0.5],
                        norm_len=norm_len,
                        norm_method=norm_method,
                        image_size=deploy_input_hw,
                        pixel_center_align=0,
                        linear_a=4,
                        linear_b=2.0,
                    )
                )
                for _ in range(prediction_return_cnt)
            ],
            node_name=f"{task_name}_anchor_desc",
        )
    elif mode == "deploy" and march == March.BAYES:
        anchor_model_desc = dict(
            type="AddDesc",
            strict=True,
            per_tensor_desc=[
                json.dumps(
                    dict(
                        task=det_task_name,
                        class_name=get_class_names_used_in_desc(classnames),
                        score_act_type="identity",
                        mean=(0, 0, 0, 0),
                        std=(1, 1, 1, 1),
                        reg_type="rcnn",
                        legacy_bbox=int(
                            data_args["legacy_bbox"]
                        ),  # use 0/1 instead of False/True
                        anchor_wh_pair=[40, 92, 92, 92],
                        nms_threshold=0.5,
                        score_threshold=[0.5, 0.5, 0.5, 0.5],
                        norm_len=norm_len,
                        norm_method=norm_method,
                        image_size=deploy_input_hw,
                        pixel_center_align=0,
                        linear_a=4,
                        linear_b=2,
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
                        task="frcnn_detection",
                        class_name=get_class_names_used_in_desc(classnames),
                        class_agnostic=True,
                        score_act_type="identity",
                        with_background=False,
                        mean=(0, 0, 0, 0),
                        std=(1, 1, 1, 1),
                        reg_type="rcnn",
                        legacy_bbox=int(
                            data_args["legacy_bbox"]
                        ),  # use 0/1 instead of False/True
                        nms_threshold=0.7,
                        roi_regions=roi_region,
                        vanishing_point=vanishing_point,
                        merge_fn_type=merge_fn_type,
                        merge_fn_params=merge_fn_params,
                        crop_desc=dict(
                            norm_len=norm_len,
                            norm_method=norm_method,
                            image_size=input_hw,
                            padding=None,
                        ),
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
        task_modules={
            T.task_name: T.get_model(mode, march) for T in task_configs
        },
        lazy_forward=False,
    )
