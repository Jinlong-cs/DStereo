import json
import os
import sys

from hat.core.proj_spec.descs import get_rcnn_classification_desc
from hat.core.proj_spec.detection import get_class_names_used_in_desc
from hat.models.backbones.mixvargenet import (
    MixVarGENetConfig,
    get_mixvargenet_stride2channels,
)
from hat.utils import Config

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from project_common import (  # noqa
    batch_size_factor,
    enable_model_tracking,
    input_size,
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
model_type = "traffic_light_multitask"
model_name = "_".join([model_type, model_setting])
compile_model = False
mask_in_bpu = True
deploy_batch = 1
if model_name_postfix:
    model_name = "_".join([model_type, model_name_postfix])

if tasks is not None:
    tasks = json.loads(tasks)
else:
    tasks = [
        dict(name="traffic_light_color_cls"),
        dict(name="traffic_light_dire_cls"),
        dict(name="traffic_light_type_cls"),
        dict(name="traffic_light_fp_cls"),
        dict(name="traffic_light_time_cls"),
        dict(name="traffic_light_lens_det"),
    ]

# dataset
ds_path = os.path.join(
    os.path.dirname(__file__), "../datasets/train_datasets.py"
)
if os.path.exists(ds_path):
    datapaths = Config.fromfile(ds_path).datapaths

is_local_train = not os.path.exists("/running_package")

if is_local_train:
    batch_size = 1 * batch_size_factor
    log_freq = 5
    save_prefix = "tmp_output"
else:
    batch_size = 12 * batch_size_factor
    log_freq = 100
    save_prefix = "/job_data/models/"

pred_batch_size = pred_batch_size

# model_config
extend_ratio = 0.1
bn_kwargs = dict(eps=1e-5, momentum=0.1)
cls_pooling_padding = 0  # (96, 96)
# cls_pooling_padding = 1  # (64, 64)

loss_weight = {
    "shell_color": 1,
    "shell_type": 1,
    "shell_fp": 1,
    "shell_time": 1,
    "lens_reg": 1,
    "lens_cls": 2,
}

# data_config
inter_method = 10
pixel_center_aligned = False
min_valid_clip_area_ratio = 0.02
random_roi_ratio = 0.0
norm_length = int(input_size[0] * 0.9)
norm_method = "max_width_height"
padding_context = None


def get_mixvargenet_config(
    net_config, output_list, input_channels=3, disable_quanti_input=False
):
    mixvargenet_backbone = dict(
        type="MixVarGENet",
        net_config=net_config,
        output_list=output_list,
        input_channels=input_channels,
        input_sequence_length=1,
        input_resize_scale=None,
        num_classes=1000,
        include_top=False,
        bn_kwargs=bn_kwargs,
        bias=True,
        disable_quanti_input=True if mask_in_bpu else False,
        node_name="mix_backbone",
    )

    return mixvargenet_backbone


tiny_mixvargenet_config = [
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
            out_channels=16,
            head_op="mixvarge_f4_gb16",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # stride 4
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
            extra_downsample_num=0,
        ),  # stride 8
    ],
    [
        MixVarGENetConfig(
            in_channels=32,
            out_channels=64,
            head_op="mixvarge_f4_gb16",
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
        ),  # stride 16
    ],
    [
        MixVarGENetConfig(
            in_channels=64,
            out_channels=128,
            head_op="mixvarge_f2_gb16",
            stack_ops=["mixvarge_f1_gb16", "mixvarge_f1_gb16"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # stride 32
    ],
]

# TinyMixVarGENet
tiny_mixvargenet_stride2channels = get_mixvargenet_stride2channels(
    net_config=tiny_mixvargenet_config, strides=[2, 4, 8, 16, 32]
)
backbone = get_mixvargenet_config(
    net_config=tiny_mixvargenet_config, output_list=[0, 1, 2, 3, 4]
)


input_preprocess = dict(
    type="TrafficLightPreprocess",
    mask_in_bpu=mask_in_bpu,
    transpose_hw=False,
    node_name="input_preprocess",
)

out_strides = (4,)

# stride2channels, backbone_channels = get_stride2channels()
backbone_channels = list(tiny_mixvargenet_stride2channels.values())
stride2channels = tiny_mixvargenet_stride2channels


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
        image_size=input_size,
        norm_method="max_width_height",
        norm_len=norm_length,
        extend_ratio=extend_ratio,
        location="CN",
    )
    del per_tensor_desc["output_name"]
    del per_tensor_desc["class_name"]
    classification_model_desc = dict(
        type="AddDesc",
        strict=True,
        per_tensor_desc=[
            json.dumps(per_tensor_desc) for _ in range(prediction_return_cnt)
        ],
        node_name=f"{output_name}_classification_desc",
    )
    return classification_model_desc


# detection specific
def get_anchor_model_desc(
    classnames,
    data_args,
    anchor_args,
    out_strides,
    task_name,
    prediction_return_cnt=3,
):
    anchor_wh_pair = []
    for anchor_wh in anchor_args["anchor_wh_groups"][0]:
        anchor_wh_pair.extend(anchor_wh)
    per_tensor_desc = (
        [
            json.dumps(
                dict(
                    task="traffic_lens_detection",
                    class_name=get_class_names_used_in_desc(classnames),
                    reg_type="frcnn",
                    anchor_wh_pair=anchor_wh_pair,
                    linear_a=int(out_strides[-1]),
                    linear_b=int(out_strides[-1] // 2),
                    legacy_bbox=int(
                        data_args["legacy_bbox"]
                    ),  # use 0/1 instead of False/True
                    pixel_center_align=int(pixel_center_aligned),
                    score_threshold=[0.5],
                    norm_len=norm_length,
                    extend_ratio=extend_ratio,
                    norm_method=norm_method,
                    image_size=input_size,
                )
            )
            for _ in range(prediction_return_cnt)
        ]
        + [
            json.dumps(
                dict(
                    task="traffic_lens_type_classification",
                    properties=[
                        {
                            "channel_labels": [
                                "L_Circle",
                                "L_Forward",
                                "L_Left",
                                "L_Right",
                                "L_Return",
                                "L_Pedestrain",
                                "L_Non_Motor",
                                "L_Time",
                                "L_left_and_return",
                                "L_Forward_and_Left",
                                "L_Forward_and_Right",
                                "L_No_Drive_into",
                                "L_Allow_Drive_into",
                            ]
                        }
                    ],
                    location="CN",
                    norm_len=norm_length,
                    extend_ratio=extend_ratio,
                    norm_method=norm_method,
                    image_size=input_size,
                )
            ),
            json.dumps(
                dict(
                    task="traffic_lens_color_classification",
                    properties=[
                        {"channel_labels": ["green", "yellow", "red"]}
                    ],
                    location="CN",
                    norm_len=norm_length,
                    extend_ratio=extend_ratio,
                    norm_method=norm_method,
                    image_size=input_size,
                )
            ),
        ],
    )
    anchor_model_desc = dict(
        type="AddDesc",
        strict=True,
        per_tensor_desc=per_tensor_desc[0],
        node_name=f"{task_name}_anchor_desc",
    )

    return anchor_model_desc


def get_model_track_desc(
    task_name,
):
    strides = list(stride2channels.keys())
    per_tensor_desc = [
        json.dumps(
            dict(
                task="tracking_feature",
                size=[
                    1,
                    input_size[1] // strides[-1],
                    input_size[0] // strides[-1],
                    backbone_channels[-1],
                ],
            )
        )
    ]
    model_track_desc = dict(
        type="AddDesc",
        strict=True,
        per_tensor_desc=per_tensor_desc,
        node_name=f"{task_name}_tracking_desc",
    )
    return model_track_desc


def get_unet_neck(
    in_strides=(2, 4, 8, 16, 32),
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
    dequant_output,
    *,
    is_dim_match=False,
    factor=2,
    group_base=4,
    mode="train",
):
    anchor_head = dict(
        type="TinyVarGNetV2LenAttrHead",
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
        attr_list=anchor_args["attr_list"],
        dequant_output=dequant_output,
        mode=mode,
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
    pre_nms_top_k=500,
    post_nms_top_k=100,
    nms_margin=0.0,
    nms_padding_mode="pad_zero",
    bbox_min_hw=(1, 1),
    mode="train",
):
    if mode == "train" or mode == "val":
        task_name = "traffic_lens"
        anchor_pred = dict(
            type="TLAttrDetPostProcess",
            num_classes=anchor_args["num_fg_classes"],
            class_offsets=[0] * len(anchor_args["feat_strides"]),
            use_clippings=use_clippings,
            image_hw=input_size,
            nms_iou_threshold=nms_iou_threshold,
            pre_nms_top_k=pre_nms_top_k,
            post_nms_top_k=post_nms_top_k,
            nms_margin=nms_margin,
            input_key="rpn_head_out",
            box_filter_threshold=box_filter_threshold,
            nms_padding_mode=nms_padding_mode,
            bbox_min_hw=bbox_min_hw,
            attr_list=anchor_args["attr_list"],
            node_name=f"{task_name}_anchor_pred",
        )
    elif mode == "test":
        anchor_pred = dict(
            type="RPNTrafficLightFilter",
            threshold=box_filter_threshold,
            strides=[1],  # not used
            idx_range=None,
            for_compile=True,
            node_name=f"{task_name}_anchor_pred",
        )
    return anchor_pred


val_decoders = {}

val_transforms = [
    dict(
        type="RoiTransformer",
        roi_crop_parm=dict(
            norm_len=norm_length,
            norm_method=norm_method,
            output_wh=input_size,
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
            target_wh=input_size,
            inter_method=2,
            use_pyramid=True,
            pyramid_min_step=0.8,
            pyramid_max_step=0.8,
            pixel_center_aligned=pixel_center_aligned,
        ),
        bbox_ts_parm=dict(
            clip=False,
            min_valid_area=10,
            min_valid_clip_area_ratio=0,
            min_edge_size=1,
            label_type="classification",
        ),
        convert_roi=True,
    ),
    dict(
        type="MaskOutsideRegion",
        input_wh=input_size,
        extend_ratio=extend_ratio,
        mask_in_bpu=mask_in_bpu,
    ),
    dict(
        type="ToTensor",
    ),
]

vis_tasks = []
