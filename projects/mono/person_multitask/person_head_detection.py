import copy
import os
from collections import OrderedDict

import torch
from horizon_plugin_pytorch.march import March

from hat.data.collates.collates import collate_2d
from hat.utils import Config

# -------------------------- common --------------------------
cfg_dir = os.path.dirname(__file__)
BASE_CONFIG = Config.fromfile(os.path.join(cfg_dir, "auto2d_base.py"))

unet_out_strides = BASE_CONFIG.unet_out_strides
bn_kwargs = BASE_CONFIG.bn_kwargs
stride2channels = BASE_CONFIG.stride2channels
norm_len = BASE_CONFIG.norm_len
norm_method = BASE_CONFIG.norm_method
input_size = BASE_CONFIG.input_size
task_loss_weights = BASE_CONFIG.task_loss_weights
log_freq = BASE_CONFIG.log_freq
test_image_dir = BASE_CONFIG.test_image_dir
training_step = BASE_CONFIG.training_step
train_num_workers = BASE_CONFIG.train_num_workers
val_num_workers = BASE_CONFIG.val_num_workers
test_num_workers = BASE_CONFIG.test_num_workers
input_sequence_length = BASE_CONFIG.input_sequence_length
test_batch_size_per_gpu = BASE_CONFIG.test_batch_size_per_gpu
compile_model = BASE_CONFIG.compile_model
get_anchor_model_desc = BASE_CONFIG.get_anchor_model_desc
get_anchor_post_process_cfg = BASE_CONFIG.get_anchor_post_process_cfg

dataset_val_img_path = BASE_CONFIG.dataset_val_img_path
train_batch_size_per_gpu = BASE_CONFIG.train_batch_size_per_gpu
val_batch_size_per_gpu = BASE_CONFIG.val_det_batch_size_per_gpu
dataset_train_rect_path = BASE_CONFIG.dataset_train_rect_path
annotations_train_path = BASE_CONFIG.annotations_train_path


# -------------------------- task --------------------------
dataset_index = 0
class_name = "person_head"
task_type = "detection"
task_name = f"{class_name}_{task_type}"
stacked_convs = 2
head_out_strides = [4]
divisor = 64
num_classes = 1
norm_target_bbox = False
use_iou_replace_ctrness = True
enable_auto_assign = False
INF = 1e8
regress_ranges = (
    (-1, INF),
    # (-1, 32), #p3
    # (32, 64),
    # (64, 128),
    # (128, INF),
    # (256, INF)
    # (256, 512),
    # (512, INF),
)
task_loss_weight = task_loss_weights[task_name]
head_det_final_layer_out_shift = 6
anchor_wh = (16.0, 16.0)
anchor_legacy = False


# -------------------------- data --------------------------
data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="DenseboxDataset",
        data_path=dataset_train_rect_path[dataset_index],
        anno_path=annotations_train_path[dataset_index],
        task_type=task_type,
        class_id=[9],
        category={9: 0},
        to_rgb=False,
        ignore_hard=True,
        maximum_instances_per_image=30,
        # note: Normalize needs to be after ToTensor if `to_yuv`==True
        transforms=[
            dict(
                type="RoiTransformer",
                roi_crop_parm=dict(
                    norm_len=norm_len,
                    norm_method=norm_method,
                    output_wh=input_size,
                    input_wh=None,
                    min_crop_scale=0.9,
                    max_crop_scale=1.1,
                    max_coord_jitter_ratio=0.05,
                    img_min_scale=0.01,
                    img_max_scale=100,
                    padd_val=0,
                    random_roi_ratio=0.0,
                    restrict_roi_in_center=False,
                    flip_ratio=0,
                ),
                img_crop_parm=dict(
                    target_wh=input_size,
                    inter_method=10,
                    use_pyramid=True,
                    pyramid_min_step=0.7,
                    pyramid_max_step=0.8,
                    pixel_center_aligned=False,
                ),
                bbox_ts_parm=dict(
                    clip=False,
                    min_valid_area=100,
                    min_valid_clip_area_ratio=0.02,
                    min_edge_size=0,
                    label_type=task_type,
                ),
            ),
            # dict(type="RandomFlip", px=0.5),
            dict(type="ToTensor", to_yuv=True),
            dict(type="Normalize", mean=128.0, std=128.0),
            dict(
                type="Batchify",
                size=input_size[::-1],
                divisor=divisor,
                repeat=input_sequence_length,
            ),
            dict(type="RenameKeys", keys=["imgs|img"]),
        ],
    ),
    # sampler=dict(type=torch.utils.data.DistributedSampler),
    collate_fn=collate_2d,
    batch_size=train_batch_size_per_gpu,
    # shuffle=True,
    num_workers=train_num_workers,
    pin_memory=True,
    persistent_workers=train_num_workers > 0,
)
data_loader = BASE_CONFIG.change_dataset(data_loader, task_name, "mono")

val_transforms = [
    dict(type="TopDownAffine", image_size=(64, 128)),
    dict(type="ToTensor", to_yuv=True),
    dict(type="Normalize", mean=128.0, std=128.0),
    dict(
        type="Batchify",
        size=input_size[::-1],
        divisor=divisor,
        repeat=input_sequence_length,
    ),
    dict(type="RenameKeys", keys=["imgs|img"]),
]


val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Auto2dFromImage",
        data_path=dataset_val_img_path[dataset_index],
        to_rgb=True,
        transforms=val_transforms,
        infer_model_type="full_image_2pe",
    ),
    batch_size=val_batch_size_per_gpu,
    collate_fn=collate_2d,
    shuffle=False,
    num_workers=val_num_workers,
    pin_memory=True,
    drop_last=False,
)

# set keep_ratio=False in test_data_loader
test_transforms = copy.deepcopy(val_transforms)


test_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Auto2dFromImage",
        data_path=test_image_dir,
        to_rgb=True,
        transforms=test_transforms,
    ),
    batch_size=test_batch_size_per_gpu,
    collate_fn=collate_2d,
    shuffle=False,
    num_workers=test_num_workers,
    pin_memory=True,
    drop_last=False,
)

# -------------------------- model --------------------------
inputs = dict(
    # # TODO(min.du): use imgs instead of img #
    img_id=None,
    img_name=None,
    img_height=None,
    img_width=None,
    gt_bboxes=None,
    gt_classes=None,
    color_space=None,
    layout=None,
    img_shape=None,
    pad_shape=None,
    ig_bboxes=None,
    crop_roi=None,
)

val_inputs = copy.deepcopy(inputs)
val_inputs.pop("gt_bboxes")
val_inputs.pop("gt_classes")
test_inputs = {}


def topo_builder(nodes, _inputs, feats, mode):
    # filter by inputs keys
    if mode == "train":
        inner_inputs = {k: _inputs[k] for k in inputs}
    elif mode == "val":
        inner_inputs = {k: _inputs[k] for k in val_inputs}
    elif mode == "test":
        inner_inputs = {k: _inputs[k] for k in test_inputs}
    else:
        raise Exception("error mode")
    name2out = OrderedDict()
    out_module = nodes[f"{task_name}_head"]
    if f"{task_name}_neck" in nodes.keys():
        neck_module = nodes[f"{task_name}_neck"]
        feats = neck_module(feats)

    name2out.update({task_name: out_module(feats, inner_inputs)})
    return name2out


# model neck
neck_module = dict(
    type="Unet",
    in_strides=(2, 4, 8, 16, 32, 64),
    out_strides=unet_out_strides,
    stride2channels=stride2channels,
    group_base=8,
    # fusion_block_name="onepath",
    bn_kwargs=bn_kwargs,
)

# model head
head_type = "rpn"
if head_type == "fcos":
    out_module = dict(
        type="OutputModule",
        head=dict(
            type="FCOSHead",
            num_classes=num_classes,
            in_strides=unet_out_strides,
            out_strides=head_out_strides,
            stride2channels=stride2channels,
            feat_channels=16,
            stacked_convs=stacked_convs,
            use_sigmoid=True,
            share_bn=False,
            share_conv=False,
            upscale_bbox_pred=not norm_target_bbox,
        ),
        head_parser=None,
        target=dict(
            type="FCOSTarget",
            strides=head_out_strides,
            regress_ranges=regress_ranges,
            cls_out_channels=num_classes,
            background_label=num_classes,
            use_iou_replace_ctrness=use_iou_replace_ctrness,
            norm_on_bbox=norm_target_bbox,
            bbox_coder="DeltaXYWH",
        ),
        loss=dict(
            type="FCOSLoss",
            cls_loss=dict(
                type="FocalLoss",
                loss_name="loss_cls",
                num_classes=num_classes + 1,
                alpha=0.25,
                gamma=0.5,
                loss_weight=task_loss_weight,
                hard_neg_mining_cfg=dict(
                    keep_pos=True,
                    neg_ratio=0.75,
                    hard_ratio=0.5,
                    min_keep_num=64,
                    ignore_largest_n=1,
                ),
            ),
            reg_loss=dict(
                type="CIoULoss",
                loss_name="loss_bbox",
                loss_weight=task_loss_weight,
            ),
            # centerness_loss=dict(
            #     type="CrossEntropyLoss",
            #     use_sigmoid=True,
            #     loss_name="loss_centerness"
            #     if not use_iou_replace_ctrness
            #     else "loss_iou",
            #     loss_weight=task_loss_weight,
            # ),
        ),
        prefix="head",
    )
elif head_type == "rpn":
    out_module = dict(
        type="OutputModule",
        head=dict(
            type="RPNVarGNetHead",
            in_channels=[16],
            num_channels=[16],
            num_classes=num_classes,
            num_anchors=[1],
            feat_strides=[4],
            is_dim_match=True,
            bn_kwargs=bn_kwargs,
            factor=2,
            group_base=4,
            output_shift=head_det_final_layer_out_shift,
        ),
        head_parser=None,
        target=dict(
            type="FCOSTarget4RPNHead",
            strides=head_out_strides,
            regress_ranges=regress_ranges,
            cls_out_channels=num_classes,
            background_label=num_classes,
            use_iou_replace_ctrness=use_iou_replace_ctrness,
            norm_on_bbox=norm_target_bbox,
            soft_label=True,
            reference_anchor_width=int(anchor_wh[0]) - int(anchor_legacy),
            reference_anchor_height=int(anchor_wh[1]) - int(anchor_legacy),
        ),
        loss=dict(
            type="FCOSLoss",
            cls_loss=dict(
                type="ElementwiseL2HingeLoss",
                reduction="mean",
                pos_label=0.1,
                hard_neg_mining_cfg=dict(
                    keep_pos=True,
                    neg_ratio=0.75,
                    hard_ratio=0.5,
                    min_keep_num=64,
                    # ignore_largest_n=1,
                ),
                loss_weight=task_loss_weight,
            ),
            reg_loss=dict(
                type="CIoULoss",
                loss_name="loss_bbox",
                loss_weight=task_loss_weight,
            ),
        ),
        prefix="head",
    )


val_out_module = copy.deepcopy(out_module)
if head_type == "fcos":
    val_out_module["head"].update(
        dict(
            upscale_bbox_pred=False,
            dequant_output=False,
            enable_centerness=False,
            use_scale=False,
        )
    )
val_out_module["target"] = None
val_out_module["loss"] = None
test_out_module = copy.deepcopy(val_out_module)

anchor_args = dict(
    feat_strides=head_out_strides,
    anchor_wh_groups=[[anchor_wh]],
    num_fg_classes=num_classes,
    exclude_background=True,
)
anchor_generator = dict(
    type="AnchorGenerator",
    feat_strides=head_out_strides,
    anchor_wh_groups=anchor_args["anchor_wh_groups"],
    legacy_bbox=anchor_legacy,
)
if not compile_model:
    anchor_pred = get_anchor_post_process_cfg(
        anchor_args,
        nms_iou_threshold=0.4,
        box_filter_threshold=0,
        task_name=task_name,
        input_shift=head_det_final_layer_out_shift,
    )
else:
    if BASE_CONFIG.march == March.BAYES:
        anchor_pred = dict(
            type="RPNDetFilter",
            threshold=0.25,
            num_anchor=len(anchor_args["anchor_wh_groups"]),
        )
    else:
        anchor_pred = get_anchor_post_process_cfg(
            anchor_args,
            nms_iou_threshold=0.4,
            box_filter_threshold=0.25,
            task_name=task_name,
            input_shift=head_det_final_layer_out_shift,
        )
# train_pred = copy.deepcopy(anchor_pred)
# train_pred.update(
#     pre_nms_top_k=10000,
#     post_nms_top_k=1000,
#     nms_padding_mode="rollover",
# )
add_desc_pp = get_anchor_model_desc(
    class_name,
    anchor_legacy,
    task_name=task_name,
    image_hw=input_size[::-1],
    prediction_return_cnt=4 if BASE_CONFIG.march != March.BAYES else 3,
    anchor_wh=anchor_wh,
)
dpp_postprocess = dict(
    type="MultiInputSequential",
    modules=[anchor_pred, add_desc_pp],  # should be the last module
)

nodes = {f"{task_name}_head": out_module, f"{task_name}_neck": neck_module}
val_nodes = {
    f"{task_name}_head": val_out_module,
    f"{task_name}_neck": neck_module,
    "anchor_generator": anchor_generator,
    "dpp_postprocess": dpp_postprocess,
}
test_nodes = {
    f"{task_name}_head": test_out_module,
    f"{task_name}_neck": neck_module,
}


# -------------------------- solver --------------------------
def get_update_metric():
    def update_metric(metrics, batch, model_outs):
        for metric, out in zip(metrics, model_outs[0]):
            metric.update(out)

    return update_metric


metric_updater = dict(
    type="MetricUpdater",
    metrics=[
        dict(type="LossShow", name="loss_cls"),
        dict(type="LossShow", name="loss_bbox"),
    ]
    if not enable_auto_assign
    else [
        dict(type="LossShow", name="loss_pos"),
        dict(type="LossShow", name="loss_neg"),
        dict(type="LossShow", name="loss_center"),
    ],
    metric_update_func=get_update_metric(),
    # TODO(min.du, 0.5): 1 should not show up here #
    filter_condition=lambda x: x[1] == task_name,
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
)
