import copy
import json
import os
from collections import OrderedDict

import torch
import yaml
from common import (
    batch_size_factor,
    debug_mode,
    feat_channels,
    global_desc,
    input_hw,
    log_freq,
    out_stride2channels,
    out_strides,
    pipeline_test,
    training_step,
)

from hat.data.collates.collates import collate_2d

# dataloader
train_num_workers = 0 if debug_mode else 2 if pipeline_test else 8
val_num_workers = 0 if debug_mode else 2 if pipeline_test else 2
test_num_workers = 0
compose_batchsize = [
    max(1, round(12 * batch_size_factor)),
    max(1, round(6 * batch_size_factor)),
]
train_batch_size_per_gpu = int(sum(compose_batchsize))
test_batch_size_per_gpu = int(6 * batch_size_factor)
val_batch_size_per_gpu = int(15 * batch_size_factor)
input_sequence_length = 1
task_name = "face_plate"
task_loss_weight = 1.0
loss_weight = 1.0
stacked_convs = 1

share_conv = False
norm_target_bbox = True
# skip stride 4 in resize-module's head
head_out_strides = [4]
int8_output = True
desc_task_name = "fcos_detection"  # "detection"
desc_class_names = list(task_name.split("_"))
num_classes = len(desc_class_names)
desc_out_names = [
    "score",
    "bbox",
    "centerness",
]
score_threshold = 0.5
divisor = 64
yaml_path = os.path.join(
    os.path.dirname(__file__),
    "dataset_test.yaml" if debug_mode or pipeline_test else "dataset.yaml",
)
dataset_dict = yaml.load(open(yaml_path, "r"), Loader=yaml.FullLoader)

face_data_paths = dataset_dict[desc_class_names[0]]
plate_data_paths = dataset_dict[desc_class_names[1]]

use_iou_replace_ctrness = True
default_regress_ranges = ((-1, 512),)

range_multiplier = 0.5
regress_ranges = tuple(
    (x * range_multiplier, y * range_multiplier)
    for x, y in default_regress_ranges
)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.1,
    nms=dict(name="nms", iou_threshold=0.2),
    max_per_img=100,
)

# -------------------------- data --------------------------
resize_hw = (540, 960)
transforms = [
    dict(
        type="Resize",
        img_scale=resize_hw,
        keep_ratio=True,
        ratio_range=(0.5, 1.5),
    ),
    dict(
        type="RandomCrop",
        size=input_hw,
        min_area=-1,
        min_iou=0.1,
        truncate_gt=False,
    ),
    dict(type="RandomFlip", px=0.5),
    dict(type="Pad", size=input_hw, pad_val=0),
    # note: Normalize needs to be after ToTensor if `to_yuv`==True
    dict(type="ToTensor", to_yuv=True),
    dict(type="Normalize", mean=128.0, std=128.0),
]
# face cls_id 10
# plate cls_id 2
face_dataset = dict(
    type="ConcatDataset",
    datasets=[
        dict(
            type="DenseboxDataset",
            data_path=data_path,
            anno_path=data_path.replace(".rec", ".anno.pb_rec"),
            task_type="detection",
            class_id=10,
            category=0,
            to_rgb=True,
            ignore_hard=True,
            # note: Normalize needs to be after ToTensor if `to_yuv`==True
            transforms=[
                dict(
                    type="Resize",
                    img_scale=resize_hw,
                    keep_ratio=True,
                    ratio_range=(0.5, 1.5),
                ),
                dict(
                    type="RandomCrop",
                    size=input_hw,
                    min_area=-1,
                    min_iou=0.1,
                    truncate_gt=False,
                ),
                dict(type="RandomFlip", px=0.5),
                dict(type="Pad", size=input_hw, pad_val=0),
                dict(type="ToTensor", to_yuv=True),
                dict(type="Normalize", mean=128.0, std=128.0),
            ],
        )
        for data_path in face_data_paths
    ],
)
plate_dataset = dict(
    type="ConcatDataset",
    datasets=[
        dict(
            type="DenseboxDataset",
            data_path=data_path,
            anno_path=data_path.replace(".rec", ".anno.pb_rec"),
            task_type="detection",
            class_id=2,
            category=1,
            to_rgb=True,
            ignore_hard=True,
            transforms=transforms,
        )
        for data_path in plate_data_paths
    ],
)
data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ComposeDataset",
        batchsize_list=compose_batchsize,
        datasets=[face_dataset, plate_dataset],
    ),
    sampler=dict(
        type="DistributedCycleMultiDatasetSampler",
        batchsize_list=compose_batchsize,
    ),
    collate_fn=collate_2d,
    batch_size=train_batch_size_per_gpu,
    shuffle=True,
    num_workers=train_num_workers,
    pin_memory=False,
    persistent_workers=train_num_workers > 0,
)

if input_hw == (192, 960):  # h w
    crop_size = (0, 220, 960, 192)  # left top w h
elif input_hw == (256, 960):
    crop_size = (0, 188, 960, 256)
else:
    crop_size = (960, 512)

resize_hw_for_val = (540, 960)
val_transforms = [
    dict(type="Resize", img_scale=resize_hw_for_val, keep_ratio=True),
    dict(type="FixedCrop", size=crop_size),
    dict(type="ToTensor", to_yuv=True),
    dict(type="Normalize", mean=128.0, std=128.0),
    dict(type="Pad", divisor=divisor),
    dict(
        type="Batchify",
        size=input_hw,
        divisor=divisor,
        repeat=input_sequence_length,
    ),
    dict(type="RenameKeys", keys=["imgs|img"]),
]

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Auto2dFromImage",
        data_path=None,
        to_rgb=True,
        return_orig_img=True,
        transforms=val_transforms,
    ),
    batch_size=val_batch_size_per_gpu,
    collate_fn=collate_2d,
    shuffle=False,
    num_workers=val_num_workers,
    pin_memory=False,
    drop_last=False,
)

# set keep_ratio=False in test_data_loader for
# testing different size of images
test_transforms = copy.deepcopy(val_transforms)
# test_transforms[0]["keep_ratio"] = True

test_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Auto2dFromImage",
        data_path="./tmp_img_savedir",
        to_rgb=True,
        return_orig_img=True,
        transforms=test_transforms,
    ),
    batch_size=test_batch_size_per_gpu,
    collate_fn=collate_2d,
    shuffle=False,
    num_workers=test_num_workers,
    pin_memory=False,
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
    scale_factor=None,
    img_shape=None,
    resized_shape=None,
    pad_shape=None,
    keep_ratio=None,
    # unneccesay key-value
    scale=None,
    scale_idx=None,
    # TODO(min.du): using patch #
    crop_offset=None,
    ig_bboxes=None,
    before_pad_shape=None,
    data_path=None,
    padded_img=None,
    crop_bbox=None,
)

val_inputs = copy.deepcopy(inputs)
val_inputs.update(
    dict(
        before_crop_shape=None,
        orig_img=None,
    )
)
val_inputs.pop("gt_bboxes")
val_inputs.pop("gt_classes")
val_inputs.pop("ig_bboxes")
val_inputs.pop("data_path")
val_inputs.pop("crop_bbox")

if training_step == "int_infer":
    test_inputs = dict()
else:
    test_inputs = copy.deepcopy(val_inputs)


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
    name2out.update({task_name: out_module(feats, inner_inputs)})
    return name2out


out_module = dict(
    type="OutputModule",
    head=dict(
        type="FCOSHead",
        num_classes=num_classes,
        in_strides=out_strides,
        out_strides=head_out_strides,
        stride2channels=out_stride2channels,
        feat_channels=feat_channels // 2,
        stacked_convs=stacked_convs,
        use_sigmoid=True,
        share_bn=False,
        share_conv=share_conv,
        upscale_bbox_pred=not norm_target_bbox,
        int8_output=int8_output,
    ),
    head_parser=None,
    target=dict(
        type="FCOSTarget",
        strides=head_out_strides,
        regress_ranges=regress_ranges,
        cls_out_channels=num_classes,
        background_label=num_classes,
        norm_on_bbox=norm_target_bbox,
        use_iou_replace_ctrness=use_iou_replace_ctrness,
        task_batch_list=compose_batchsize,
    ),
    loss=dict(
        type="FCOSLoss",
        cls_loss=dict(
            type="FocalLoss",
            loss_name="loss_cls",
            num_classes=num_classes + 1,
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0 * task_loss_weight * loss_weight,
        ),
        reg_loss=dict(
            type="GIoULoss",
            loss_name="loss_bbox",
            loss_weight=1.0 * task_loss_weight * loss_weight,
        ),
        centerness_loss=dict(
            type="CrossEntropyLoss",
            use_sigmoid=True,
            loss_name="loss_centerness"
            if not use_iou_replace_ctrness
            else "loss_iou",  # noqa
            loss_weight=1.0 * task_loss_weight * loss_weight,
        ),
    ),
    prefix=task_name + "_head",
)


val_out_module = copy.deepcopy(out_module)
val_out_module["head"].update(
    dict(upscale_bbox_pred=True, dequant_output=True)
)
val_out_module["target"] = None
val_out_module["loss"] = None
val_out_module["postprocess"] = dict(
    type="FCOSDecoder",
    num_classes=num_classes,
    strides=head_out_strides,
    test_cfg=test_cfg,
    nms_sqrt=True,
    transforms=val_transforms,
    inverse_transform_key=[
        "scale_factor",
        "crop_offset",
        "before_crop_shape",
    ],
    filter_score_mul_centerness=False,
)

test_out_module = copy.deepcopy(val_out_module)
test_out_module["target"] = None
test_out_module["loss"] = None
add_desc_pp = dict(
    type="AddDesc",
    per_tensor_desc=[
        # num of desc == num of pred output tensors (one tensor per out stride)
        json.dumps(
            dict(
                task=desc_task_name,
                class_name=desc_class_names,
                output_name=out_name,
                stride=s,
                score_threshold=score_threshold,
                **global_desc,
            )
        )
        for s in head_out_strides
        for out_name in desc_out_names
    ],
)
if training_step == "int_infer":
    test_out_module["postprocess"] = add_desc_pp
    test_out_module["head"]["upscale_bbox_pred"] = False
    test_out_module["head"]["dequant_output"] = True
else:
    test_out_module["head"]["upscale_bbox_pred"] = True
    test_out_module["head"]["dequant_output"] = True
# only for train
# out_module["head"]["enable_act"] = True
nodes = {f"{task_name}_head": out_module}
val_nodes = {f"{task_name}_head": val_out_module}
test_nodes = {f"{task_name}_head": test_out_module}


# -------------------------- solver --------------------------
def coco_metric_reorganize(out):
    pred_bboxes = [
        data for field, data in zip(out._fields, out) if "pred_bboxes" in field
    ][0]
    img_name = [
        data
        for field, data in zip(out._fields, out)
        if "predict_img_name" in field  # noqa
    ]
    # TODO: need change in multitask
    img_id = getattr(out, task_name + "_" + task_name + "_head_predict_img_id")
    return {"pred_bboxes": pred_bboxes, "img_name": img_name, "img_id": img_id}


def get_update_metric(is_train=False):
    def update_metric(metrics, batch, model_outs):
        if is_train:
            for metric, out in zip(metrics, model_outs[0]):
                if is_train:
                    metric.update(out)
        else:
            # reorganize model prediction results
            assert len(metrics) == 1
            metrics[0].update(coco_metric_reorganize(model_outs[0]))

    return update_metric


metric_updater = dict(
    type="MetricUpdater",
    metrics=[
        dict(type="LossShow", name="loss_cls"),
        dict(type="LossShow", name="loss_bbox"),
        dict(
            type="LossShow",
            name="loss_centerness"
            if not use_iou_replace_ctrness
            else "loss_iou",
        ),  # noqa
    ],
    metric_update_func=get_update_metric(is_train=True),
    # TODO(min.du, 0.5): 1 should not show up here #
    filter_condition=lambda x: x[1] == task_name,
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
)
