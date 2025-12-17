import json
import os
from copy import deepcopy

import numpy as np
import torch

from hat.data.collates.collates import collate_2d
from ..common import (  # noqa
    aidi_eval,
    backbone,
    bifpn_out_strides,
    bifpn_stride2channels,
    bn_kwargs,
    bpu_transforms,
    compare_version,
    fcos_max_per_img,
    fcos_stacked_convs,
    feat_channels,
    is_int_infer,
    job_name,
    log_freq,
    model_input_size,
    model_version,
    neck,
    pipeline_test_dump,
    project_id,
    rpn_out_strides,
    save_prefix,
    test_num_workers,
    train_num_workers,
    training_step,
    use_mixup_aug,
    use_mosaic_aug,
    val_num_workers,
    val_only_transforms,
)
from ..datasets.auto_2d.datasets import (
    PREDICT_TAGS,
    adas_eval_datadet_id_list,
    dataset_val_img_path,
)
from ..lib.aidieval_helper import reformat_prediction_fn

PI = np.pi


# ------------------- mtfcos 2d/3d common -------------------
INF = 1e8
enable_auto_assign = False
use_iou_replace_ctrness = True
score_threshold = 0
use_2d_score = True
use_output_parser = False
use_multibin = True
range_multiplier = 0.5
multibin_margin = 10.0 / 180.0 * PI
multibin_centers = (0.0, PI / 2, PI, -PI / 2)
default_regress_ranges_base = 48
drrb = default_regress_ranges_base
default_regress_ranges = (
    (-1, drrb),
    (drrb, drrb * 2),
    (drrb * 2, drrb * 4),
    (drrb * 4, INF),
)
regress_ranges = tuple(
    (
        (x * range_multiplier, y * range_multiplier)
        for x, y in (default_regress_ranges)
    )
)

head_channels = dict(
    cls=[1],
    offset_2d_reg=[4],
    offset_3d_group_reg=[2],
    depth_3d_group_reg=[1],
    dim_3d_group_reg=[3],
    dir_reg=[2]
    if not use_multibin
    else [len(multibin_centers)],  # hard code should be fix
    ctrness_2d_reg=[1],
    ctrness_3d_reg=[1],
)
if use_multibin:
    if multibin_margin > 0.0:
        head_channels.update(rot_3d_group_reg=[len(multibin_centers)])
    else:
        head_channels.update(rot_3d_group_reg=[1])
else:
    head_channels.update(rotsin_3d_group_reg=[1])  # local yaw/alpha

loss_3d_reg_weights = [1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0]
use_dynamic_loss_weight = False
if use_dynamic_loss_weight:
    loss_3d_reg_weights = dict(
        hm=1.0,
        rot=10.0,
        dep=3.0,
        dim=1.0,
        loc_offset=3.0,
        wh=0.01,
    )

category_id_to_name = {
    1: ["pedestrian", "pedestrian"],
    2: ["car", "vehicle"],
    3: ["cyclist", "cyclist"],
    4: ["bus", "vehicle"],
    5: ["truck", "vehicle"],
    6: ["specialcar", "vehicle"],
    7: ["tricycle", "vehicle"],
    8: ["dontcar", "dontcare"],
}

roi_head = dict(
    type="RCNNVarGNetShareHead",
    bn_kwargs=bn_kwargs,
    roi_out_channel=32,
    gc_num_filter=128,
    pw_num_filter=128,
    pw_num_filter2=128,
    group_base=8,
    factor=1,
    stride=2,
)

roi_feat_extractor = dict(
    type="MultiScaleRoIAlign",
    output_size=(8, 8),
    feature_strides=rpn_out_strides,
    canonical_level=5,
    aligned=None,
)


roi_task_train_post_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.0,
    nms=dict(name="nms", iou_threshold=0.7),
    max_per_img=30,
)

roi_task_test_post_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.3,
    nms=dict(name="nms", iou_threshold=0.5),
    max_per_img=15,
)

pick_stride_neck = dict(
    type="PickStrideNeck",
    in_strides=bifpn_out_strides,
    out_strides=rpn_out_strides,
    node_name="pick_stride_neck",
)

inputs = dict(
    img_id=torch.Tensor(
        [
            [1],
        ]
    ),
    img_name=[
        "xxx.jpg",
    ],
    img_height=torch.Tensor(
        [
            [model_input_size[0]],
        ]
    ),
    img_width=torch.Tensor(
        [
            [model_input_size[1]],
        ]
    ),
    gt_bboxes=[torch.Tensor([[1, 2, 3, 4], [1, 2, 3, 4]])],
    gt_classes=[torch.Tensor([0, 0])],
    color_space=[
        "yuv",
    ],
    layout=[
        "hwc",
    ],
    scale_factor=torch.randn((1, 4)),
    img_shape=[[*model_input_size[:2], 3]],
    pad_shape=[[*model_input_size[:2], 3]],
    keep_ratio=torch.Tensor([False]),
    # unneccesay key-value
    scale=[
        torch.Tensor([model_input_size[1]]),
        torch.Tensor([model_input_size[0]]),
    ],
    scale_idx=torch.Tensor([0]),
    crop_offset=[[1, 2, 3, 4]],
)

val_inputs = deepcopy(inputs)
val_inputs.pop("pad_shape")
val_inputs.pop("gt_bboxes")
val_inputs.pop("gt_classes")
val_inputs.pop("scale_factor")
val_inputs.pop("keep_ratio")
val_inputs.pop("scale")
val_inputs.pop("scale_idx")

test_inputs = deepcopy(val_inputs)

if aidi_eval:
    val_inputs = test_inputs


def get_inputs(mode):
    if mode == "train":
        return inputs
    elif mode == "val":
        return val_inputs
    else:
        if training_step == "int_infer":
            return {}
        else:
            return test_inputs


def get_mtfcos3d_module(object_type, mode, **kwargs):

    loss_2d_2dBatch_weight = kwargs.get("loss_2d_2dBatch_weight", 1.0)
    loss_2d_3dBatch_weight = kwargs.get("loss_2d_3dBatch_weight", 1.0)
    lr_factor = kwargs.get("lr_factor", 1.0)
    is_train_3d_branch = kwargs.get("is_train_3d_branch", True)
    reg_2d_factor = kwargs.get("reg_2d_factor", 1.0)
    disentangled_corner3D = kwargs.get("disentangled_corner3D", False)
    return get_out_module(
        object_type,
        mode,
        num_classes=1,
        head_channels=head_channels,
        regress_ranges=regress_ranges,
        is_train_3d_branch=is_train_3d_branch,
        dir_offset=0.7854,
        depth_type="Cylindrical",
        loss_3d_reg_weights=loss_3d_reg_weights,
        loss_2d_2dBatch_weight=loss_2d_2dBatch_weight,
        loss_2d_3dBatch_weight=loss_2d_3dBatch_weight,
        disentangled_corner3D=disentangled_corner3D,
        lr_factor=lr_factor,
        rescale=False,
        roi_task_post_cfg=roi_task_train_post_cfg
        if mode == "train"
        else roi_task_test_post_cfg,
        target_attribute_flag="is_train_3d_branch",
        head_attribute_name=(
            "offset_3d_group_reg",
            "depth_3d_group_reg",
            "dim_3d_group_reg",
            "rotsin_3d_group_reg",
            "dir_reg",
            "ctrness_3d_reg",
        ),
        use_multibin=True,
        multibin_centers=multibin_centers,
        multibin_margin=multibin_margin,
        nms_score_thr=0.05,
        reg_2d_factor=reg_2d_factor,
        node_name=f"fcos2d3d_{object_type}",
    )


def get_model_mtf3d(mode, object_type, **kwargs):

    return dict(
        type="SingleStageDetector",
        backbone=backbone,
        # neck=dict(type="ExtSequential", modules=[neck, pick_stride_neck]),
        neck=neck,
        box_module=get_mtfcos3d_module(object_type, mode, **kwargs),
    )


val_only_transforms = deepcopy(val_only_transforms)
val_only_transforms.append(dict(type="DeleteKeys", keys=["before_crop_shape"]))


def get_out_module(
    obj_type,
    mode,
    num_classes=3,
    head_channels=None,
    regress_ranges=None,
    is_train_3d_branch=True,
    dir_offset=0.7854,
    depth_type="Cartesian",
    loss_3d_reg_weights=None,
    loss_2d_2dBatch_weight=1.0,
    loss_2d_3dBatch_weight=1.0,
    disentangled_corner3D=False,
    lr_factor=1.0,
    rescale=False,
    roi_task_post_cfg=None,
    target_attribute_flag="is_train_3d_branch",
    head_attribute_name=(
        "offset_3d_group_reg",
        "depth_3d_group_reg",
        "dim_3d_group_reg",
        "rotsin_3d_group_reg",
        "dir_reg",
        "ctrness_3d_reg",
    ),
    use_multibin=True,
    multibin_centers=(0.0, PI / 2, PI, -PI / 2),
    multibin_margin=0.0,
    nms_score_thr=0.4,
    reg_2d_factor=1.0,
    node_name=None,
):
    assert mode in ["train", "val", "test"]
    head_attribute_name = list(head_attribute_name)
    if use_multibin:
        head_attribute_name[-3] = "rot_3d_group_reg"

    module = dict(
        type="MTFCOS3DOutputModule",
        node_name=node_name,
        head=dict(
            type="MTFCOS3DHead",
            node_name=node_name + "_fcos_head",
            num_classes=num_classes,
            in_strides=bifpn_out_strides,
            out_strides=rpn_out_strides,
            head_channels=head_channels,
            stride2channels=bifpn_stride2channels,
            feat_channels=feat_channels,
            stacked_convs=fcos_stacked_convs,
            head_conv_kernel_size=1,  # output conv kernel size.
            use_sigmoid=True,
            share_bn=False,
            upscale_bbox_pred=False,  # if test mode, True
            int8_output=False,  # high prs
        ),
        head_parser=None,
        roi_decoder=dict(
            type="FCOSDecoder4RCNN",
            node_name=node_name + "_fcos_decoder",
            num_classes=num_classes,
            strides=rpn_out_strides,
            input_shape=model_input_size,
            test_cfg=roi_task_post_cfg,
            nms_sqrt=True,
        ),
        target=dict(
            type="MTFCOS3DTarget",
            node_name=f"{node_name}_fcos_target",
            num_classes=num_classes,
            strides=rpn_out_strides,
            regress_ranges=regress_ranges,
            head_channels=head_channels,
            dir_offset=dir_offset,
            loss_3d_reg_weights=loss_3d_reg_weights,
            loss_2d_2dBatch_weight=loss_2d_2dBatch_weight,
            loss_2d_3dBatch_weight=loss_2d_3dBatch_weight,
            disentangled_corner3D=disentangled_corner3D,
            is_train_3d_branch=is_train_3d_branch,
            use_2d_ctr_pos=True,
            use_iou_replace_ctrness=True,
            depth_type=depth_type,
            use_multibin=use_multibin,
            multibin_centers=multibin_centers,
            multibin_margin=multibin_margin,
        ),
        head_freeze=dict(
            type="FreezeOutputAttibuteGrad",
            node_name=node_name + "_fcos_freeze",
            target_attribute_flag=target_attribute_flag,
            head_attribute_name=head_attribute_name,
        ),
        loss=dict(
            type="MTFCOS3DLoss",
            node_name=node_name + "_fcos_loss",
            cls_loss=dict(
                type="FocalLoss",
                node_name=node_name + "_loss_cls",
                loss_name="loss_cls",
                num_classes=num_classes + 1,
                alpha=0.25,
                gamma=2.0,
                loss_weight=lr_factor,
            ),
            reg_loss=dict(
                type="GIoULoss",
                node_name=node_name + "_loss_bbox",
                loss_name="loss_bbox",
                loss_weight=lr_factor * reg_2d_factor,
            ),
            group_reg_loss=dict(
                type="SmoothL1Loss",
                node_name=node_name + "_group_reg_loss",
                beta=1.0 / 9.0,
                loss_weight=lr_factor,
            ),
            centerness_loss=dict(
                type="CrossEntropyLoss",
                node_name=node_name + "_loss_centerness_reg",
                use_sigmoid=True,
                loss_name="loss_centerness_reg",
                loss_weight=lr_factor,
            ),
            centerness3d_loss=dict(
                type="CrossEntropyLoss",
                node_name=node_name + "_loss_centerness3d_reg",
                use_sigmoid=True,
                loss_name="loss_centerness3d_reg",
                loss_weight=lr_factor,
            ),
            dir_loss=dict(
                type="CrossEntropyLoss",
                node_name=node_name + "_loss_dir_reg",
                use_sigmoid=use_multibin and multibin_margin > 0,
                loss_name="loss_dir_reg",
                loss_weight=lr_factor,
            ),
            corner_loss=dict(
                type="SmoothL1Loss",
                node_name=node_name + "_corner_loss",
                beta=0.05,
                loss_weight=lr_factor,
            ),
            mask_l1_loss=dict(
                type="MaskL1Loss",
                node_name=node_name + "_mask_l1_loss",
                loss_weight=lr_factor,
            ),
        ),
        prefix="fcos3d_head",
    )
    if mode in ["val", "test"]:
        module["head"].update(upscale_bbox_pred=True, dequant_output=True)
        module["target"] = None
        module["loss"] = None

        fcos3d_decoder = (
            dict(
                type="MTFCOS3DDecoder",
                node_name=f"mtfcos_decoder_{obj_type}",
                num_classes=num_classes,
                head_channels=head_channels,
                strides=rpn_out_strides,
                rescale=rescale,  # pred rescale to org image
                dir_offset=dir_offset,  # same as target cfg
                nms_kwargs=dict(
                    nms_pre=200,
                    score_thr=nms_score_thr,
                    nms_thr=0.0,
                    max_per_img=fcos_max_per_img,
                    iou_threshold=0.5,
                    replace=True,
                    nms_sqrt=True,
                    use_score2d=True,
                ),
                depth_type=depth_type,
                is_train_3d_branch=is_train_3d_branch,
                use_multibin=use_multibin,
                multibin_centers=multibin_centers,
                multibin_margin=multibin_margin,
            ),
        )
        add_desc_pp = dict(
            type="AddDesc",
            node_name=f"add_desc_mtfcos_{obj_type}",
            # num of desc == num of pred output tensors
            per_tensor_desc=get_desc(obj_type),
        )

        if is_int_infer and mode == "test":
            module["postprocess"] = add_desc_pp
            module["head"].update(upscale_bbox_pred=False, dequant_output=True)

        else:
            module["postprocess"] = fcos3d_decoder

    return module


def get_desc(obj_name):
    class_2d = {
        "task": f"fcos_{obj_name}_detection",
        "class_name": [obj_name],
        "output_name": "score",
        "score_threshold": 0.5,
        "nms_threshold": 0.5,
    }

    if obj_name in ["cyclist", "person"]:
        class_2d.update(score_threshold=0.45)
        class_2d.update(nms_threshold=0.4)
    else:
        class_2d.update(score_threshold=0.4)
        class_2d.update(nms_threshold=0.4)

    per_tensor_desc = [
        # in order of `head_channels`
        class_2d,
        {
            "task": f"fcos_{obj_name}_detection",
            "class_name": [obj_name],
            "output_name": "bbox",
        },
        {
            "task": f"camera_3d_{obj_name}_detection",
            "output_name": "prj2d_offset_output",
            "class_name": [obj_name],
            "properties": [{"channel_labels": ["offset_x", "offset_y"]}],
        },
        {
            "task": f"camera_3d_{obj_name}_detection",
            "output_name": "depth_output",
            "class_name": [obj_name],
            "properties": [{"channel_labels": ["depth"]}],
        },
        {
            "task": f"camera_3d_{obj_name}_detection",
            "output_name": "dim_output",
            "class_name": [obj_name],
            "properties": [{"channel_labels": ["length", "height", "width"]}],
        },
        {
            "task": f"camera_3d_{obj_name}_detection",
            "class_name": [obj_name],
            "output_name": "dir_output",
            "properties": [
                {"channel_labels": ["bin1", "bin2", "bin3", "bin4"]}
            ],
        },
        {
            "task": f"fcos_{obj_name}_detection",
            "class_name": [obj_name],
            "output_name": "centerness",
        },
        {
            "task": f"camera_3d_{obj_name}_detection",
            "output_name": "centerness3d_output",
            "class_name": [obj_name],
            "properties": [{"channel_labels": ["centerness3d"]}],
        },
        {
            "task": f"camera_3d_{obj_name}_detection",
            "output_name": "rot_output",
            "class_name": [obj_name],
            "use_multibin": 1,
            "properties": [
                {
                    "channel_labels": [
                        "bin1_offset",
                        "bin2_offset",
                        "bin3_offset",
                        "bin4_offset",
                    ]
                }
            ],
        },
    ]
    per_desc = []
    for lvl in per_tensor_desc:
        for s in rpn_out_strides:
            lvl.update(dict(stride=s))
            per_desc.append(json.dumps(lvl))
    return per_desc


# ------------------- 2d detection dataloaders common -------------------
transforms = [
    dict(
        type="RandomCrop",
        center_shake=[60, 60, 300, 300],
        center_crop_prob=0.5,
        min_area=-1,
        min_iou=0.3,
        wh_ratio_range=(0.5, 1.0) if not pipeline_test_dump else (1.0, 1.0),
        repeat_times=100,
        discriminate_ignore_classes=True,
        without_background=True,
    ),
    dict(
        type="Resize",
        img_scale=model_input_size,
        keep_ratio=True,
        pad_to_keep_ratio=True,
    ),
    dict(type="RandomFlip", px=0.5 if not pipeline_test_dump else 0),
    dict(type="ToTensor", to_yuv=False),
    dict(type="Pad", divisor=64),
    dict(type="RenameKeys", keys=["imgs|img"]),
    dict(type="DeleteKeys", keys=["before_pad_shape"]),
]

mosaic_cfg = dict(
    type="DetMosaic",
    img_scale=model_input_size,
    center_ratio_range=(0.75, 1.25),
    p=0.5,
)

mixup_cfgs = {
    "yolox": dict(
        type="DetYOLOXMixUp",
        img_scale=model_input_size,
        p=0.5,
    ),
    "yolov5": dict(
        type="DetYOLOv5MixUp",
        p=0.5,
    ),
}

mix_trans_insert_pos = 3
mix_transform_cfg = None
if use_mosaic_aug and use_mixup_aug:
    mix_transform_cfg = dict(
        type="RandomSelectOne",
        transforms=[
            mosaic_cfg,
            mixup_cfgs[use_mixup_aug],
        ],
        p_trans=[0.9, 0.1],
        p=1.0,
    )
elif use_mosaic_aug:
    mix_transform_cfg = mosaic_cfg
elif use_mixup_aug:
    mix_transform_cfg = mixup_cfgs[use_mixup_aug]

if mix_transform_cfg:
    transforms.insert(mix_trans_insert_pos, mix_transform_cfg)
    transforms.insert(
        mix_trans_insert_pos + 1,
        dict(
            type="Resize",
            img_scale=model_input_size,
            keep_ratio=True,
            pad_to_keep_ratio=True,
        ),
    )


def get_aidi_eval_dataset(dataset_name, transforms=None):
    if transforms is None:
        transforms = val_only_transforms
    return [
        dict(
            type="Auto2dFromImage",
            data_path=path,
            to_rgb=True,
            transforms=transforms,
            return_img_buf=True,
        )
        for path in dataset_val_img_path[dataset_name]
    ]


def get_2d_detection_dataset(
    mode,
    rec_paths,
    anno_paths,
    task_class_id,
    train_sample_weights=None,
    shuffle=not pipeline_test_dump,
):
    datasets_ = [
        dict(
            type="DenseboxDataset",
            data_path=rec_path_i,
            anno_path=anno_path_i,
            class_id=task_class_id,
            category=0,
            to_rgb=True,
            transforms=transforms if mode == "train" else val_only_transforms,
            ignore_hard=True,
            abandon_other_category=True,
        )
        for rec_path_i, anno_path_i in zip(rec_paths, anno_paths)
    ]
    if mode == "train" and not pipeline_test_dump:
        concat_dataset = dict(
            type="DistributedComposeRandomDataset",
            sample_weights=train_sample_weights,
            datasets=datasets_,
            shuffle=shuffle,
        )
    else:
        concat_dataset = dict(
            type="ConcatDataset",
            datasets=datasets_,
        )

    return concat_dataset


def get_2d_detection_dataloader(mode, dataset, batch_size):
    num_workers = eval(f"{mode}_num_workers")
    data_loader = dict(
        type=torch.utils.data.DataLoader,
        dataset=dataset,
        collate_fn=collate_2d,
        batch_size=batch_size[mode],
        num_workers=num_workers,
        pin_memory=False,
    )

    if mode == "train":
        if not pipeline_test_dump:
            assert dataset["type"] == "DistributedComposeRandomDataset"
        data_loader.update(
            dict(
                persistent_workers=num_workers > 0,
                multiprocessing_context=None
                if pipeline_test_dump
                else "spawn",
            )
        )
    else:
        data_loader.update(
            dict(
                drop_last=False,
            )
        )
    if pipeline_test_dump:
        data_loader["sampler"] = dict(
            type=torch.utils.data.DistributedSampler, shuffle=False
        )
    return data_loader


def get_update_metric(is_train=False, min_bbox_height=0, task_name=None):
    def update_metric(metrics, batch, model_outs):
        if is_train:
            for metric, out in zip(metrics, model_outs[0]):
                metric.update(out)
        else:
            # reorganize model prediction results
            # assert len(metrics) == 1
            invalid_gt_index = []
            gt_bboxes = batch[0]["gt_bboxes"]
            gt_labels = batch[0]["gt_classes"]

            out = model_outs[task_name]
            pred_bboxes_ = [
                data
                for field, data in out[0].items()
                if "pred_bboxes" in field
            ][0]
            # filter score==0 box
            pred_bboxes = []
            for pred_bbox in pred_bboxes_:
                pred_bboxes.append(pred_bbox[pred_bbox[:, 4] > 0])

            # del invalid gt data
            invalid_gt_index = [
                i
                for i, gt_bbox in enumerate(gt_bboxes)
                if gt_bbox.shape[0] == 0
            ]
            for index in invalid_gt_index[::-1]:
                pred_bboxes = list(pred_bboxes)
                gt_bboxes.pop(index)
                gt_labels.pop(index)
                pred_bboxes.pop(index)
                pred_bboxes = tuple(pred_bboxes)
            # difficult cases
            gt_difficults = []
            gt_unique_bboxes = []
            gt_unique_labels = []
            for index in range(len(gt_labels)):
                gt_img_labels = (
                    gt_labels[index].flatten().detach().cpu().numpy()
                )
                gt_img_bboxes = gt_bboxes[index].detach().cpu().numpy()
                # remove repeated gt bboxes
                _, unique_index = np.unique(
                    gt_img_bboxes[:, :4], return_index=True, axis=0
                )
                gt_img_labels = gt_img_labels[unique_index]
                gt_img_bboxes = gt_img_bboxes[unique_index]
                gt_unique_bboxes.append(gt_img_bboxes)
                gt_unique_labels.append(gt_img_labels)

                gt_labels_difficult = gt_img_labels < 0
                gt_bboxes_h = gt_img_bboxes[:, 3] - gt_img_bboxes[:, 1]
                gt_bboxes_difficult = gt_bboxes_h < min_bbox_height
                gt_difficult = np.logical_or(
                    gt_labels_difficult, gt_bboxes_difficult
                ).astype(np.int32)
                gt_difficults.append(torch.from_numpy(gt_difficult))
            # gt
            model_outs = {
                "gt_bboxes": gt_unique_bboxes,
                "gt_classes": gt_unique_labels,
                "gt_difficult": gt_difficults,
                "pred_bboxes": pred_bboxes,
            }

            for m in metrics:
                m.update(deepcopy(model_outs))

    return update_metric


def get_2d_detection_aidi_eval_callback(dataset_name, task_name):
    aidi_eval_page_label = task_name
    aidi_eval_dataset_id = adas_eval_datadet_id_list[dataset_name]
    aidi_eval_type = "detection"
    old_prediction = compare_version
    new_prediction = model_version

    aidi_eval_callback = []
    for id in aidi_eval_dataset_id:
        eval_callback = dict(
            type="AIDIEval",
            aidi_eval_dataset_id=id,
            output_root=os.path.join(
                save_prefix, job_name, task_name, str(id)
            ),
            prediction_name=model_version,
            prediction_tags=PREDICT_TAGS,
            project_id=project_id,
            reformat_output_fn=reformat_prediction_fn,
            reformat_out_fn_kwargs={
                "task_name": task_name,
                "transforms": bpu_transforms,
                "inverse_transform_keys": [
                    "crop_offset",
                ],
            },
        )
        aidi_eval_callback.append(
            [
                eval_callback,
                dict(
                    type="StatsMonitor",
                    log_freq=log_freq,
                ),
            ]
        )
    return (
        aidi_eval_callback,
        aidi_eval_page_label,
        aidi_eval_dataset_id,
        aidi_eval_type,
        old_prediction,
        new_prediction,
    )


def get_2d_detection_aidi_eval_loaders(
    batch_size, dataset_name, transforms=None
):
    aidi_eval_loaders = [
        dict(
            type=torch.utils.data.DataLoader,
            dataset=ds,
            sampler=dict(type=torch.utils.data.DistributedSampler),
            collate_fn=collate_2d,
            batch_size=batch_size,
            shuffle=False,
            num_workers=val_num_workers,
            pin_memory=False,
            drop_last=False,
        )
        for ds in get_aidi_eval_dataset(dataset_name, transforms)
    ]
    return aidi_eval_loaders
