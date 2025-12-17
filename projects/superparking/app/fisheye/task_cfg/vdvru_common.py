import json
import os
from copy import deepcopy

import numpy as np
import torch

from hat.data.collates.collates import collate_2d, collate_real3d
from ..common import (  # noqa
    AIDI_PREDICT_TAG,
    aidi_eval,
    all_task_data_files,
    backbone,
    bifpn_out_strides,
    bifpn_stride2channels,
    bn_kwargs,
    bpu_transforms,
    bucket_root,
    commit_level_ci,
    fcos_stacked_convs,
    feat_channels,
    fisheye_input_size,
    get_aidi_eval_info_common,
    input_size,
    is_int_infer,
    job_name,
    model_version,
    neck,
    pipeline_test,
    project_id,
    real3d_num_workers,
    rpn_out_strides,
    save_all_data_paths,
    save_prefix,
    test_image_dir,
    val_num_workers,
    val_only,
)
from ..lib.aidieval_helper import (
    TASK2AIDI_EVAL_DATA_PATH,
    reformat_model_outs_fn_2d_detection,
    reformat_model_outs_fn_rcnn_cls,
    reformat_model_outs_fn_rcnn_det,
    reformat_model_outs_fn_rcnn_kps,
)
from ..lib.utils import get_bev_real3d_lmdb, get_ci_bucket_map
from .detection_common import get_train_transform, val_transforms

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
default_regress_ranges = (
    (-1, 64),
    (64, 128),
    (128, 256),
    (256, INF),
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

_roi_head = dict(
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


_roi_feat_extractor = dict(
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
    nms_post_nms_top_k=15,
    min_bbox_size=0,
    score_thr=0.45,
    nms=dict(name="nms", iou_threshold=0.5),
    nms_padding_mode="pad_zero" if is_int_infer and (not val_only) else None,
    max_per_img=15,
)

pick_stride_neck = dict(
    type="PickStrideNeck",
    in_strides=bifpn_out_strides,
    out_strides=rpn_out_strides,
    node_name="pick_stride_neck",
)

bifpn_quant = dict(type="FPNQuant", stride_num=4, node_name="fpn_quant")

bifpn_dequant = dict(type="FPNDeQuant", node_name="fpn_dequant")

bifpn_desc = dict(
    type="AddDesc",
    per_tensor_desc=[
        # num of desc == num of pred output tensors (one tensor per out stride)
        json.dumps(
            dict(
                task="bifpn",
                size=[
                    1,
                    feat_channels,
                    input_size[0] // s,
                    input_size[1] // s,
                ],
                stride=s,
            )
        )
        for s in rpn_out_strides
    ],
    node_name="fpn_desc",
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
            [input_size[0]],
        ]
    ),
    img_width=torch.Tensor(
        [
            [input_size[1]],
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
    img_shape=[[input_size[0], input_size[1], 3]],
    pad_shape=[[input_size[0], input_size[1], 3]],
    keep_ratio=torch.Tensor([False]),
    # unneccesay key-value
    scale=[torch.Tensor([input_size[1]]), torch.Tensor([input_size[0]])],
    scale_idx=torch.Tensor([0]),
)

val_inputs = deepcopy(inputs)
val_inputs.pop("pad_shape")
val_inputs.pop("keep_ratio")
val_inputs.pop("scale")
val_inputs.pop("scale_idx")
val_inputs.pop("gt_bboxes")
val_inputs.pop("gt_classes")
val_inputs.pop("scale_factor")

test_inputs = deepcopy(val_inputs)


def get_inputs(mode):
    if mode == "train":
        return inputs
    elif mode == "val":
        return val_inputs
    else:
        if is_int_infer:
            return {}
        else:
            return test_inputs


fpn_desc = dict(
    type="AddDesc",
    node_name="fpn_desc",
    per_tensor_desc=[
        # num of desc == num of pred output tensors (one tensor per out stride)
        json.dumps(
            dict(
                task="bifpn",
                size=[
                    1,
                    feat_channels,
                    input_size[0] // s,
                    input_size[1] // s,
                ],
                stride=s,
            )
        )
        for s in rpn_out_strides
    ],
)


def get_mtfcos3d_module(object_type, mode, **kwargs):

    loss_2d_2dBatch_weight = kwargs.get("loss_2d_2dBatch_weight", 1.0)
    loss_2d_3dBatch_weight = kwargs.get("loss_2d_3dBatch_weight", 1.0)
    lr_factor = kwargs.get("lr_factor", 1.0)
    is_train_3d_branch = kwargs.get("is_train_3d_branch", True)
    reg_2d_factor = kwargs.get("reg_2d_factor", 1.0)
    disentangled_corner3D = kwargs.get("disentangled_corner3D", False)
    roi_task_key = kwargs.get("roi_task_key", "gt_boxes")
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
        nms_score_thr=0.4,
        reg_2d_factor=reg_2d_factor,
        roi_task_key=roi_task_key,
        node_name=f"fcos2d3d_{object_type}",
    )


def get_model_mtf3d(mode, object_type, **kwargs):

    return dict(
        type="SingleStageDetector",
        backbone=backbone,
        neck=dict(type="ExtSequential", modules=[neck, pick_stride_neck]),
        box_module=get_mtfcos3d_module(object_type, mode, **kwargs),
    )


def get_out_module(
    obj_type,
    mode,
    num_classes=1,
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
    roi_task_key=None,
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
            in_strides=rpn_out_strides,
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
            input_shape=input_size,
            test_cfg=roi_task_post_cfg,
            nms_sqrt=True,
        ),
        roi_task_key=roi_task_key,
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
        fpn_desc=None,
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
                    nms_thr=0.3,
                    max_per_img=100,
                    iou_threshold=0.4,
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
            node_name=f"mtfcos_adddesc_{obj_type}",
            per_tensor_desc=get_desc(obj_type),
        )

        if is_int_infer and mode == "test":
            module["postprocess"] = add_desc_pp
            module["head"].update(upscale_bbox_pred=False, dequant_output=True)
            # fpn features only need to be output once
            if obj_type == "vehicle" and not val_only:
                module["fpn_desc"] = fpn_desc
                module["output_fpn_feats"] = True
                module["fpn_dequant"] = bifpn_dequant
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
use_mosaic_aug = True
use_mixup_aug = "yolox"  # ["yolox", "yolov5", False / None / ""]


def get_2d_detection_dataset(
    mode, rec_paths, anno_paths, task_class_id, train_sample_weights=None
):
    if save_all_data_paths:
        with open(all_task_data_files, "at") as fw:
            fw.writelines([p + "\n" for p in rec_paths + anno_paths])
    if aidi_eval:
        datasets_ = [
            dict(
                type="Auto2dFromRawJson",
                anno_path=anno_path_i,
                transforms=bpu_transforms,
                buf_only=True,
                task_type="detection",
                input_hw=(1152, 1408),
            )
            for anno_path_i in anno_paths
        ]
    else:
        datasets_ = [
            dict(
                type="DenseboxDataset",
                data_path=rec_path_i,
                anno_path=anno_path_i,
                class_id=task_class_id,
                category=0,
                to_rgb=True,
                transforms=get_train_transform(use_mosaic_aug, use_mixup_aug)
                if mode == "train"
                else val_transforms,
                ignore_hard=True,
            )
            for rec_path_i, anno_path_i in zip(rec_paths, anno_paths)
        ]
    if mode == "train":
        concat_dataset = dict(
            type="DistributedComposeRandomDataset",
            sample_weights=train_sample_weights,
            datasets=datasets_,
        )
    elif mode == "val":
        concat_dataset = dict(
            type="ConcatDataset",
            datasets=datasets_,
        )

    return concat_dataset


def get_real3d_dataloader(
    batch_size,
    mode,
    transforms,
    num_classes=1,
    view="round",
    rec_paths=None,
    num_dist=8,
    use_bev_lmdb=False,
    version_yaml=None,
    spec_yaml=None,
    version=None,
    lmdb_transforms=None,
    lmdb_weight=0.5,
    infer_type="pack",
):
    assert mode in ["train", "val", "test"]
    if use_bev_lmdb:
        assert all(
            [version_yaml, spec_yaml, version, lmdb_transforms]
        ), "yaml path and version must be set when use lmdb is True."

    if save_all_data_paths:
        tbwrite = []
        if rec_paths:
            tbwrite += rec_paths
        with open(all_task_data_files, "at") as fw:
            fw.writelines([p + "\n" for p in tbwrite])
    if commit_level_ci and rec_paths is not None:
        CI_BUCKET_MAP = get_ci_bucket_map(bucket_root)
        for i in range(len(rec_paths)):
            rec_path = rec_paths[i]
            for rt_src, rt_dst in CI_BUCKET_MAP.items():
                if rt_src in rec_path:
                    rec_path = rec_path.replace(rt_src, rt_dst)
            rec_paths[i] = rec_path

    drop_last = mode == "train"
    num_workers = real3d_num_workers[mode]
    data_loader = dict(
        type=torch.utils.data.DataLoader,
        collate_fn=collate_real3d,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
        multiprocessing_context=None if pipeline_test else "spawn",
    )
    if mode in ["train", "val"]:
        datasets = [
            dict(
                type="Real3DDatasetRec",
                paths=rec_paths,
                num_classes=num_classes,
                transforms=transforms,
                num_dist=num_dist,
                to_rgb=True,
                view=view,
                select_sample=True if mode == "train" else False,
            )
        ]
        weights = [1]
        if use_bev_lmdb:
            lmdm_datasets = get_bev_real3d_lmdb(
                version_yaml,
                spec_yaml,
                version,
                bucket_root,
                lmdb_transforms,
                num_dist,
            )
            datasets.append(lmdm_datasets)
            weights = [1 - lmdb_weight, lmdb_weight]
        if mode == "train":
            dataset = dict(
                type="DistributedComposeRandomDataset",
                datasets=datasets,
                sample_weights=weights,
            )
        else:
            dataset = dict(
                type="ConcatDataset",
                datasets=datasets,
            )
        data_loader["dataset"] = dataset
    else:
        data_loader = None
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

            out = model_outs[0]
            pred_bboxes_ = [
                data
                for field, data in zip(out._fields, out)
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


def get_2d_det_aidi_eval_callback(
    dataset_id, task_name, classname, category_ids, rescale_to_shape=None
):
    aidi_eval_callback = dict(
        type="AIDIEval",
        project_id=project_id,
        output_root=os.path.join(save_prefix, job_name, "prediction"),
        prediction_name=model_version,
        prediction_tags=AIDI_PREDICT_TAG,
        aidi_eval_dataset_id=[dataset_id],
        aidi_eval_dataset_name=[],
        reformat_input_fn=None,
        reformat_output_fn=reformat_model_outs_fn_2d_detection,
        reformat_out_fn_kwargs=dict(
            task_name=task_name,
            class_name=classname,
            category_ids=category_ids,
            rescale_to_shape=rescale_to_shape,
        ),
    )
    return aidi_eval_callback


EVAL_TYPE_2D_DET = "detection"
EVAL_TYPE_2D_RCNN_CLS = "rcnn_classification"
EVAL_TYPE_2D_RCNN_DET = "rcnn_detection"
EVAL_TYPE_2D_RCNN_KPS = "rcnn_keypoints"


def get_vdvru_aidi_eval_info(task_name, classname):
    reformat_kwargs = dict(
        task_name=task_name,
        class_name=classname,
        category_ids=[0],
        rescale_to_shape=fisheye_input_size,
    )
    return get_aidi_eval_info_common(
        task_name,
        EVAL_TYPE_2D_DET,
        reformat_model_outs_fn_2d_detection,
        reformat_kwargs,
    )


def get_rcnn_aidi_eval_info(task_name, classname):
    reformat_kwargs = dict(
        task_name=task_name,
        class_name=classname,
        category_ids=[0],
        rescale_to_shape=fisheye_input_size,
    )

    if "classification" in task_name:
        reformat_fn = reformat_model_outs_fn_rcnn_cls
        eval_type = EVAL_TYPE_2D_RCNN_CLS
    if "detection" in task_name:
        reformat_fn = reformat_model_outs_fn_rcnn_det
        eval_type = EVAL_TYPE_2D_RCNN_DET
    if "kps" in task_name:
        reformat_fn = reformat_model_outs_fn_rcnn_kps
        eval_type = EVAL_TYPE_2D_RCNN_KPS

    return get_aidi_eval_info_common(
        task_name,
        eval_type,
        reformat_fn,
        reformat_kwargs,
    )


def get_aidi_eval_dataset(
    task_name,
    transforms=None,
    return_img_buf=True,
    return_orig_hw=True,
    infer_model_type=None,
):
    if transforms is None:
        transforms = bpu_transforms
        if infer_model_type == "rcnn":
            if "classification" in task_name:
                transforms[1].update(task_type="rcnn_classification")
            elif "kps" in task_name:
                transforms[1].update(task_type="rcnn_kps")
            elif "detection" in task_name:
                transforms[1].update(task_type="rcnn_detection")

    aidi_eval_datasets = TASK2AIDI_EVAL_DATA_PATH[task_name]
    if commit_level_ci:
        CI_BUCKET_MAP = get_ci_bucket_map(bucket_root)
        for i, eval_path in enumerate(aidi_eval_datasets):
            for rt_src, rt_dst in CI_BUCKET_MAP.items():
                if rt_src in eval_path:
                    eval_path = eval_path.replace(rt_src, rt_dst)
            aidi_eval_datasets[i] = eval_path
    return [
        dict(
            type="Auto2dFromImage",
            data_path=path,
            to_rgb=True,
            transforms=transforms,
            return_img_buf=return_img_buf,
            return_orig_hw=return_orig_hw,
            infer_model_type=infer_model_type,
        )
        for path in aidi_eval_datasets
    ]


def get_2d_aidi_eval_loaders(
    batch_size,
    task_name,
    infer_model_type=None,
    transforms=None,
    return_img_buf=True,
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
        for ds in get_aidi_eval_dataset(
            task_name,
            transforms,
            return_img_buf,
            infer_model_type=infer_model_type,
        )
    ]
    return aidi_eval_loaders
