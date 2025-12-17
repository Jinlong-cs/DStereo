import os
from copy import deepcopy

import numpy as np
import torch
from common import (
    backbone,
    batch_size,
    bn_kwargs,
    datapaths,
    fix_channel_neck,
    fpn_neck,
    input_hw,
    input_padding,
    is_loss_custom,
    is_with_pe,
    lmdb_data,
    log_freq,
    loss_custom_weight_init,
    model_thresh,
    pe_config,
    pe_desc,
    roi_region,
    standardized_calib_all,
    ufpn_3d_neck,
    undistort_depth_uv,
    use_rotation_iou,
    val_decoders,
    vanishing_point,
    with_angle_augmentation,
)
from person_detection import (
    anchor_desc,
    anchor_generator,
    anchor_head,
    anchor_pred,
    classnames,
    object_type,
)

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.core.proj_spec.descs import frcnn_roi_3d_detection_desc
from hat.core.proj_spec.detection import classname2id
from hat.utils.apply_func import _as_list

task_name = "person_roi_3d"
task_type = "detection_3d"


data_args = dict(legacy_bbox=True)

# input
input_wh = input_hw[::-1]
keep_res = False  # whether use original image size
normalize_depth = True
focal_length_default = 1114.3466796875

# whether adjust depth loss by iou
adjust_depth_loss_by_iou = False

# whether use exp iou label
use_exp_iou_label = True

# model
max_depth = 50
max_objs = 100

# class_id map to new class_id
classid_map = {1: 0, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1}

rpn_proposal_num_per_ctx = 128

loss_custom_weight_sparse_init = deepcopy(loss_custom_weight_init)
for key in loss_custom_weight_sparse_init:
    loss_custom_weight_sparse_init[key] = torch.ones(
        batch_size, rpn_proposal_num_per_ctx
    )  # shape: (size, 128)


def get_model(mode):
    train_pred = deepcopy(anchor_pred)
    train_pred.update(
        pre_nms_top_k=2000,
        post_nms_top_k=rpn_proposal_num_per_ctx,
        nms_padding_mode="rollover",
        bbox_min_hw=(2, 2),
        node_name=f"{task_name}_anchor_pred",
    )

    model = dict(
        type="TwoStageDetector",
        backbone=backbone,
        neck=fpn_neck,
        rpn_out_keys=[] if mode != "train" else None,
        rpn_module=dict(
            type="AnchorModule",
            anchor_generator=anchor_generator,
            head=anchor_head,
            ext_feat=fix_channel_neck,
            postprocess=train_pred if mode == "train" else anchor_pred,
            target=None,
            loss=None,
            desc=anchor_desc if mode != "train" else None,
        ),
        roi_module=dict(
            type="RoIModule",
            output_head_out=True,
            roi_key="pred_boxes",
            target_opt_keys=(
                "trans_mat",
                "calib",
                "distCoeffs",
                "eq_fu",
                "eq_fv",
                "loss_custom_weight",
            )
            if undistort_depth_uv
            else ("trans_mat", "calib", "distCoeffs", "loss_custom_weight"),
            postprocess_keys=("calib", "distCoeffs"),
            roi_feat_extractor=dict(
                type="MultiScaleRoIAlign",
                output_size=(8, 8),
                feature_strides=[4],
                canonical_level=2,
                aligned=None,
                node_name=f"{task_name}_feat_extractor",
            ),
            ext_feat=ufpn_3d_neck,
            head=dict(
                type="ExtSequential",
                modules=[
                    dict(
                        type="RCNNHM3DVarGNetHead",
                        bn_kwargs=bn_kwargs,
                        roi_out_channel=16,
                        dw_num_filter=64,
                        pw_num_filter=64,
                        pw_num_filter2=64,
                        group_base=8,
                        factor=1,
                        rot_channel=2,
                        is_dim_match=False,
                        stop_gradient=True,
                        with_iou_pred=True,
                        undistort_depth_uv=undistort_depth_uv,
                        node_name=f"{task_name}_share_head",
                    )
                ]
                + (
                    [
                        dict(
                            type="AddDesc",
                            per_tensor_desc=frcnn_roi_3d_detection_desc(
                                task_name="frcnn_roi_3d_detection",
                                roi_expand_param=1.0,
                                roi_regions=roi_region,
                                vanishing_point=vanishing_point,
                                score_threshold=[
                                    model_thresh["thresh_3d"][object_type]
                                    if model_thresh
                                    else 0.20
                                ],  # score for software
                                focal_length_default=focal_length_default,
                                scale_wh=(0.25, 0.25),
                                class_names=classnames,
                                undistort_2dcenter=True,
                                undistort_depth_uv=undistort_depth_uv,
                                input_padding=input_padding,
                            ),
                            node_name=f"{task_name}_desc",
                        )
                    ]
                    if "test" in mode
                    else []
                ),
            ),
            target=dict(
                type="ProposalTarget3D",
                matcher=dict(
                    type="MaxIoUMatcher",
                    pos_iou=0.6,
                    neg_iou=0.5,
                    allow_low_quality_match=False,
                    low_quality_match_iou=0.3,
                    legacy_bbox=data_args["legacy_bbox"],
                ),
                label_encoder=dict(
                    type="RCNN3DLabelFromMatch",
                    feat_h=8,
                    feat_w=8,
                    kps_num=1,
                    gauss_threshold=0.6,
                    gauss_3d_threshold=0.6,
                    gauss_depth_threshold=0.8,
                    undistort_depth_uv=undistort_depth_uv,
                ),
                node_name=f"{task_name}_target",
            )
            if mode == "train"
            else None,
            loss=dict(
                type="RCNNSparse3DLoss",
                num_classes=1,
                proposal_num=rpn_proposal_num_per_ctx,  # 128
                focal_length_default=focal_length_default,  # 1114.3466796875
                kps_loss=dict(
                    type="SmoothL1Loss", loss_weight=1.0, reduction="mean"
                ),
                offset_2d_loss=dict(
                    type="SmoothL1Loss", loss_weight=1.0, reduction="mean"
                ),
                offset_3d_loss=dict(
                    type="SmoothL1Loss", loss_weight=1.0, reduction="mean"
                ),
                depth_loss=dict(
                    type="SmoothL1Loss", loss_weight=1.0, reduction="mean"
                ),
                dim_loss=dict(
                    type="L1Loss", loss_weight=2.0, reduction="mean"
                ),
                iou_loss_scale=10.0 if use_exp_iou_label else 1.0,
                rot_weight=8.0,
                loss_custom_weight=loss_custom_weight_sparse_init
                if is_loss_custom
                else None,
                undistort_depth_uv=undistort_depth_uv,
                use_rotation_iou=use_rotation_iou,
                adjust_depth_loss_by_iou=adjust_depth_loss_by_iou,
                use_exp_iou_label=use_exp_iou_label,
                node_name=f"{task_name}_loss",
            )
            if mode == "train"
            else None,
            postprocess=dict(
                type="ROI3DDecoder",
                focal_length_default=focal_length_default,
                input_padding=input_padding,
                scale_wh=(0.25, 0.25),
                undistort_depth_uv=undistort_depth_uv,
                image_hw=input_hw,
                node_name=f"{task_name}_decoder",
            )
            if "val" in mode
            else None,
        ),
    )

    if is_with_pe:
        model.update(
            type="TwoStageDetectorPE",
            pe_desc=pe_desc if mode != "train" else None,
        )

        model["roi_module"].update(type="RoIModulePE")

    return model


assert object_type in val_decoders
val_decoders[object_type][0].append(task_name)
val_decoders[object_type][1]["task_descs"][task_name] = (
    "pred_roi_3d",
    task_type,
)

# inputs
inputs = dict(
    train=dict(
        gt_boxes=torch.zeros((1, 100, 15))
        if undistort_depth_uv
        else torch.zeros((1, 100, 14)),
        gt_boxes_num=torch.zeros(1),
        trans_mat=torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
        calib=torch.zeros((1, 3, 4)),
        distCoeffs=torch.zeros((1, 8)),
        im_hw=torch.zeros((1, 2)),
    ),
    val=dict(),
    test=dict(),
)
if undistort_depth_uv:
    inputs["train"]["eq_fu"] = torch.zeros((1, 1, 2))
    inputs["train"]["eq_fv"] = torch.zeros((1, 1, 2))

if is_loss_custom:
    inputs["train"]["loss_custom_weight"] = loss_custom_weight_init

if is_with_pe:
    inputs["train"]["coordinate_map"] = torch.zeros(
        (1, pe_config["pe_c"], pe_config["pe_h"], pe_config["pe_w"])
    )

if undistort_depth_uv:
    loss_names = [
        "center_2d_loss",
        "offset_2d_loss",
        "offset_3d_loss",
        "depth_u_loss",
        "depth_v_loss",
        "dim_loss",
        "rot_loss",
        "iou_loss",
    ]
else:
    loss_names = [
        "center_2d_loss",
        "offset_2d_loss",
        "offset_3d_loss",
        "depth_loss",
        "dim_loss",
        "rot_loss",
        "iou_loss",
    ]

metric_updater = dict(
    type="MetricUpdater",
    metrics=[dict(type="LossShow", name=name) for name in loss_names],
    metric_update_func=update_metric_using_regex(
        per_metric_patterns=[  # corresponding to metrics
            dict(
                label_pattern=None,
                pred_pattern=f"^.*{task_name}_{name}$",
            )
            for name in loss_names
        ]
    ),
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
)


# data
num_classes = len(classnames)
transforms = [
    dict(
        type="Image3DTransform",
        input_wh=input_wh,
        keep_res=keep_res,
        shift=np.array([0, 0], dtype=np.float32),
        keep_aspect_ratio=False,
    ),
    dict(
        type="ROIHeatmap3DDetectionLableGenerate",
        num_classes=num_classes,
        classid_map=classid_map,
        normalize_depth=normalize_depth,
        focal_length_default=focal_length_default,
        filtered_name="__front_0820__",
        min_box_edge=6,  # original image size
        max_depth=max_depth,
        max_gt_boxes_num=100,
        is_train=True,
        use_bbox2d=True,
        use_project_bbox2d=False,
        undistort_depth_uv=undistort_depth_uv,
        shift=np.array([0, 0], dtype=np.float32),
        pe_config=pe_config if is_with_pe else None,
        keep_meta_keys=["uv_map", "img_wh", "im_hw"]
        if with_angle_augmentation
        else ["img_wh", "im_hw"],
    ),
    dict(
        type="RoI3DDetInputPadding",
        input_padding=input_padding,
    ),
]

if with_angle_augmentation:
    cam_stand_transform = dict(
        type="CameraStandardization",
        **standardized_calib_all,
    )
    mask_image_edge_transform = dict(
        type="MaskImageEdgeTransform",
        # left top right bottom
        mask_ranges=((0, 50), (0, 50), (430, 480), (270, 320)),
        seed=0,
        prob=0.5,
    )
    transforms.insert(0, cam_stand_transform)
    transforms.insert(3, mask_image_edge_transform)

if lmdb_data:
    ds = datapaths.person_3d_detection
    classname2idxs = list(map(lambda x: classname2id[x], classnames))
    data_paths = [d["data_path"] for d in ds["train_data_paths"]]
    sample_weights = [d["sample_weight"] for d in ds["train_data_paths"]]

    dataset = dict(
        type="DistributedComposeRandomDataset",
        sample_weights=sample_weights,
        datasets=[
            dict(
                type="ConcatDataset",
                datasets=[
                    dict(
                        type="DetSeg2DAnnoDataset",
                        idx_path=os.path.join(path_i, "idx"),
                        img_path=os.path.join(path_i, "img"),
                        anno_path=os.path.join(path_i, "anno"),
                        transforms=transforms,
                    )
                    for path_i in _as_list(path)
                ],
            )
            for path in data_paths
        ],
    )
    data_loader = dict(
        type=torch.utils.data.DataLoader,
        drop_last=True,
        num_workers=1,
        persistent_workers=True,
        batch_size=batch_size,
        dataset=dataset,
    )

else:
    ds = datapaths.person_3d_detection
    rec_paths = [d["rec_path"] for d in ds["train_data_paths"]]
    anno_paths = [d["anno_path"] for d in ds["train_data_paths"]]
    sample_weights = [d["sample_weight"] for d in ds["train_data_paths"]]
    loss_custom_weight = []
    for d in ds["train_data_paths"]:
        if "loss_custom_weight" not in d.keys():
            loss_custom_weight.append(loss_custom_weight_init)
        else:
            loss_custom_weight.append(d["loss_custom_weight"])

    num_classes = len(classnames)
    classname2idxs = list(map(lambda x: classname2id[x], classnames))

    data_loader = dict(
        type="MultiCachedDataLoader",
        __build_recursive=False,
        input_padding=input_padding,
        last_batch="pad",
        batch_size=batch_size,
        num_workers=1,
        shuffle=True,
        chunk_size=32,
        min_prefetch=1,
        max_prefetch=2,
        min_chunk_num=2,
        max_chunk_num=4,
        batched_transform=True,
        skip_batchify=False,
        prefetcher_using_thread=True,
        dataset=dict(
            type="MultiFusedIterableDataset",
            dataset=[
                dict(
                    type="SplitDataset",
                    dataset=dict(
                        type="ConcatDataset",
                        dataset=[
                            dict(
                                type="DefaultBufWithAnnoDataset",
                                img_rec_path=rec_path_ii,
                                anno_rec_path=anno_path_ii,
                                decode_img=False,
                                decode_anno=True if is_loss_custom else False,
                                to_rgb=False,
                            )
                            for rec_path_ii, anno_path_ii in zip(
                                rec_path_i, anno_path_i
                            )
                        ],
                    )
                    if isinstance(rec_path_i, list)
                    and isinstance(anno_path_i, list)
                    else dict(
                        type="DefaultBufWithAnnoDataset",
                        img_rec_path=rec_path_i,
                        anno_rec_path=anno_path_i,
                        decode_img=False,
                        decode_anno=True if is_loss_custom else False,
                        to_rgb=False,
                    ),
                    even_split=False,
                )
                for rec_path_i, anno_path_i in zip(rec_paths, anno_paths)
            ],
            prob=sample_weights,
            balance=True,
        ),
        transform=[
            dict(
                type="DefaultBufDecoder",
                to_rgb=True,
                decode_img=True,
                decode_anno=False if is_loss_custom else True,
            ),
            dict(type="MergeAll3DData", label_key="objects"),
            dict(
                type="ImageTransform",
                input_wh=input_wh,
                keep_res=keep_res,
                shift=np.array([0, 0], dtype=np.float32),
                keep_aspect_ratio=False,
            ),
            dict(
                type="ROIHeatmap3DDetectionLableGenerate",
                num_classes=num_classes,
                classid_map=classid_map,
                normalize_depth=normalize_depth,
                focal_length_default=focal_length_default,
                filtered_name="__front_0820__",
                min_box_edge=6,  # original image size
                max_depth=max_depth,
                max_gt_boxes_num=100,
                is_train=True,
                use_bbox2d=True,
                use_project_bbox2d=False,
                undistort_depth_uv=undistort_depth_uv,
                shift=np.array([0, 0], dtype=np.float32),
                input_padding=[0, 0, *input_padding[:2]],
                pe_config=pe_config if is_with_pe else None,
                keep_meta_keys=["im_hw", "loss_custom_weight"],
            ),
        ],
    )
    if is_loss_custom:
        data_loader["dataset"]["loss_custom_weight"] = loss_custom_weight
