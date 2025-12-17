import os
from copy import deepcopy

import numpy as np
import torch
from common import (
    backbone,
    batch_size,
    bn_kwargs,
    crop_roi_3d,
    datapaths,
    decoder3d_input_hw,
    fix_channel_neck,
    fpn_neck,
    image_edge_mask_ranges,
    input_hw,
    is_with_pe,
    log_freq,
    model_thresh,
    pe_config,
    pe_desc,
    roi_region,
    standardized_calib_all,
    ufpn_3d_neck,
    undistort_depth_uv,
    use_rotation_iou,
    vanishing_point,
    with_cam_standiardization,
    with_image_edge_mask,
    with_sparse_training_stage,
)
from person_detection import (
    anchor_desc,
    anchor_generator,
    anchor_head,
    anchor_pred,
    classnames,
    object_type,
    val_decoders,
)

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.core.proj_spec.descs import frcnn_roi_3d_detection_desc
from hat.core.proj_spec.detection import classname2id
from hat.models.backbones.mixvargenet import MixVarGENetConfig
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
use_exp_iou_label = True

# model
max_depth = 80
max_objs = 100

# class_id map to new class_id
classid_map = {1: 0, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1}

rpn_proposal_num_per_ctx = 128
down_stride = 4

# TODO(zihan.qiu): rm old transforms and module
proposal_target = (
    dict(
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
    if with_sparse_training_stage
    else dict(
        type="ProposalTargetRoi3D",
        focal_length_default=focal_length_default,
        classid_map=classid_map,
        min_box_edge=6,  # original image size
        max_gt_boxes_num=max_objs,
        max_depth=max_depth,
        undistort_depth_uv=undistort_depth_uv,
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
)


def get_model(mode):
    train_pred = deepcopy(anchor_pred)
    train_pred.update(
        pre_nms_top_k=2000,
        post_nms_top_k=rpn_proposal_num_per_ctx,
        nms_padding_mode="rollover",
        bbox_min_hw=(2, 2),
        node_name=f"{task_name}_anchor_pred",
    )

    if with_sparse_training_stage:
        roi_target_opt_keys = (
            (
                "trans_mat",
                "calib",
                "distCoeffs",
                "eq_fu",
                "eq_fv",
            )
            if undistort_depth_uv
            else ("trans_mat", "calib", "distCoeffs")
        )
    else:
        roi_target_opt_keys = (
            "calib",
            "bboxes",
            "location_offsets",
            "dims",
            "rotation_ys",
            "depths",
            "locations",
            "trans_matrix",
            "distCoeffs",
            "size",
            "img_wh",
            "cls_ids",
            "im_hw",
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
            target_keys=("gt_boxes", "gt_boxes_num")
            if with_sparse_training_stage
            else (),
            target_opt_keys=roi_target_opt_keys,
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
                        type="RCNNHM3DMixVarGEHead",
                        bn_kwargs=bn_kwargs,
                        mid_num_filter=64,
                        rot_channel=2,
                        stop_gradient=with_sparse_training_stage,
                        with_iou_pred=True,
                        head_config=[
                            MixVarGENetConfig(
                                in_channels=16,
                                out_channels=64,
                                head_op="mixvarge_f2",
                                stack_ops=[],
                                stride=1,
                                extra_downsample_num=0,
                            ),
                            MixVarGENetConfig(
                                in_channels=64,
                                out_channels=64,
                                head_op="mixvarge_f4",
                                stack_ops=[],
                                stride=1,
                                extra_downsample_num=0,
                            ),
                            MixVarGENetConfig(
                                in_channels=64,
                                out_channels=64,
                                head_op="mixvarge_f4",
                                stack_ops=[],
                                stride=1,
                                extra_downsample_num=0,
                            ),
                        ],
                        undistort_depth_uv=undistort_depth_uv,
                        node_name=f"{task_name}_share_head",
                    ),
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
                                scale_wh=(0.5, 0.5),
                                class_names=classnames,
                                undistort_2dcenter=True,
                                undistort_depth_uv=undistort_depth_uv,
                            ),
                            node_name=f"{task_name}_desc",
                        )
                    ]
                    if "test" in mode
                    else []
                ),
            ),
            target=proposal_target if mode == "train" else None,
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
                iou_loss_scale=10.0,
                rot_weight=8.0,
                undistort_depth_uv=undistort_depth_uv,
                use_rotation_iou=use_rotation_iou,
                adjust_depth_loss_by_iou=adjust_depth_loss_by_iou,
                use_exp_iou_label=use_exp_iou_label,
                node_name=f"{task_name}_loss",
            )
            if "train" in mode
            else None,
            postprocess=dict(
                type="ROI3DDecoder",
                focal_length_default=focal_length_default,
                scale_wh=(0.5, 0.5),
                undistort_depth_uv=undistort_depth_uv,
                image_hw=decoder3d_input_hw,
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

        model["roi_module"].update(
            type="RoIModulePE",
        )

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
    )
    if with_sparse_training_stage
    else dict(
        location_offsets=torch.zeros((1, 1, 3)),
        trans_matrix=torch.tensor(
            [[[1, 0, 0], [0, 1, 0]]], dtype=torch.float32
        ),
        ignore_mask=torch.zeros(
            (1, input_hw[0] // down_stride, input_hw[1] // down_stride, 1)
        ),
        bboxes=torch.tensor([[[1, 2, 100, 200]]]),
        alphas=torch.ones((1, 1)),
        dims=torch.ones((1, 1, 3)),
        depths=torch.ones((1, 1)),
        locations=torch.ones((1, 1, 3)),
        rotation_ys=torch.ones((1, 1)),
        cls_ids=torch.tensor([[0]]),
        calib=torch.tensor([[[1, 0, 1, 0], [0, 1, 1, 0], [0, 0, 1, 0]]]),
        distCoeffs=torch.ones((1, 8)),
        img_wh=torch.tensor([[input_hw[1], input_hw[0]]]),
        im_hw=torch.tensor([[input_hw[0], input_hw[1]]]),
    ),
    val=dict(),
    test=dict(),
)

if undistort_depth_uv:
    inputs["train"]["eq_fu"] = torch.zeros((1, 1, 2))
    inputs["train"]["eq_fv"] = torch.zeros((1, 1, 2))


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

new_transforms = [
    dict(
        type="BBoxGenerate",
        use_bbox2d=True,
        filtered_name="__front_0820__",
        undistort_2dcenter=True,
        crop_roi=crop_roi_3d,
    ),
    dict(type="PEGenerator", pe_config=pe_config),
    dict(
        type="AffineTransform",
        input_wh=input_wh,
        keep_res=keep_res,
        max_objs=max_objs,
        shift=np.array([0, 0], dtype=np.float32),
        keep_aspect_ratio=False,
        undistort_depth_uv=undistort_depth_uv,
        crop_roi=crop_roi_3d,
    ),
]

transforms = [
    dict(
        type="Image3DTransform",
        input_wh=input_wh,
        keep_res=keep_res,
        shift=np.array([0, 0], dtype=np.float32),
        keep_aspect_ratio=False,
        crop_roi=crop_roi_3d,
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
        crop_roi=crop_roi_3d,
        pe_config=pe_config if is_with_pe else None,
        keep_meta_keys=["uv_map", "im_hw"]
        if with_cam_standiardization
        else ["im_hw"],
    ),
]
if with_image_edge_mask:
    transforms.append(
        dict(
            type="MaskImageEdgeTransform",
            # left top right bottom
            mask_ranges=image_edge_mask_ranges,
            seed=0,
            prob=0.5,
            image_channel_order="chw",
        ),
    )
if with_cam_standiardization:
    transforms.insert(
        0,
        dict(
            type="CameraStandardization",
            **standardized_calib_all,
        ),
    )
    new_transforms.insert(
        0,
        dict(
            type="CameraStandardization",
            **standardized_calib_all,
        ),
    )

ds = datapaths.person_3d_detection

classname2idxs = list(map(lambda x: classname2id[x], classnames))
data_paths = [d["data_path"] for d in ds["train_data_paths"]]
sample_weights = [d["sample_weight"] for d in ds["train_data_paths"]]
data_loader = dict(
    type=torch.utils.data.DataLoader,
    drop_last=True,
    num_workers=1,
    persistent_workers=True,
    batch_size=batch_size,
    dataset=dict(
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
                        transforms=transforms
                        if with_sparse_training_stage
                        else new_transforms,
                    )
                    for path_i in _as_list(path)
                ],
            )
            for path in data_paths
        ],
    ),
)
