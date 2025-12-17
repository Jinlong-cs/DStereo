# flake8: noqa

import os
from collections import OrderedDict
from functools import reduce

import numpy as np
import torch
from common import (
    backbone,
    batch_size,
    bn_kwargs,
    crop_roi_3d,
    datapaths,
    fpn_neck,
    image_edge_mask_ranges,
    input_hw,
    is_with_pe,
    log_freq,
    pe_config,
    resize_hw,
    roi_region,
    standardized_calib_all,
    ufpn_3d_neck,
    undistort_depth_uv,
    vanishing_point,
    with_cam_standiardization,
    with_image_edge_mask,
    with_sparse_training_stage,
)

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.core.proj_spec.descs import (
    frcnn_headmap_3d_detection_desc,
    position_encoding_desc,
)
from hat.utils.apply_func import _as_list

task_name = "vehicle_heatmap_3d_detection"

object_type = "vehicle"
classnames = ["vehicle"]
num_classes = len(classnames)
classname2idxs = [i for i in range(1, num_classes + 1)]
flatten_classnames = reduce(lambda x, y: x + "_" + y, classnames)

# input
input_wh = input_hw[::-1]
down_stride = 4
keep_res = False  # whether use original image size
normalize_depth = True
focal_length_default = 1114.3466796875

# whehter use depth min mode in dense 3d
depth_min_option = False
# ----- Seperate line -----

# model
max_depth = 50
max_objs = 100
use_depth_rotation_multiscale = False
output_head_with_2d_wh = True

# class_id map to new class_id
classid_map = {1: -1, 2: 0, 3: -1, 4: 0, 5: 0, 6: 0, 7: 0, 8: -1}
# head
rot_channel = 2
if undistort_depth_uv:
    outputs = OrderedDict(
        hm=num_classes,
        dep_u=1,
        dep_v=1,
        rot=rot_channel,
        dim=3,
        loc_offset=2,
    )
    outputs_prefix = OrderedDict(
        hm="vehicle_hm",
        dep_u="dep_u",
        dep_v="dep_v",
        rot="rot",
        dim="dim",
        loc_offset="loc_offset",
        wh="wh",
    )
else:
    outputs = OrderedDict(
        hm=num_classes,
        dep=1,
        rot=rot_channel,
        dim=3,
        loc_offset=2,
    )
    outputs_prefix = OrderedDict(
        hm="vehicle_hm",
        dep="dep",
        rot="rot",
        dim="dim",
        loc_offset="loc_offset",
        wh="wh",
    )

if output_head_with_2d_wh:
    outputs["wh"] = 2

output_cfg = {
    out: {
        "out_channels": ch,
        "out_conv_channels": 32,
        "prefix": outputs_prefix[out],
    }
    for out, ch in outputs.items()
}

target = (
    dict(
        type="HeatMap3DTargetGenerator",
        num_classes=num_classes,
        classid_map=classid_map,
        normalize_depth=normalize_depth,
        focal_length_default=focal_length_default,
        down_stride=down_stride,
        min_box_edge=8,  # original image size
        max_depth=max_depth,
        max_objs=max_objs,
        undistort_2dcenter=True,
        undistort_depth_uv=undistort_depth_uv,
        depth_min_option=depth_min_option,
        node_name=f"{task_name}_target",
    )
    if not with_sparse_training_stage
    else None
)


def get_model(mode):
    model = dict(
        type="Camera3D",
        backbone=backbone,
        neck=dict(type="ExtSequential", modules=[fpn_neck, ufpn_3d_neck]),
        head=dict(
            type="ExtSequential",
            modules=[
                dict(
                    type="Camera3DHead",
                    output_cfg=output_cfg,
                    bn_kwargs=bn_kwargs,
                    in_strides=[4],
                    in_channels=[16],
                    out_stride=4,
                    node_name=f"{task_name}_head",
                ),
                dict(
                    type="AddDesc",
                    per_tensor_desc=frcnn_headmap_3d_detection_desc(
                        task_name="frcnn_camera_3d_detection",
                        classnames=classnames,
                        undistort_2dcenter=True,
                        undistort_depth_uv=undistort_depth_uv,
                        use_multibin=False,
                        score_threshold=0.48,  # score for software
                        roi_regions=roi_region,
                        vanishing_point=vanishing_point,
                        focal_length_default=focal_length_default,
                        output_head_with_2d_wh=output_head_with_2d_wh,
                    ),
                    node_name=f"{task_name}_desc",
                ),
            ],
        ),
        target=target if mode == "train" else None,
        loss=dict(
            type="Camera3DLoss",
            hm_loss=dict(type="HMFocalLoss"),
            box2d_wh_loss=dict(type="HML1Loss", heatmap_type="weighted"),
            dimensions_loss=dict(type="HML1Loss", heatmap_type="weighted"),
            location_offset_loss=dict(
                type="HML1Loss", heatmap_type="weighted"
            ),
            depth_loss=dict(type="HML1Loss", heatmap_type="weighted"),
            loss_weights=dict(
                heatmap=0.5,
                box2d_wh=0.01,
                depth=0.5,
                dimensions=0.5,
                rotation=2,
                location_offset=0.5,
            ),
            use_depth_rotation_multiscale=use_depth_rotation_multiscale,
            undistort_depth_uv=undistort_depth_uv,
            output_head_with_2d_wh=output_head_with_2d_wh,
            max_depth=max_depth,
            node_name=f"{task_name}_loss",
        )
        if mode == "train"
        else None,
        postprocess=dict(
            type="HeatMap3DDecoder",
            focal_length_default=focal_length_default,
            center=np.array(list(input_wh)),
            scale_wh=(0.5, 0.5),
            undistort_depth_uv=undistort_depth_uv,
            image_hw=input_hw,
            down_stride=down_stride,
            use_bev_nms=True,
            nms_iou_thresh=0.5,
            undistort_2dcenter=True,
            topk=100,
            shift=np.array([0, 0]),
            node_name=f"{task_name}_decoder",
        )
        if "val" in mode
        else None,
    )

    if is_with_pe:
        model.update(
            type="Camera3DPE",
            neck=fpn_neck,
            neck_ufpn=ufpn_3d_neck,
        )

    return model


# inputs
inputs = dict(
    train=dict(
        heatmap=torch.zeros(
            (1, 1, input_hw[0] // down_stride, input_hw[1] // down_stride)
        ),
        box2d_wh=torch.zeros(
            (1, 2, input_hw[0] // down_stride, input_hw[1] // down_stride)
        ),
        dimensions=torch.zeros(
            (1, 3, input_hw[0] // down_stride, input_hw[1] // down_stride)
        ),
        location_offset=torch.zeros(
            (1, 2, input_hw[0] // down_stride, input_hw[1] // down_stride)
        ),
        depth=torch.zeros(
            (1, 2, input_hw[0] // down_stride, input_hw[1] // down_stride)
        )
        if undistort_depth_uv
        else torch.zeros(
            (1, 1, input_hw[0] // down_stride, input_hw[1] // down_stride)
        ),
        heatmap_weight=torch.zeros(
            (1, 1, input_hw[0] // down_stride, input_hw[1] // down_stride)
        ),
        ignore_mask=torch.zeros(
            (1, 1, input_hw[0] // down_stride, input_hw[1] // down_stride)
        ),
        index=torch.zeros((1, 100)),
        index_mask=torch.zeros((1, 100)),
        location=torch.zeros((1, 100, 3)),
        rotation_y=torch.zeros((1, 100, 1)),
        dimensions_=torch.zeros((1, 100, 3)),
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
    val=dict(
        calib=torch.zeros((1, 3, 4)),
        distCoeffs=torch.zeros((1, 8)),
    ),
    test=dict(),
)


loss_names = [
    "hm_loss",
    "rot_loss",
    "dep_loss",
    "wh_loss",
    "dim_loss",
    "loc_offset_loss",
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

# TODO(zihan.qiu): rm old transfroms
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
        type="Heatmap3DDetectionLableGenerate",
        num_classes=num_classes,
        classid_map=classid_map,
        normalize_depth=normalize_depth,
        focal_length_default=focal_length_default,
        alpha_in_degree=False,
        down_stride=down_stride,
        use_bbox2d=True,
        use_project_bbox2d=False,
        enable_ignore_area=True,
        shift=np.array([0, 0], dtype=np.float32),
        filtered_name="__front_0820__",
        min_box_edge=8,  # original image size
        max_depth=max_depth,
        max_objs=max_objs,
        undistort_2dcenter=True,
        undistort_depth_uv=undistort_depth_uv,
        crop_roi=crop_roi_3d,
        pe_config=pe_config if is_with_pe else None,
        depth_min_option=depth_min_option,
        keep_meta_keys=["uv_map"] if with_cam_standiardization else None,
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

ds = datapaths.vehicle_3d_detection
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
