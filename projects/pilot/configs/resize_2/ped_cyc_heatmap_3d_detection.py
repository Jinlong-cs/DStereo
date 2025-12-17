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
    datapaths,
    fpn_neck,
    input_hw,
    is_loss_custom,
    is_with_pe,
    lmdb_data,
    log_freq,
    loss_custom_weight_init,
    pe_config,
    resize_hw,
    roi_region,
    standardized_calib_all,
    ufpn_3d_neck,
    undistort_depth_uv,
    vanishing_point,
    with_angle_augmentation,
)

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.core.proj_spec.descs import frcnn_headmap_3d_detection_desc
from hat.utils.apply_func import _as_list

task_name = "ped_cyc_heatmap_3d_detection"

# 行人
# cls_ids_save = [0]
# object_type = "person"

# 骑行人
cls_ids_save = [1]
object_type = "cyclist"

classnames = ["person", "cyclist"]
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
classid_map = {1: 0, 2: -1, 3: 1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1}
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
        hm="ped_cyc_hm",
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
        hm="ped_cyc_hm",
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
                rotation=8,
                location_offset=0.5,
            ),
            loss_custom_weight=loss_custom_weight_init
            if is_loss_custom
            else None,
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
            center=np.array(input_wh),
            scale_wh=(0.5, 0.5),
            undistort_depth_uv=undistort_depth_uv,
            image_hw=input_hw,
            down_stride=down_stride,
            use_bev_nms=True,
            nms_iou_thresh=0.5,
            undistort_2dcenter=True,
            topk=100,
            shift=np.array([0, 0]),
            cls_ids_save=cls_ids_save,
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
        heatmap=torch.zeros((1, 1, resize_hw[0] // 4, resize_hw[1] // 4)),
        box2d_wh=torch.zeros((1, 2, resize_hw[0] // 4, resize_hw[1] // 4)),
        dimensions=torch.zeros((1, 3, resize_hw[0] // 4, resize_hw[1] // 4)),
        location_offset=torch.zeros(
            (1, 2, resize_hw[0] // 4, resize_hw[1] // 4)
        ),
        depth=torch.zeros((1, 2, resize_hw[0] // 4, resize_hw[1] // 4))
        if undistort_depth_uv
        else torch.zeros((1, 1, resize_hw[0] // 4, resize_hw[1] // 4)),
        heatmap_weight=torch.zeros(
            (1, 1, resize_hw[0] // 4, resize_hw[1] // 4)
        ),
        ignore_mask=torch.zeros((1, 1, resize_hw[0] // 4, resize_hw[1] // 4)),
        index=torch.zeros((1, 100)),
        index_mask=torch.zeros((1, 100)),
        location=torch.zeros((1, 100, 3)),
        rotation_y=torch.zeros((1, 100, 1)),
        dimensions_=torch.zeros((1, 100, 3)),
    ),
    val=dict(),
    test=dict(),
)
if is_loss_custom:
    inputs["train"]["loss_custom_weight"] = loss_custom_weight_init

if is_with_pe:
    inputs["train"]["coordinate_map"] = torch.zeros(
        (1, pe_config["pe_c"], pe_config["pe_h"], pe_config["pe_w"])
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

transforms = [
    dict(
        type="Image3DTransform",
        input_wh=input_wh,
        keep_res=keep_res,
        shift=np.array([0, 0], dtype=np.float32),
        keep_aspect_ratio=False,
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
        pe_config=pe_config if is_with_pe else None,
        depth_min_option=depth_min_option,
        keep_meta_keys=["uv_map", "img_wh"]
        if with_angle_augmentation
        else None,
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
        mask_ranges=((0, 100), (0, 100), (860, 960), (540, 640)),
        seed=0,
        prob=0.5,
    )
    transforms.insert(0, cam_stand_transform)
    transforms.insert(3, mask_image_edge_transform)


if lmdb_data:
    ds = datapaths.person_3d_detection
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
        num_workers=1,  # lmdb
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

    data_loader = dict(
        type="MultiCachedDataLoader",
        __build_recursive=False,
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
                pe_config=pe_config if is_with_pe else None,
                depth_min_option=depth_min_option,
                keep_meta_keys=["img_wh", "loss_custom_weight"],
            ),
        ],
    )
    if is_loss_custom:
        data_loader["dataset"]["loss_custom_weight"] = loss_custom_weight
