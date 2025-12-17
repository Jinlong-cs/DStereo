import os

import torch
from common import (
    backbone,
    bev_feat_encoder,
    bev_h_value,
    bev_w_value,
    enable_temporal_fusion,
    get_dataset_list,
    get_lmdb_dataset_list,
    img_neck,
    local_or_remote_debug,
    point_cloud_range,
    task_name,
    train_batch_size,
)

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.utils import Config

task_num_classes = 3


def get_model():
    model = dict(
        type="MVT4D",
        enable_temporal_fusion=enable_temporal_fusion,
        bev_h=bev_h_value,
        bev_w=bev_w_value,
        pc_range=point_cloud_range,
        backbone=backbone,
        img_neck=img_neck,
        bev_feat_encoder=bev_feat_encoder,
        pts_bbox_head=dict(
            type="DeformableHead",
            pc_range=point_cloud_range,
            num_query=900,
            num_classes=task_num_classes,
            bev_h=bev_h_value,
            bev_w=bev_w_value,
            with_box_refine=True,
            stage2nd_cfg=dict(num_points=5),
            code_index=(
                "CX",
                "CY",
                "W",
                "L",
                "CZ",
                "H",
                "SIN_YAW",
                "COS_YAW",
            ),
            num_layers=6,
            transformerlayers=dict(
                type="BaseTransformerLayer",
                attn_cfgs=[
                    dict(
                        type="MultiheadAttention",
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1,
                    ),
                    dict(
                        type="ObjectDetr3DCrossAtten",
                        pc_range=point_cloud_range,
                        num_points=4,
                        num_levels=1,
                        num_heads=8,
                        embed_dims=256,
                    ),
                ],
                ffn_cfgs=dict(
                    type="FFN",
                    embed_dims=256,
                    feedforward_channels=512,
                    num_fcs=2,
                    ffn_drop=0.1,
                ),
                operation_order=(
                    "self_attn",
                    "norm",
                    "cross_attn",
                    "norm",
                    "ffn",
                    "norm",
                ),
            ),
        ),
        losses=dict(
            type="HungarianBBox3DLoss",
            num_classes=task_num_classes,
            sync_cls_avg_factor=True,
            code_weights=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            loss_cls=dict(
                type="FocalLoss",
                loss_name="loss_cls",
                num_classes=task_num_classes + 1,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0,
            ),
            loss_bbox=dict(type="L1Loss", loss_weight=0.25),
            assigner=dict(
                type="HungarianBBoxAssigner3D",
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
                pc_range=point_cloud_range,
            ),
        ),
        postprocess=dict(
            type="MVTPostProcess",
            num_classes=task_num_classes,
            bbox_coder=dict(
                type="NMSFreeBBoxCoder",
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                pc_range=point_cloud_range,
                max_num=300,
            ),
            test_cfg=dict(
                nms_cfg=dict(
                    use_rotate_nms=True,
                    nms_thr=0.8,
                    score_thr=0.001,
                    max_num=300,
                ),
            ),
        ),
    )

    return model


loss_names = ["loss_cls", "loss_bbox"]
for lay_i in range(6 - 1):
    loss_names.append(f"d{lay_i}.loss_cls")
    loss_names.append(f"d{lay_i}.loss_bbox")

train_metrics = [dict(type="LossShow", name=name) for name in loss_names]

metric_updater = dict(
    type="MetricUpdater",
    metrics=train_metrics,
    metric_update_func=update_metric_using_regex(
        per_metric_patterns=[  # corresponding to metrics
            dict(
                label_pattern=None,
                pred_pattern=f"{name}",
            )
            for name in loss_names
        ]
    ),
    step_log_freq=50,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
)

train_dataset_list = []

# as33_ds_path = os.path.join(
#     os.path.dirname(__file__),
#     f"../../../pilot/configs/datasets/as33_multiview_datasets.py",
# )
# ass33_datapaths = Config.fromfile(as33_ds_path).datapaths
# ass33_train_datapaths = ass33_datapaths.multiview_dynamic_3d_detection.train_data_paths
# for train_data_path in ass33_train_datapaths:
#     train_dataset_list.extend(
#         get_dataset_list(
#             train_data_path["rec_path"],
#             model_setting = "as33",
#             mode = "train")
#     )
try:
    x3c_ds_path = os.path.join(
        os.path.dirname(__file__),
        "../../../pilot/configs/datasets/x3c_multiview_lmdb_datasets.py",
    )
    x3c_datapaths = Config.fromfile(x3c_ds_path).datapaths
    x3c_train_datapaths = (
        x3c_datapaths.multiview_dynamic_3d_detection.train_data_paths
    )
except BaseException:
    x3c_train_datapaths = []

for train_data_path in x3c_train_datapaths:
    train_dataset_list.extend(
        get_lmdb_dataset_list(
            train_data_path["lmdb_path"], model_setting="x3c", mode="train"
        )
    )

try:
    x3c_rec_ds_path = os.path.join(
        os.path.dirname(__file__),
        "../../../pilot/configs/datasets/galaxy_x3c_multiview_datasets.py",
    )
    x3c_rec_datapaths = Config.fromfile(x3c_rec_ds_path).datapaths
    x3c_rec_train_datapaths = (
        x3c_rec_datapaths.multiview_dynamic_3d_detection.train_data_paths
    )
except BaseException:
    x3c_rec_train_datapaths = []

for train_data_path in x3c_rec_train_datapaths:
    train_dataset_list.extend(
        get_dataset_list(
            train_data_path["rec_path"], model_setting="x3c", mode="train"
        )
    )

if local_or_remote_debug:
    train_dataset_list = [train_dataset_list[0], train_dataset_list[-1]]

from functools import partial

from mmcv.parallel import collate

train_dataloader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ConcatDataset",
        datasets=train_dataset_list,
        with_flag=True,
        accumulate_flag=True,
    ),
    batch_size=1,
    batch_sampler=dict(
        type="DistributedGroupInBatchSampler",
        dataset=dict(
            type="ConcatDataset",
            datasets=train_dataset_list,
            with_flag=True,
            accumulate_flag=True,
        ),
        batch_size=train_batch_size,
        # skip_prob=0.75,
        skip_prob=-1,
    ),
    sampler=None,
    collate_fn=partial(collate, samples_per_gpu=train_batch_size),
    # shuffle=True, # shuffle in custom_sampler
    num_workers=4,
    pin_memory=False,
)
