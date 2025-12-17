import os

import numpy as np
import torch
from common import enable_temporal_fusion, num_cameras
from common import point_cloud_range as pcr
from common import task_name

from hat.callbacks.metric_updater import update_metric_using_regex

num_classes = 3
embed_dims = 256
num_groups = 8
num_single_frame_decoder = 1
num_decoder = 6
use_deformable_func = os.getenv("SPARSE4D_USE_DEFORMABLE_FUNC", "1") == "1"
strides = [4, 8, 16, 32]
num_levels = len(strides)
num_depth_layers = 3


def generate_points(x0, y0, z0, x1, y1, z1, nx, ny, nz):
    z = np.linspace(z0, z1, nz)
    x = np.linspace(x0, x1, nx)
    y = np.linspace(y0, y1, ny)

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    points = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T
    return points


def gen_anchor_array(num_anchor_x, num_anchor_y, num_anchor_z):
    num_anchor = num_anchor_x * num_anchor_y * num_anchor_z
    anchor = np.zeros((num_anchor, 11))
    xyz = generate_points(
        pcr[0],
        pcr[1],
        (pcr[2] + pcr[5]) / 2.0,
        pcr[3],
        pcr[4],
        (pcr[2] + pcr[5]) / 2.0,
        num_anchor_x,
        num_anchor_y,
        num_anchor_z,
    )
    anchor[:, :3] = xyz[:, :3]
    anchor[:, 3:7] = 1.0
    return anchor


def get_sparse_model():

    model = dict(
        type="Sparse4D",
        use_deformable_func=use_deformable_func,
        backbone=dict(
            type="ResNet50V2",
            num_classes=1000,
            group_base=8,
            include_top=False,
            extend_features=False,
            bn_kwargs=dict(eps=1e-5, momentum=0.1),
        ),
        neck=dict(
            type="FPN",
            in_strides=[2, 4, 8, 16, 32],
            in_channels=[64, 256, 512, 1024, 2048],
            out_strides=strides,
            out_channels=[embed_dims] * 4,
            bn_kwargs=dict(eps=1e-5, momentum=0.1),
        ),
        # depth_branch=dict(  # for auxiliary supervision only
        #     type="DenseDepthNet",
        #     embed_dims=embed_dims,
        #     num_depth_layers=num_depth_layers,
        #     loss_weight=0.2,
        # ),
        head=dict(
            type="Sparse4DHead",
            cls_threshold_to_reg=0.05,
            instance_bank=dict(
                type="InstanceBank",
                num_anchor=900,
                embed_dims=embed_dims,
                anchor=gen_anchor_array(30, 30, 1),
                # anchor="/horizon-bucket/matrix/users/xuewu.lin/nuscenes_kmeans900.npy",
                anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
                num_temp_instances=600 if enable_temporal_fusion else 0,
                confidence_decay=0.6,
                enable_trans_with_vel=False,
            ),
            anchor_encoder=dict(
                type="SparseBox3DEncoder",
                embed_dims=embed_dims,
                vel_dims=-1,
            ),
            num_single_frame_decoder=num_single_frame_decoder,
            operation_order=[
                "deformable",
                "ffn",
                "norm",
                "refine",
            ]
            * num_single_frame_decoder
            + [
                "temp_interaction",
                "interaction",
                "norm",
                "deformable",
                "ffn",
                "norm",
                "refine",
            ]
            * (num_decoder - num_single_frame_decoder),
            temp_instance_interaction=dict(
                type="MultiheadAttention",
                embed_dims=embed_dims,
                num_heads=num_groups,
                batch_first=True,
                dropout=0.1,
            ),
            instance_interaction=dict(
                type="MultiheadAttention",
                embed_dims=embed_dims,
                num_heads=num_groups,
                batch_first=True,
                dropout=0.1,
            ),
            norm_layer=dict(
                type=torch.nn.LayerNorm, normalized_shape=embed_dims
            ),
            ffn=dict(
                type="AsymmetricFFN",
                activate=dict(type="ReLU", inplace=True),
                embed_dims=embed_dims,
                num_fcs=2,
                ffn_drop=0.1,
                pre_norm=True,
                in_channels=embed_dims * 2,
                feedforward_channels=embed_dims * 4,
            ),
            deformable_model=dict(
                type="DeformableFeatureAggregation",
                use_deformable_func=use_deformable_func,
                embed_dims=embed_dims,
                num_groups=num_groups,
                num_levels=num_levels,
                num_cams=num_cameras,
                attn_drop=0.15,
                residual_mode="cat",
                use_camera_embed=True,
                enable_trans_with_vel=False,
                kps_generator=dict(
                    type="SparseBox3DKeyPointsGenerator",
                    num_learnable_pts=6,
                    fix_scale=[
                        [0, 0, 0],
                        [0.45, 0, 0],
                        [-0.45, 0, 0],
                        [0, 0.45, 0],
                        [0, -0.45, 0],
                        [0, 0, 0.45],
                        [0, 0, -0.45],
                    ],
                    embed_dims=embed_dims,
                ),
            ),
            refine_layer=dict(
                type="SparseBox3DRefinementModule",
                embed_dims=embed_dims,
                num_cls=num_classes,
                refine_yaw=False,
            ),
            target=dict(
                type="SparseBox3DTarget",
                cls_weight=2.0,
                box_weight=0.25,
                reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 2,
                cls_wise_reg_weights={},
            ),
            loss_cls=dict(
                type="FocalLoss",
                loss_name="loss_cls",
                num_classes=num_classes + 1,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0,
            ),
            loss_reg=dict(type="L1Loss", loss_weight=0.25),
            gt_cls_key="gt_labels_3d",
            gt_reg_key="gt_bboxes_3d",
            decoder=dict(type="SparseBox3DDecoder"),
            reg_weights=[2.0] * 3 + [1.0] * 5,
        ),
    )

    return model


loss_names = []
for decoder_idx in range(num_decoder):
    loss_names.extend([f"loss_cls_{decoder_idx}", f"loss_reg_{decoder_idx}"])
train_metrics = [dict(type="LossShow", name=name) for name in loss_names]

sparse4d_metric_updater = dict(
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
