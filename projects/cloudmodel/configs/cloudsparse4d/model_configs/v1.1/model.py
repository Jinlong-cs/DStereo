import copy
import os

import torch
from config import num_cameras, num_classes

from projects.cloudmodel.configs.cloudsparse4d.utils import get_hdfs_file

embed_dims = 256
num_groups = 8
num_single_frame_decoder = 1
num_decoder = 6
use_deformable_func = os.getenv("SPARSE4D_USE_DEFORMABLE_FUNC", "1") == "1"
strides = [4, 8, 16, 32]
num_levels = len(strides)
num_depth_layers = 3
num_anchor = 1200
# anchor_file = "hdfs://hobot-bigdata/user/cloud_model/files/cloudsparse4d/anchor_sd_1200_vcs_-108.8_-51.2_108.8_51.2.npy"  # noqa
# anchor = get_hdfs_file(anchor_file, reader="np")
anchor = "/horizon-bucket/adas/borui.zhao/cloudbev/sparse4d-anchors/anchor_ped_sd_1800_vcs_-108.8_-51.2_108.8_51.2.npy"
num_temp_instances = 600

# front
front_backbone = dict(
    type="VoVNet",
    spec_name="V-99-eSE",
    norm_eval=True,
    frozen_stages=-1,
    input_ch=3,
    out_features=(
        "stage2",
        "stage3",
        "stage4",
        "stage5",
    ),
)
side_backbone = copy.deepcopy(front_backbone)
# neck
front_fpn_neck = dict(
    type="FPN",
    in_strides=[4, 8, 16, 32],
    in_channels=[256, 512, 768, 1024],
    out_strides=[4, 8, 16, 32],
    out_channels=[256] * 4,
    bn_kwargs=dict(eps=1e-5, momentum=0.1),
)
side_fpn_neck = copy.deepcopy(front_fpn_neck)

# post process
decoder_score_threshold = 0.2
let_nms_args = dict(
    let_nms_param=[
        {
            "p_t": 0.35,
            "min_t": 4.0,
            "max_t": 8.0,
            "radius": 0.36,
            "e_loc_threshold": 0.4,
            "angle_threshold": 3.2,
            "area_threshold": 0.85,
        },
        {
            "p_t": 0.35,
            "min_t": 4.0,
            "max_t": 8.0,
            "radius": 1.6,
            "e_loc_threshold": 0.3,
            "angle_threshold": 0.3,
            "area_threshold": 0.4,
            "allow_yaw_opposite": True,
            "ct_nms_param": {
                "scale_l": 0.85,
                "scale_w": 0.6,
                "use_yaw_filter": True,
                "yaw_threshold": 0.1,
                "use_mutual_ctnms": True,
            },
        },
        {
            "p_t": 0.35,
            "min_t": 4.0,
            "max_t": 8.0,
            "radius": 0.6,
            "e_loc_threshold": 0.5,
            "angle_threshold": 2.4,
            "area_threshold": 0.8,
        },
    ],
    agnostic=False,
)


def get_sparse_model():
    model = dict(
        type="Sparse4D",
        use_deformable_func=use_deformable_func,
        backbone=[front_backbone, side_backbone],
        neck=[front_fpn_neck, side_fpn_neck],
        head=dict(
            type="Sparse4DHead",
            cls_threshold_to_reg=0.05,
            instance_bank=dict(
                type="InstanceBank",
                num_anchor=num_anchor,
                embed_dims=embed_dims,
                anchor=anchor,
                anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
                num_temp_instances=num_temp_instances,
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
                view_pad_mask=False,
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
                with_centerness_branch=True,
                with_yawness_branch=True,
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
            loss_cns=dict(type="CrossEntropyLoss", use_sigmoid=True),
            loss_yns=dict(type="GaussianFocalLoss"),
            gt_cls_key="gt_labels_3d",
            gt_reg_key="gt_bboxes_3d",
            decoder=dict(
                type="SparseBox3DDecoder",
                score_threshold=decoder_score_threshold,
                let_nms_kwargs=let_nms_args,
            ),
            reg_weights=[2.0] * 3 + [1.0] * 5,  # (x y z)=2.0, (h w l yaw)=1.0
        ),
    )
    return model


def get_loss_names():
    loss_names = []
    loss_prefix_list = ["loss_cls", "loss_reg", "loss_cns", "loss_yns"]
    for decoder_idx in range(num_decoder):
        loss_names.extend(
            [f"{prefix}_{decoder_idx}" for prefix in loss_prefix_list]
        )
    return loss_names
