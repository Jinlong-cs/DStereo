import copy

import pytest
import torch

from hat.registry import build_from_registry

try:
    import mmcv

    _MMCV_IMPORTED = True
except ImportError:
    _MMCV_IMPORTED = False


def get_model_config_and_data():
    bev_h_value = 50
    bev_w_value = 50
    task_num_classes = 3
    point_cloud_range = [-51.2, -51.2, -3.0, 51.2, 51.2, 5.0]

    stage2nd_num_points = 5

    num_cams_ = 5

    mvt4d_model_cfg = dict(
        type="MVT4D",
        bev_h=bev_h_value,
        bev_w=bev_w_value,
        pc_range=point_cloud_range,
        enable_temporal_fusion=False,
        backbone=dict(
            type="ResNet50V2",
            num_classes=1000,
            group_base=8,
            include_top=False,
            extend_features=False,
            bn_kwargs=dict(eps=1e-5, momentum=0.1),
        ),
        img_neck=dict(
            type="FPN",
            in_strides=[2, 4, 8, 16, 32],
            in_channels=[64, 256, 512, 1024, 2048],
            out_strides=[32],
            out_channels=[256],
            bn_kwargs=dict(eps=1e-5, momentum=0.1),
        ),
        bev_feat_encoder=dict(
            type="BevFeatEncoder",
            bev_h=bev_h_value,
            bev_w=bev_w_value,
            num_cams=num_cams_,
            bev_num_refs=4,
            num_layers=2,
            pc_range=point_cloud_range,
            transformerlayers=dict(
                type="BaseTransformerLayer",
                attn_cfgs=[
                    dict(
                        type="BevDeformableTemporalAttention",
                        num_levels=1,
                        embed_dims=256,
                        bev_h=bev_h_value,
                        bev_w=bev_w_value,
                    ),
                    dict(
                        type="BevSpatialCrossAtten",
                        pc_range=point_cloud_range,
                        num_cams=num_cams_,
                        deformable_attention=dict(
                            type="MSDeformableAttention3D",
                            embed_dims=256,
                            num_points=8,
                            num_levels=1,
                        ),
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
            positional_encoding=dict(
                type="LearnedPositionalEncoding",
                num_feats=128,
                row_num_embed=bev_h_value,
                col_num_embed=bev_w_value,
            ),
        ),
        pts_bbox_head=dict(
            type="DeformableHead",
            pc_range=point_cloud_range,
            num_query=900,
            num_classes=task_num_classes,
            bev_h=bev_h_value,
            bev_w=bev_w_value,
            with_box_refine=True,
            stage2nd_cfg=dict(num_points=stage2nd_num_points),
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
                    score_thr=0.00,
                    max_num=300,
                ),
            ),
        ),
    )

    bs = 1

    data = dict(
        img=torch.randn(
            bs, num_cams_, 3, 640 // 5, 960 // 5
        ).cuda(),  # // 5 to save gpu
        gt_labels_3d=torch.zeros(bs, 300, dtype=torch.int64).cuda(),
        gt_bboxes_3d=torch.randn(bs, 300, 7).cuda(),
        img_metas=dict(
            img_shape=torch.tensor(
                [
                    [
                        [640 // 5, 960 // 5],
                    ]
                    * num_cams_
                ]
                * bs
            ).cuda(),
            T_vcs2img=torch.randn(bs, num_cams_, 3, 4).cuda(),
            timestamp=torch.randn(bs).cuda(),
        ),
    )

    return mvt4d_model_cfg, data


def get_mvt4dv2_config_and_data():
    bev_h_value = 50
    bev_w_value = 50
    task_num_classes = 3
    point_cloud_range = [-51.2, -51.2, -3.0, 51.2, 51.2, 5.0]

    stage2nd_num_points = 5

    num_cams_ = 6

    mvt4dv2_model_cfg = dict(
        type="MVT4D",
        bev_h=bev_h_value,
        bev_w=bev_w_value,
        pc_range=point_cloud_range,
        enable_temporal_fusion=True,
        backbone=dict(
            type="ResNet50V2",
            num_classes=1000,
            group_base=8,
            include_top=False,
            extend_features=False,
            bn_kwargs=dict(eps=1e-5, momentum=0.1),
        ),
        img_neck=dict(
            type="FPN",
            in_strides=[2, 4, 8, 16, 32],
            in_channels=[64, 256, 512, 1024, 2048],
            out_strides=[32],
            out_channels=[256],
            bn_kwargs=dict(eps=1e-5, momentum=0.1),
        ),
        bev_feat_encoder=dict(
            type="BevFeatEncoder",
            bev_h=bev_h_value,
            bev_w=bev_w_value,
            num_cams=num_cams_,
            bev_num_refs=4,
            num_layers=2,
            pc_range=point_cloud_range,
            max_interval=600.0 / 1e3,
            temporal_layer=dict(
                type="BaseTransformerLayer",
                attn_cfgs=[
                    dict(
                        type="BevDeformableTemporalAttention",
                        dropout=0.1,
                        num_levels=1,
                        embed_dims=256,
                        bev_h=bev_h_value,
                        bev_w=bev_w_value,
                        qv_cat=False,
                    )
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
                    "ffn",
                    "norm",
                ),
            ),
            transformerlayers=dict(
                type="BaseTransformerLayer",
                attn_cfgs=[
                    dict(
                        type="MultiScaleDeformableAttention",
                        num_levels=1,
                        embed_dims=256,
                    ),
                    dict(
                        type="BevSpatialCrossAtten",
                        pc_range=point_cloud_range,
                        num_cams=num_cams_,
                        deformable_attention=dict(
                            type="MSDeformableAttention3D",
                            embed_dims=256,
                            num_points=8,
                            num_levels=3,
                        ),
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
            positional_encoding=dict(
                type="LearnedPositionalEncoding",
                num_feats=128,
                row_num_embed=bev_h_value,
                col_num_embed=bev_w_value,
            ),
        ),
        pts_bbox_head=dict(
            type="DeformableHead",
            pc_range=point_cloud_range,
            num_query=900,
            num_classes=task_num_classes,
            bev_h=bev_h_value,
            bev_w=bev_w_value,
            with_box_refine=True,
            stage2nd_cfg=dict(num_points=stage2nd_num_points),
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
            num_layers=3,
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
                    score_thr=0.00,
                    max_num=300,
                ),
            ),
        ),
    )

    bs = 2

    T_vcs2global = torch.randn(bs, 4, 4).cuda()
    data = dict(
        img=torch.randn(
            bs, num_cams_, 3, 640 // 5, 960 // 5
        ).cuda(),  # // 5 to save gpu
        gt_labels_3d=torch.zeros(bs, 300, dtype=torch.int64).cuda(),
        gt_bboxes_3d=torch.randn(bs, 300, 7).cuda(),
        img_metas=dict(
            img_shape=torch.tensor(
                [
                    [
                        [640 // 5, 960 // 5],
                    ]
                    * num_cams_
                ]
                * bs
            ).cuda(),
            T_vcs2img=torch.randn(bs, num_cams_, 3, 4).cuda(),
            T_vcs2global=T_vcs2global,
            T_global2vcs=torch.linalg.inv(T_vcs2global),
            timestamp=torch.randn(bs).cuda(),
        ),
    )

    return mvt4dv2_model_cfg, data


@pytest.mark.skipif(not _MMCV_IMPORTED, reason="mmcv is required for MVT4D")
def test_mvt4dv1():

    assert mmcv.__version__

    mvt4d_cfg, data = get_model_config_and_data()
    val_data = copy.deepcopy(data)
    model = build_from_registry(mvt4d_cfg)
    model = model.cuda()
    losses = model(data)

    assert isinstance(losses["loss_cls"], torch.Tensor)
    assert isinstance(losses["loss_bbox"], torch.Tensor)

    model.eval()
    with torch.no_grad():
        preds = model(val_data)

    assert isinstance(preds[0]["boxes_3d"], torch.Tensor)
    assert isinstance(preds[0]["scores_3d"], torch.Tensor)
    assert isinstance(preds[0]["labels_3d"], torch.Tensor)


@pytest.mark.skipif(not _MMCV_IMPORTED, reason="mmcv is required for MVT4D")
def test_mvt4dv2():

    assert mmcv.__version__

    mvt4dv2_cfg, data = get_mvt4dv2_config_and_data()
    val_data = copy.deepcopy(data)
    model = build_from_registry(mvt4dv2_cfg)
    model = model.cuda()
    losses = model(data)

    assert isinstance(losses["loss_cls"], torch.Tensor)
    assert isinstance(losses["loss_bbox"], torch.Tensor)

    model.eval()
    with torch.no_grad():
        preds = model(val_data)

    assert isinstance(preds[0]["boxes_3d"], torch.Tensor)
    assert isinstance(preds[0]["scores_3d"], torch.Tensor)
    assert isinstance(preds[0]["labels_3d"], torch.Tensor)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
