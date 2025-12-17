import numpy as np
import pytest
import torch

try:
    from mmcv.cnn.bricks.transformer import MultiheadAttention
except ImportError:
    MultiheadAttention = None

from hat.registry import build_from_registry


@pytest.mark.parametrize(
    ["mutl_frame_sample", "recurrent_fusion"],
    [
        pytest.param(False, False),
        pytest.param(True, False),
        pytest.param(False, True),
        pytest.param(True, True),
    ],
)
def test_sparse4d_head(mutl_frame_sample, recurrent_fusion):
    if MultiheadAttention is None:
        return

    embed_dims = 20
    num_anchor = 30
    num_groups = 4
    num_levels = 4
    num_classes = 5
    num_cams = 6
    num_decoder = 4
    state_dims = 11
    anchor = np.random.uniform(size=(num_anchor, state_dims))
    if recurrent_fusion:
        num_temp_instances = 10
        num_single_frame_decoder = 1
    else:
        num_temp_instances = 0
        num_single_frame_decoder = -1

    if mutl_frame_sample:
        max_queue_length = 2
    else:
        max_queue_length = -1
    cfg = dict(
        type="Sparse4DHead",
        cls_threshold_to_reg=0.05,
        instance_bank=dict(
            type="InstanceBank",
            num_anchor=900,
            embed_dims=embed_dims,
            anchor=anchor,
            anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
            num_temp_instances=num_temp_instances,
            confidence_decay=0.6,
            max_queue_length=max_queue_length,
        ),
        anchor_encoder=dict(
            type="SparseBox3DEncoder",
            embed_dims=embed_dims,
            vel_dims=3,
        ),
        num_single_frame_decoder=num_single_frame_decoder,
        operation_order=[
            "deformable",
            "ffn",
            "norm",
            "refine",
        ]
        + [
            "temp_interaction",
            "interaction",
            "norm",
            "deformable",
            "ffn",
            "norm",
            "refine",
        ]
        * (num_decoder - 1),
        temp_instance_interaction=dict(
            type=MultiheadAttention,
            embed_dims=embed_dims,
            num_heads=num_groups,
            batch_first=True,
            dropout=0.1,
        ),
        instance_interaction=dict(
            type=MultiheadAttention,
            embed_dims=embed_dims,
            num_heads=num_groups,
            batch_first=True,
            dropout=0.1,
        ),
        norm_layer=dict(type=torch.nn.LayerNorm, normalized_shape=embed_dims),
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
            use_deformable_func=False,
            embed_dims=embed_dims,
            num_groups=num_groups,
            num_levels=num_levels,
            num_cams=6,
            attn_drop=0.15,
            residual_mode="cat",
            use_camera_embed=True,
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
            refine_yaw=True,
        ),
        target=dict(
            type="SparseBox3DTarget",
            cls_weight=2.0,
            box_weight=0.25,
            reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 5,
            num_dn_groups=3,
            add_neg_dn=True,
            max_dn_gt=30,
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
        reg_weights=[2.0] * 3 + [1.0] * 8,
    )
    head = build_from_registry(cfg)
    head.init_weights()
    bs = 2
    num_gt = 10
    feature_maps = [
        torch.randn(bs, num_cams, embed_dims, 32 // (i + 1), 32 // (i + 1))
        for i in range(num_levels)
    ]
    metas = dict(
        timestamp=torch.ones(bs),
        projection_mat=torch.randn(bs, num_cams, 4, 4),
        image_wh=torch.ones(bs, num_cams, 2),
        img_metas=dict(
            T_global=torch.randn(bs, 4, 4),
            T_global_inv=torch.randn(bs, 4, 4),
        ),
        gt_labels_3d=torch.ones(bs, num_gt).to(dtype=torch.long),
        gt_bboxes_3d=torch.ones(bs, num_gt, state_dims - 1),
    )
    input = dict(
        feature_maps=feature_maps,
        metas=metas,
        feature_queue=[feature_maps] * max_queue_length
        if max_queue_length > 0
        else None,
        meta_queue=[metas] * max_queue_length
        if max_queue_length > 0
        else None,
    )
    head.train()
    for _ in range(5):
        model_outs = head(**input)
        classification = model_outs["classification"]
        prediction = model_outs["prediction"]
        assert len(classification) == num_decoder
        assert len(prediction) == num_decoder
        for cls, pred in zip(classification, prediction):
            assert cls.shape == (bs, num_anchor, num_classes)
            assert pred.shape == (bs, num_anchor, state_dims)

        loss = head.loss(model_outs, metas)
        for i in range(num_decoder):
            assert f"loss_cls_{i}" in loss
            assert f"loss_reg_{i}" in loss
            if head.target.num_dn_groups > 0:
                assert f"dn_loss_cls_{i}" in loss
                assert f"dn_loss_reg_{i}" in loss

    head.eval()
    for _ in range(5):
        model_outs = head(**input)
        classification = model_outs["classification"]
        prediction = model_outs["prediction"]
        assert len(classification) == num_decoder
        assert len(prediction) == num_decoder
        for cls, pred in zip(classification, prediction):
            assert cls is None or cls.shape == (bs, num_anchor, num_classes)
            assert pred.shape == (bs, num_anchor, state_dims)
