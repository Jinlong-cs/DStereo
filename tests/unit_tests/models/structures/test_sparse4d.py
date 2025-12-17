import copy

import numpy as np
import pytest
import torch

try:
    from mmcv.cnn.bricks.transformer import MultiheadAttention
except ImportError:
    MultiheadAttention = None

from hat.registry import build_from_registry


@pytest.mark.skipif(
    MultiheadAttention is None, reason="mmcv.MultiheadAttention is needed"
)
@pytest.mark.parametrize(
    ["mutl_frame_sample", "recurrent_fusion"],
    [
        pytest.param(False, False),
        pytest.param(True, False),
        pytest.param(False, True),
        pytest.param(True, True),
    ],
)
def test_sparse4d(mutl_frame_sample, recurrent_fusion):
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
        type="Sparse4D",
        use_deformable_func=False,
        backbone=dict(
            type="ResNet50",
            num_classes=None,
            include_top=False,
            bn_kwargs=dict(eps=1e-5, momentum=0.1),
        ),
        neck=dict(
            type="FPN",
            in_strides=[2, 4, 8, 16, 32],
            in_channels=[64, 256, 512, 1024, 2048],
            out_strides=[4, 8, 16, 32],
            out_channels=[embed_dims] * 4,
        ),
        head=dict(
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
            decoder=dict(type="SparseBox3DDecoder", num_output=num_anchor - 1),
            reg_weights=[2.0] * 3 + [1.0] * 8,
        ),
    )
    sparse4d = build_from_registry(cfg)
    bs = 2
    num_gt = 10

    def get_data(img=None):
        data = dict(
            timestamp=torch.ones(bs),
            projection_mat=torch.randn(bs, num_cams, 4, 4),
            image_wh=torch.ones(bs, num_cams, 2),
            img_metas=dict(
                T_global=torch.randn(bs, 4, 4),
                T_global_inv=torch.randn(bs, 4, 4),
            ),
            gt_labels_3d=torch.ones(bs, num_gt).to(dtype=torch.long),
            gt_bboxes_3d=torch.randn(bs, num_gt, state_dims - 1),
        )
        if img is not None:
            data["img"] = img
        else:
            data["img"] = torch.randn(bs, num_cams, 3, 64, 64)
        if max_queue_length > 0:
            data["data_queue"] = [
                copy.deepcopy(data) for _ in range(max_queue_length)
            ]
        return data

    sparse4d.train()
    for _ in range(1):
        loss = sparse4d(get_data())
        for i in range(num_decoder):
            assert f"loss_cls_{i}" in loss
            assert f"loss_reg_{i}" in loss

    sparse4d.eval()
    for _ in range(1):
        results = sparse4d(get_data())
        assert len(results) == bs
        for ret in results:
            assert "boxes_3d" in ret
            assert "scores_3d" in ret
            assert "labels_3d" in ret

    cfg_multi_backbone = copy.deepcopy(cfg)
    cfg_multi_backbone["backbone"] = [
        copy.deepcopy(cfg["backbone"]),
        copy.deepcopy(cfg["backbone"]),
    ]
    cfg_multi_backbone["neck"] = [
        copy.deepcopy(cfg["neck"]),
        copy.deepcopy(cfg["neck"]),
    ]
    sparse4d = build_from_registry(cfg_multi_backbone)

    sparse4d.train()
    for _ in range(1):
        img = [
            torch.randn(bs, num_cams // 2, 3, 64, 64),
            torch.randn(bs, num_cams - num_cams // 2, 3, 96, 96),
        ]
        data = get_data(img)
        loss = sparse4d(data)
        for i in range(num_decoder):
            assert f"loss_cls_{i}" in loss
            assert f"loss_reg_{i}" in loss

    sparse4d.eval()
    for _ in range(1):
        img = [
            torch.randn(bs, num_cams // 2, 3, 64, 64),
            torch.randn(bs, num_cams - num_cams // 2, 3, 96, 96),
        ]
        data = get_data(img)
        results = sparse4d(data)
        assert len(results) == bs
        for ret in results:
            assert "boxes_3d" in ret
            assert "scores_3d" in ret
            assert "labels_3d" in ret


if __name__ == "__main__":
    pytest.main(["-s", __file__])
