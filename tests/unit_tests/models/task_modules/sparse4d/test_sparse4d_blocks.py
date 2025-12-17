import pytest
import torch

from hat.models.task_modules.sparse4d import (
    AsymmetricFFN,
    DeformableFeatureAggregation,
    DenseDepthNet,
    LinearFusionModule,
)
from hat.registry import build_from_registry


@pytest.mark.parametrize(
    ["enable_temporal_weight", "shape"],
    [
        pytest.param(True, (10,)),
        pytest.param(True, (2, 5)),
        pytest.param(False, (10,)),
        pytest.param(False, (2, 5)),
    ],
)
def test_linear_fusion_module(enable_temporal_weight, shape):
    embed_dims = 256
    fusion_module = LinearFusionModule(
        embed_dims=embed_dims, enable_temporal_weight=enable_temporal_weight
    )

    feature_1 = torch.randn((*shape, embed_dims))
    feature_2 = torch.randn((*shape, embed_dims))
    time_interval = torch.randn(shape[0])

    feature = fusion_module(feature_1, feature_2, time_interval)
    assert feature.shape == (*shape, embed_dims)


def test_deformable_feature_aggregation():
    embed_dims = 256
    num_levels = 3
    num_cams = 6
    cfg = dict(
        type=DeformableFeatureAggregation,
        embed_dims=embed_dims,
        num_groups=8,
        num_levels=num_levels,
        num_cams=num_cams,
        proj_drop=0.1,
        attn_drop=0.15,
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
    )
    deformable_model = build_from_registry(cfg)
    batch_size = 2
    num_anchor = 100

    feature_maps = [
        torch.randn(
            (batch_size, num_cams, embed_dims, 64 // (i + 1), 64 // (i + 1))
        )
        for i in range(num_levels)
    ]
    metas = dict(
        timestamp=torch.randn(batch_size),
        projection_mat=torch.eye(4)[None, None].tile(
            batch_size, num_cams, 1, 1
        ),
        image_wh=torch.tensor([[[64, 64]] * num_cams] * batch_size),
        img_metas=dict(
            T_global=torch.eye(4)[None].tile(batch_size, 1, 1),
            T_global_inv=torch.eye(4)[None].tile(batch_size, 1, 1),
        ),
    )
    queue_length = 4
    input = dict(
        instance_feature=torch.randn((batch_size, num_anchor, embed_dims)),
        anchor=torch.randn((batch_size, num_anchor, 10)),
        anchor_embed=torch.randn((batch_size, num_anchor, embed_dims)),
        feature_maps=feature_maps,
        metas=metas,
        feature_queue=[feature_maps] * queue_length,
        meta_queue=[metas] * queue_length,
    )

    deformable_model.train()
    output = deformable_model(**input)
    assert output.shape == (batch_size, num_anchor, embed_dims)

    deformable_model.eval()
    output = deformable_model(**input)
    assert output.shape == (batch_size, num_anchor, embed_dims)


def test_dense_depth_net():
    num_depth_layers = 3
    batch_size = 2
    embed_dims = 256
    num_cams = 6

    depth_net = DenseDepthNet(
        embed_dims=embed_dims,
        num_depth_layers=num_depth_layers,
    )
    feature_maps = [
        torch.randn(
            (batch_size, num_cams, embed_dims, 64 // (i + 1), 64 // (i + 1))
        )
        for i in range(num_depth_layers)
    ]
    focal = torch.randn(batch_size, num_cams)
    depths = depth_net(feature_maps, focal)
    for i, depth in enumerate(depths):
        assert depth.shape == (
            batch_size * num_cams,
            1,
            64 // (i + 1),
            64 // (i + 1),
        )
    gt_depths = [
        torch.randn((batch_size, num_cams, 1, 64 // (i + 1), 64 // (i + 1)))
        for i in range(num_depth_layers)
    ]
    loss = depth_net.loss(depths, gt_depths)
    assert len(loss.shape) == 0


@pytest.mark.parametrize(
    ["in_channels", "embed_dims"],
    [
        pytest.param(20, 20),
        pytest.param(40, 20),
        pytest.param(20, 40),
    ],
)
def test_asymmetric_ffn(in_channels, embed_dims):
    cfg = dict(
        type=AsymmetricFFN,
        activate=dict(type=torch.nn.ReLU),
        pre_norm=True,
        in_channels=in_channels,
        embed_dims=embed_dims,
        add_identity=True,
    )
    ffn = build_from_registry(cfg)

    input = torch.randn((10, in_channels))
    output = ffn(input)
    assert output.shape == (10, embed_dims)

    output = ffn(input, identity=torch.randn((10, in_channels)))
    assert output.shape == (10, embed_dims)
