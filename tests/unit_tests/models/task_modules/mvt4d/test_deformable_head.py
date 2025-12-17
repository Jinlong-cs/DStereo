import pytest
import torch

from hat.registry import build_from_registry

try:
    import mmcv

    _MMCV_IMPORTED = True
except ImportError:
    _MMCV_IMPORTED = False


def get_head_config_and_x():
    bev_h_value = 256
    bev_w_value = 256
    task_num_classes = 3
    point_cloud_range = [-51.2, -51.2, -3.0, 51.2, 51.2, 5.0]

    stage2nd_num_points = 5

    pts_bbox_head_cfg = dict(
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
    )

    x = dict(
        bev_feat=torch.randn(bev_h_value * bev_w_value, 1, 256),
        img_metas=dict(
            img_shape=torch.tensor([[640, 960]]),
            T_vcs2img=torch.randn(1, 5, 3, 4),
            timestamp=torch.randn(1),
        ),
        img_feats=None,
    )

    return pts_bbox_head_cfg, x


@pytest.mark.skipif(
    not _MMCV_IMPORTED, reason="mmcv is required for DeformableHead"
)
def test_deformable_head():

    assert mmcv.__version__

    # stage2nd head from bev feat
    head_cfg, x = get_head_config_and_x()
    head = build_from_registry(head_cfg)

    pre_img_metas = None
    y = head(x["bev_feat"], x["img_metas"], pre_img_metas, x["img_feats"])
    assert isinstance(y, dict)
    assert y["all_cls_scores"].shape == (6, 1, 900, 3)
    assert y["all_bbox_preds"].shape == (6, 1, 900, 8)
    assert y["refined_bbox_preds"].shape == (6, 1, 900, 8)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
