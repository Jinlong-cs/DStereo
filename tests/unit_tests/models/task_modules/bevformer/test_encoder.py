import pytest
import torch

from hat.registry import build_from_registry


def gen_bevformerencoder_data():
    bev_query = torch.randn(2500, 2, 256)
    mlvl_feats = [torch.randn(2, 6, 256, 15, 25)]
    bev_pos = torch.randn(2500, 2, 256)
    prev_bev = torch.randn(2, 2500, 256)
    hybird_ref_2d = torch.randn(4, 2500, 1, 2)
    reference_points_cam = torch.randn(6, 2, 2500, 4, 2)
    return (
        bev_query,
        mlvl_feats,
        bev_pos,
        prev_bev,
        hybird_ref_2d,
        reference_points_cam,
    )


@pytest.mark.serial_task
def test_bevformerencoder():
    config = dict(
        type="BEVFormerEncoder",
        num_layers=3,
        return_intermediate=False,
        bev_h=50,
        bev_w=50,
        embed_dims=256,
        encoder_layer=dict(
            type="BEVFormerEncoderLayer",
            selfattention=dict(
                type="HorizonTemporalSelfAttention",
                embed_dims=256,
                num_levels=1,
                view_gird_in=1,
                view_gird_out=100,
                view_num=8,
                feats_size=[[50, 50]],
            ),
            crossattention=dict(
                type="HorizonSpatialCrossAttention",
                view_num=8,
                deformable_attention=dict(
                    type="HorizonMSDeformableAttention3D",
                    embed_dims=256,
                    num_points=8,
                    num_levels=1,
                    view_gird_in=160,
                    view_gird_out=100,
                    feats_size=[[25, 15]],
                ),
                embed_dims=256,
            ),
            dropout=0.1,
        ),
    )
    model = build_from_registry(config)
    data = gen_bevformerencoder_data()
    model(*data)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
