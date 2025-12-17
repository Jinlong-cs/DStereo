import pytest
import torch

from hat.registry import build_from_registry


def gen_bevformerviewtransformer_data():
    data = dict(
        prev_bev=torch.randn((1, 2500, 256)),
        hybird_ref_2d=torch.randn((2, 2500, 1, 2)),
        reference_points_cam=torch.randn((6, 1, 2500, 4, 2)),
        prev_bev_ref=torch.randn((1, 50, 50, 2)),
    )
    feats = torch.randn(6, 256, 15, 25)
    return [feats], data


@pytest.mark.serial_task
def test_bevformerviewtransformer():
    config = dict(
        type="BevFormerViewTransformer",
        bev_h=50,
        bev_w=50,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        num_points_in_pillar=4,
        embed_dims=256,
        queue_length=1,
        in_indices=(-1,),
        is_compile=True,
        positional_encoding=dict(
            type="LearnedPositionalEncoding",
            num_feats=128,
            row_num_embed=50,
            col_num_embed=50,
        ),
        encoder=dict(
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
        ),
    )
    model = build_from_registry(config)
    data = gen_bevformerviewtransformer_data()
    model(*data)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
