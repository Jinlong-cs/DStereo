import pytest
import torch

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test

_dim_ = 256
_pos_dim_ = _dim_ // 2
bev_h_ = 50
bev_w_ = 50
_num_levels_ = 1
num_classes = 10
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
config = dict(
    type="BevFormer",
    out_indices=(-1,),
    backbone=dict(
        type="ResNet50",
        num_classes=1000,
        bn_kwargs={},
        include_top=False,
    ),
    neck=dict(
        type="FPN",
        in_strides=[32],
        in_channels=[2048],
        out_strides=[32],
        out_channels=[_dim_],
        bn_kwargs=dict(eps=1e-5, momentum=0.1),
    ),
    view_transformer=dict(
        type="BevFormerViewTransformer",
        bev_h=bev_h_,
        bev_w=bev_w_,
        is_compile=True,
        pc_range=point_cloud_range,
        num_points_in_pillar=4,
        embed_dims=_dim_,
        queue_length=1,
        in_indices=(-1,),
        positional_encoding=dict(
            type="LearnedPositionalEncoding",
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        encoder=dict(
            type="BEVFormerEncoder",
            num_layers=3,
            return_intermediate=False,
            bev_h=bev_h_,
            bev_w=bev_w_,
            embed_dims=_dim_,
            encoder_layer=dict(
                type="BEVFormerEncoderLayer",
                selfattention=dict(
                    type="HorizonTemporalSelfAttention",
                    embed_dims=_dim_,
                    num_levels=1,
                    view_gird_in=1,
                    view_gird_out=100,
                    view_num=8,
                    feats_size=[[bev_w_, bev_h_]],
                ),
                crossattention=dict(
                    type="HorizonSpatialCrossAttention",
                    view_num=8,
                    deformable_attention=dict(
                        type="HorizonMSDeformableAttention3D",
                        embed_dims=_dim_,
                        num_points=8,
                        num_levels=_num_levels_,
                        view_gird_in=160,
                        view_gird_out=100,
                        feats_size=[[25, 15]],
                    ),
                    embed_dims=_dim_,
                ),
                dropout=0.1,
            ),
        ),
    ),
    bev_decoders=[
        dict(
            type="BEVFormerDetDecoder",
            bev_h=bev_h_,
            bev_w=bev_w_,
            num_query=900,
            embed_dims=_dim_,
            pc_range=point_cloud_range,
            is_compile=True,
            decoder=dict(
                type="DetectionTransformerDecoder",
                num_layers=6,
                return_intermediate=True,
                decoder_layer=dict(
                    type="DetrTransformerDecoderLayer",
                    crossattention=dict(
                        type="HorizonMSDeformableAttention",
                        embed_dims=_dim_,
                        num_levels=1,
                        view_gird_out=6,
                        view_gird_in=4,
                        feats_size=[[bev_w_, bev_h_]],
                    ),
                    dropout=0.1,
                ),
            ),
        ),
    ],
)


def gen_data():
    data = dict(
        img=torch.randn((6, 3, 480, 800)),
        prev_bev=torch.randn((1, 2500, 256)),
        hybird_ref_2d=torch.randn((2, 2500, 1, 2)),
        reference_points_cam=torch.randn((6, 1, 2500, 4, 2)),
        prev_bev_ref=torch.randn((1, 50, 50, 2)),
    )
    return data


@pytest.mark.serial_task
def test_bevformer():
    detector = build_from_registry(config)
    data = gen_data()
    detector(data)
    qat_test(detector, data, with_quantized=False)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
