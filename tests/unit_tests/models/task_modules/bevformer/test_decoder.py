import pytest
import torch

from hat.registry import build_from_registry


def gen_bevformerdetdecoder_data():
    bev_embed = torch.randn(1, 2500, 256)
    return bev_embed


@pytest.mark.serial_task
def test_bevformerdetdecoder():
    config = dict(
        type="BEVFormerDetDecoder",
        bev_h=50,
        bev_w=50,
        num_query=900,
        embed_dims=256,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        is_compile=True,
        decoder=dict(
            type="DetectionTransformerDecoder",
            num_layers=6,
            return_intermediate=True,
            decoder_layer=dict(
                type="DetrTransformerDecoderLayer",
                crossattention=dict(
                    type="HorizonMSDeformableAttention",
                    embed_dims=256,
                    num_levels=1,
                    view_gird_out=6,
                    view_gird_in=4,
                    feats_size=[[50, 50]],
                ),
                dropout=0.1,
            ),
        ),
    )
    model = build_from_registry(config)
    data = gen_bevformerdetdecoder_data()
    model(data)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
