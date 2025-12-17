import torch

from hat.models.task_modules.bev.bev_parkingrod_head import (
    ANCBEVParkingRodHead,
)


def test_bevparkingrod_head():
    num_parkingrod_class = 3
    bev_stage2_feats_name = "bev_stage2_feats"
    downsampling_stride = 2
    in_strides = [2, 4, 8, 16, 32]
    in_channels = 48

    parkingrod_head = ANCBEVParkingRodHead(
        num_parkingrod_class=num_parkingrod_class,
        feature_name=bev_stage2_feats_name,
        in_strides=in_strides,
        out_strides=downsampling_stride * 2,
        in_channels=in_channels,
        sep_conv=False,
        stack=1,
        use_bias=True,
    )
    bev_stage2_feats = [
        torch.randn((2, 48, 192, 128)),
        torch.randn((2, 48, 96, 64)),
        torch.randn((2, 96, 48, 32)),
        torch.randn((2, 192, 24, 26)),
        torch.randn((2, 192, 12, 8)),
    ]
    data = {}
    data[bev_stage2_feats_name] = [bev_stage2_feats]
    (
        classification_pred,
        endpoint_offset_pred,
    ) = parkingrod_head(data)

    assert classification_pred.shape == (2, 1, 96, 64)
    assert endpoint_offset_pred.shape == (2, 4, 96, 64)
