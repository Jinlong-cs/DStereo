from collections import OrderedDict

import torch

from hat.models.backbones.vargnetv2 import get_vargnetv2_stride2channels
from hat.registry import build_from_registry


def test_bev_head():
    bev_bs, views = 2, 8
    ipm_output_size = (512, 512)
    grid_quant_scale = 0.03125
    config = dict(
        type="ANCBEVFusionModule",
        grid_quant_scale=grid_quant_scale,
        random_rotation_cfg=dict(
            type="RandomRotation",
            angles=(0, 90, 180, 270),
            height=ipm_output_size[0],
            width=ipm_output_size[1],
            grid_quant_scale=grid_quant_scale,
            mode="bilinear",
            padding_mode="zeros",
            use_horizon_grid_sample=True,
        ),
        drop_view_prob=0.2,
        views=views,
        ipm_output_size=ipm_output_size,
    )

    bev_fusion = build_from_registry(config)
    input_w, input_h = 960, 512
    feats = [
        torch.randn((bev_bs * views, 8, input_h // 4, input_w // 4)),
    ]

    meta = dict(meta_info={})
    meta["meta_info"]["homography"] = torch.randn((bev_bs, views, 3, 3))
    meta["meta_info"]["homo_offset"] = (
        torch.randn((views, *ipm_output_size, 2)),
    )

    out = bev_fusion(feats, meta)
    assert out[1].shape == (bev_bs, views, *ipm_output_size)


def test_bev3d_head():
    num_classes = 3
    bev_stage2_feats_name = "bev_stage2_feats"
    stage2_stride2channels = get_vargnetv2_stride2channels(0.75)
    bn_kwargs = dict(eps=1e-5, momentum=0.1)
    in_strides = [2, 4, 8, 16, 32]
    out_strides = [2]
    feat_channel = stage2_stride2channels[out_strides[0]]
    config = dict(
        type="ANCBEV3DHead",
        feature_name=bev_stage2_feats_name,
        in_strides=in_strides,
        out_strides=out_strides,
        in_channels=feat_channel,
        forward_frame_idx=0,
        head_channels=OrderedDict(
            bev3d_hm=num_classes,
            bev3d_dim=3,  # h, w, l
            bev3d_rot=2,  # cos, sin
            bev3d_ct_offset=2,
            bev3d_loc_z=1,  # vcs z axis value
        ),
        last_conv_kernel_size=1,
        use_bias=False,
        bn_kwargs=bn_kwargs,
        dw_with_relu=True,
        pw_with_relu=False,
        factor=2,
        group_base=8,
    )
    bev3d_head = build_from_registry(config)
    bev_bs = 2
    input_bev_size = (512, 512)
    bev_stage2_feats = [
        torch.randn(
            (
                bev_bs,
                feat_channel,
                input_bev_size[0] // 2,
                input_bev_size[1] // 2,
            )
        ),
        torch.randn(
            (
                bev_bs,
                feat_channel,
                input_bev_size[0] // 4,
                input_bev_size[1] // 4,
            )
        ),
        torch.randn(
            (
                bev_bs,
                feat_channel * 2,
                input_bev_size[0] // 8,
                input_bev_size[1] // 8,
            )
        ),
        torch.randn(
            (
                bev_bs,
                feat_channel * 4,
                input_bev_size[0] // 16,
                input_bev_size[1] // 16,
            )
        ),
        torch.randn(
            (
                bev_bs,
                feat_channel * 8,
                input_bev_size[0] // 32,
                input_bev_size[1] // 32,
            )
        ),
    ]
    data = dict()
    data[bev_stage2_feats_name] = [bev_stage2_feats]

    out = bev3d_head(data)
    assert isinstance(out, dict)
    assert out["bev3d_hm"].shape == (2, 3, 256, 256)
    assert out["bev3d_dim"].shape == (2, 3, 256, 256)
    assert out["bev3d_rot"].shape == (2, 2, 256, 256)
    assert out["bev3d_loc_z"].shape == (2, 1, 256, 256)
    assert out["bev3d_ct_offset"].shape == (2, 2, 256, 256)
