from collections import OrderedDict

import torch

from hat.registry import build_from_registry

num_classes = 1
rot_channel = 2
outputs = OrderedDict(
    hm=num_classes,
    dep=1,
    rot=rot_channel,
    dim=3,
    loc_offset=2,
)
outputs["wh"] = 2

outputs_prefix = OrderedDict(
    hm="vehicle_hm",
    dep="dep",
    rot="rot",
    dim="dim",
    loc_offset="loc_offset",
    wh="wh",
)

output_cfg = {
    out: {
        "out_channels": ch,
        "out_conv_channels": 32,
        "prefix": outputs_prefix[out],
    }
    for out, ch in outputs.items()
}
feat_channels = 16


def get_head_config_and_x():
    real3d_bias = True
    cfg = dict(
        type="Camera3DHead",
        output_cfg=output_cfg,
        bn_kwargs=dict(eps=1e-5, momentum=0.1),
        in_strides=[4],
        in_channels=[feat_channels],
        out_stride=4,
        use_bias=real3d_bias,
    )
    x = [
        torch.randn(1, feat_channels, 128, 128),
        torch.randn(1, feat_channels, 64, 64),
        torch.randn(1, feat_channels, 32, 32),
        torch.randn(1, feat_channels, 16, 16),
        torch.randn(1, feat_channels, 8, 8),
    ]
    return cfg, x


def test_Camera3DHead():
    head_cfg, x = get_head_config_and_x()
    head = build_from_registry(head_cfg)
    y = head(x)

    assert isinstance(y, OrderedDict)
    assert y.keys() == outputs.keys()
    for k, v in outputs.items():
        # [hm, dep, rot, dim, loc_offset, wh]
        assert y[k].shape == (1, v, 128, 128)
