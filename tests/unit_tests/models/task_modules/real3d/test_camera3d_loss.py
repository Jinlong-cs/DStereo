from collections import OrderedDict
from typing import Dict

import torch

from hat.registry import build_from_registry

use_simplified_rot = True
output_head_with_2d_wh = True
max_depth = 50
resize_hw = (640, 1024)
num_classes = 1
rot_channel = 2
real3d_bias = True
feat_channels = 16
is_loss_custom = False

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

head_cfg = dict(
    type="Camera3DHead",
    output_cfg=output_cfg,
    bn_kwargs=dict(eps=1e-5, momentum=0.1),
    in_strides=[4],
    in_channels=[feat_channels],
    out_stride=4,
    use_bias=real3d_bias,
)

loss_cfg = dict(
    type="Camera3DLoss",
    hm_loss=dict(type="HMFocalLoss"),
    box2d_wh_loss=dict(type="HML1Loss", heatmap_type="weighted"),
    dimensions_loss=dict(type="HML1Loss", heatmap_type="weighted"),
    location_offset_loss=dict(
        type="HML1Loss",
        heatmap_type="weighted",
    ),
    depth_loss=dict(type="HML1Loss", heatmap_type="weighted"),
    loss_weights=dict(
        heatmap=0.5,
        box2d_wh=0.01,
        depth=0.5,
        dimensions=0.5,
        rotation=8,
        location_offset=0.5,
    ),
    output_head_with_2d_wh=output_head_with_2d_wh,
    max_depth=max_depth,
)

x = [
    torch.randn(1, feat_channels, 160, 256),
]
# 防止dataset没有传入loss_custom_weight报错，这里可以给一个初始值
loss_custom_weight_init = {
    "dense_heatmap": 1,
    "dense_rotation": 1,
    "dense_depth": 1,
    "dense_dimensions": 1,
    "dense_location_offset": 1,
    "dense_box2d_wh": 1,
    "sparse_center_2d": 1,
    "sparse_offset_2d": 1,
    "sparse_offset_3d": 1,
    "sparse_depth": 1,
    "sparse_depth_u": 1,  # if undistort_depth_uv=True
    "sparse_depth_v": 1,  # if undistort_depth_uv=True
    "sparse_dim": 1,
    "sparse_rot": 1,
    "sparse_iou": 1,
}

inputs = dict(
    heatmap=torch.zeros((1, 1, resize_hw[0] // 4, resize_hw[1] // 4)),
    box2d_wh=torch.zeros((1, 2, resize_hw[0] // 4, resize_hw[1] // 4)),
    dimensions=torch.zeros((1, 3, resize_hw[0] // 4, resize_hw[1] // 4)),
    location_offset=torch.zeros((1, 2, resize_hw[0] // 4, resize_hw[1] // 4)),
    depth=torch.zeros((1, 1, resize_hw[0] // 4, resize_hw[1] // 4)),
    heatmap_weight=torch.zeros((1, 1, resize_hw[0] // 4, resize_hw[1] // 4)),
    ignore_mask=torch.zeros((1, 1, resize_hw[0] // 4, resize_hw[1] // 4)),
    index=torch.zeros((1, 100)),
    index_mask=torch.zeros((1, 100)),
    location=torch.zeros((1, 100, 3)),
    rotation_y=torch.zeros((1, 100, 1)),
    dimensions_=torch.zeros((1, 100, 3)),
)
if is_loss_custom:
    inputs["loss_custom_weight"] = loss_custom_weight_init


def test_camera3d_loss():
    head = build_from_registry(head_cfg)
    camera3d_loss = build_from_registry(loss_cfg)
    pred = head(x)
    output = camera3d_loss(
        pred,
        inputs,
    )
    assert isinstance(output, Dict)
