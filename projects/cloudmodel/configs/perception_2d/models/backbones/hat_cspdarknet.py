import os

from hatbc.workflow.trace import GraphTracer

from hat.models.backbones.csp_darknet import CSPDarknet
from projects.cloudmodel.configs.perception_2d.common import bucket_root

checkpoints = {
    "cspdarknet_nano": os.path.join(
        bucket_root, "adas/big_model/model_zoo/cspdarknet_nano.pth.tar"
    ),
    "cspdarknet_tiny": os.path.join(
        bucket_root, "adas/big_model/model_zoo/cspdarknet_tiny.pth.tar"
    ),
    "cspdarknet_s": os.path.join(
        bucket_root, "adas/big_model/model_zoo/cspdarknet_s.pth.tar"
    ),
    "cspdarknet_m": os.path.join(
        bucket_root, "adas/big_model/model_zoo/cspdarknet_m.pth.tar"
    ),
    "cspdarknet_l": os.path.join(
        bucket_root, "adas/big_model/model_zoo/cspdarknet_l.pth.tar"
    ),
    "cspdarknet_x": os.path.join(
        bucket_root, "adas/big_model/model_zoo/cspdarknet_x.pth.tar"
    ),
}

stride2channels_dict = {
    "cspdarknet_nano": {4: 32, 8: 64, 16: 128, 32: 256},
    "cspdarknet_tiny": {4: 48, 8: 96, 16: 192, 32: 384},
    "cspdarknet_s": {4: 64, 8: 128, 16: 256, 32: 512},
    "cspdarknet_m": {4: 96, 8: 192, 16: 384, 32: 768},
    "cspdarknet_l": {4: 128, 8: 256, 16: 512, 32: 1024},
    "cspdarknet_x": {4: 160, 8: 320, 16: 640, 32: 1280},
}

depth_width_params_dict = {
    "cspdarknet_nano": [0.33, 0.25],
    "cspdarknet_tiny": [0.33, 0.375],
    "cspdarknet_s": [0.33, 0.5],
    "cspdarknet_m": [0.57, 0.75],
    "cspdarknet_l": [1.0, 1.0],
    "cspdarknet_x": [1.33, 1.25],
}


def build_backbone_hat_cspdarknet(arch="cspdarknet_m", node_name="backbone"):

    checkpoint = checkpoints[arch]
    stride2channels = stride2channels_dict[arch]

    hat_cspdarknet_backbone = dict(
        type="CSPDarknet",
        dep_mul=depth_width_params_dict[arch][0],
        wid_mul=depth_width_params_dict[arch][1],
        node_name=node_name,
    )

    return dict(
        backbone=hat_cspdarknet_backbone,
        stride2channels=stride2channels,
        checkpoint=checkpoint,
    )


# Register network in GraphTracer
GraphTracer.register_basic_types(CSPDarknet)
