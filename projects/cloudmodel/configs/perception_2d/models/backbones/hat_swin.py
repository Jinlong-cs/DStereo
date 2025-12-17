import os

from hatbc.workflow.trace import GraphTracer

from hat.models.backbones.swin_transformer import SwinTransformer
from projects.cloudmodel.configs.perception_2d.common import bucket_root

# *_w12 means window_size = 12, while others use window_size = 7
checkpoints = {
    "tiny": os.path.join(
        bucket_root,
        "adas/big_model/model_zoo/swin_tiny_patch4_window7_224_22k.pth",
    ),
    "small": os.path.join(
        bucket_root,
        "adas/big_model/model_zoo/swin_small_patch4_window7_224_22k.pth",
    ),
    "base": os.path.join(
        bucket_root,
        "adas/big_model/model_zoo/swin_base_patch4_window7_224_22k.pth",
    ),
    "base_w12": os.path.join(
        bucket_root,
        "adas/big_model/model_zoo/swin_base_patch4_window12_384_22k.pth",
    ),
    "large": os.path.join(
        bucket_root,
        "adas/big_model/model_zoo/swin_large_patch4_window7_224_22k.pth",
    ),
    "large_w12": os.path.join(
        bucket_root,
        "adas/big_model/model_zoo/swin_large_patch4_window12_384_22k.pth",
    ),
}


stride2channels_dict = {
    "tiny": {4: 96, 8: 192, 16: 384, 32: 768},
    "small": {4: 96, 8: 192, 16: 384, 32: 768},
    "base": {4: 128, 8: 256, 16: 512, 32: 1024},
    "base_w12": {4: 128, 8: 256, 16: 512, 32: 1024},
    "large": {4: 192, 8: 384, 16: 768, 32: 1536},
    "large_w12": {4: 192, 8: 384, 16: 768, 32: 1536},
}
embed_dims = {
    "tiny": 96,
    "small": 96,
    "base": 128,
    "base_w12": 128,
    "large": 192,
    "large_w12": 192,
}
depths = {
    "tiny": [2, 2, 6, 2],
    "small": [2, 2, 18, 2],
    "base": [2, 2, 18, 2],
    "base_w12": [2, 2, 18, 2],
    "large": [2, 2, 18, 2],
    "large_w12": [2, 2, 18, 2],
}
num_heads = {
    "tiny": [3, 6, 12, 24],
    "small": [3, 6, 12, 24],
    "base": [4, 8, 16, 32],
    "base_w12": [4, 8, 16, 32],
    "large": [6, 12, 24, 48],
    "large_w12": [6, 12, 24, 48],
}
window_sizes = {
    "tiny": 7,
    "small": 7,
    "base": 7,
    "base_w12": 12,
    "large": 7,
    "large_w12": 12,
}


def build_backbone_hat_swin(arch="large", node_name="backbone"):

    stride2channels = stride2channels_dict[arch]

    hat_swin_backbone = dict(
        type="SwinTransformer",
        depth_list=depths[arch],
        num_heads=num_heads[arch],
        embedding_dims=embed_dims[arch],
        window_size=window_sizes[arch],
        drop_path_ratio=0.2,
        include_top=False,
        flat_output=False,
        pretrained_path=checkpoints[arch],
        node_name=node_name,
    )

    return dict(
        backbone=hat_swin_backbone,
        stride2channels=stride2channels,
        checkpoint=None,
    )


# Register network in GraphTracer
GraphTracer.register_basic_types(SwinTransformer)


if __name__ == "__main__":
    from hat.registry import build_from_registry

    hat_swin_backbone = build_backbone_hat_swin("small")
    backbone = build_from_registry(hat_swin_backbone["backbone"])
    import torch

    x = torch.rand((1, 3, 480, 480))
    feats = backbone(x)

    print("End")
