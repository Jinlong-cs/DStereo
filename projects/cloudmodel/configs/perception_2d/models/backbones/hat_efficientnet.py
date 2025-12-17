import os

from hatbc.workflow.trace import GraphTracer

from hat.models.backbones.efficientnet import efficientnet
from projects.cloudmodel.configs.perception_2d.common import bucket_root

checkpoints = {
    "b0": os.path.join(
        bucket_root,
        "adas/big_model/model_zoo/hat_noisy_student_efficientnet-b0.pth",
    ),
    "b1": os.path.join(
        bucket_root,
        "adas/big_model/model_zoo/hat_noisy_student_efficientnet-b1.pth",
    ),
    "b2": os.path.join(
        bucket_root,
        "adas/big_model/model_zoo/hat_noisy_student_efficientnet-b2.pth",
    ),
    "b3": os.path.join(
        bucket_root,
        "adas/big_model/model_zoo/hat_noisy_student_efficientnet-b3.pth",
    ),
    "b4": os.path.join(
        bucket_root,
        "adas/big_model/model_zoo/hat_noisy_student_efficientnet-b4.pth",
    ),
    "b5": os.path.join(
        bucket_root,
        "adas/big_model/model_zoo/hat_noisy_student_efficientnet-b5.pth",
    ),
    "b6": os.path.join(
        bucket_root,
        "adas/big_model/model_zoo/hat_noisy_student_efficientnet-b6.pth",
    ),
    "b7": os.path.join(
        bucket_root,
        "adas/big_model/model_zoo/hat_noisy_student_efficientnet-b7.pth",
    ),
}

stride2channels_dict = {
    "b0": {2: 16, 4: 24, 8: 40, 16: 112, 32: 320},
    "b1": {2: 16, 4: 24, 8: 40, 16: 112, 32: 320},
    "b2": {2: 16, 4: 24, 8: 48, 16: 120, 32: 352},
    "b3": {2: 24, 4: 32, 8: 48, 16: 136, 32: 384},
    "b4": {2: 24, 4: 32, 8: 56, 16: 160, 32: 448},
    "b5": {2: 24, 4: 40, 8: 64, 16: 176, 32: 512},
    "b6": {2: 32, 4: 40, 8: 72, 16: 200, 32: 576},
    "b7": {2: 32, 4: 48, 8: 80, 16: 224, 32: 640},
    "b8": {2: 32, 4: 56, 8: 88, 16: 248, 32: 704},
}

coefficient_params_dict = {
    "b0tiny": (0.5, 0.5, 224, 0.2),
    "b0small": (0.5, 1.0, 224, 0.2),
    "b0": (1.0, 1.0, 224, 0.2),
    "b1": (1.0, 1.1, 240, 0.2),
    "b2": (1.1, 1.2, 260, 0.3),
    "b3": (1.2, 1.4, 300, 0.3),
    "b4": (1.4, 1.8, 380, 0.4),
    "b5": (1.6, 2.2, 456, 0.4),
    "b6": (1.8, 2.6, 528, 0.5),
    "b7": (2.0, 3.1, 600, 0.5),
}


def build_backbone_hat_efficientnet(arch="b5", node_name="backbone"):

    checkpoint = checkpoints[arch]
    stride2channels = stride2channels_dict[arch]
    coefficient_params = coefficient_params_dict[arch]

    hat_efficientnet_backbone = dict(
        type="EfficientNet",
        model_type=arch,
        coefficient_params=coefficient_params,
        num_classes=1000,
        activation="swish",
        use_se_block=True,
        include_top=False,
        flat_output=False,
        bn_kwargs={},
        node_name=node_name,
    )

    return dict(
        backbone=hat_efficientnet_backbone,
        stride2channels=stride2channels,
        checkpoint=checkpoint,
    )


# Register network in GraphTracer
GraphTracer.register_basic_types(efficientnet)


if __name__ == "__main__":
    from hat.registry import build_from_registry

    hat_efficientnet_backbone = build_backbone_hat_efficientnet("b5")
    backbone = build_from_registry(hat_efficientnet_backbone["backbone"])
    import torch

    x = torch.rand((1, 3, 480, 480))
    feats = backbone(x)

    print("End")
