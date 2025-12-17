from .hat_cspdarknet import build_backbone_hat_cspdarknet
from .hat_efficientnet import build_backbone_hat_efficientnet
from .hat_swin import build_backbone_hat_swin


def build_backbone(name, arch, node_name="backbone"):

    if name == "hat_efficientnet":
        backbone = build_backbone_hat_efficientnet(
            arch=arch, node_name=node_name
        )
    elif name == "hat_swin":
        backbone = build_backbone_hat_swin(arch=arch, node_name=node_name)
    elif name == "hat_cspdarknet":
        backbone = build_backbone_hat_cspdarknet(
            arch=arch, node_name=node_name
        )
    else:
        raise NotImplementedError(name)

    return backbone
