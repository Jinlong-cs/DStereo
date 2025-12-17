import torch

from .bifpn import build_neck_bifpn
from .csppafpn import build_neck_csppafpn
from .fpn import build_neck_fpn
from .pafpn import build_neck_pafpn


def build_neck_identity(stride2channels):
    return dict(
        neck=torch.nn.Identity(),
        stride2channels=stride2channels,
    )


def build_neck(
    name: str,
    in_stride2channels: dict,
    out_stride2channels: dict,
    neck_stack: int = 3,
    node_name: str = "neck",
):
    if name == "bifpn":
        return build_neck_bifpn(
            in_stride2channels,
            out_stride2channels,
            neck_stack=neck_stack,
            node_name=node_name,
        )
    elif name == "pafpn":
        return build_neck_pafpn(
            in_stride2channels, out_stride2channels, node_name=node_name
        )
    elif name == "fpn":
        return build_neck_fpn(
            in_stride2channels, out_stride2channels, node_name=node_name
        )
    elif name == "identity":
        return build_neck_identity(in_stride2channels)
    elif name == "csppafpn":
        return build_neck_csppafpn(
            in_stride2channels, out_stride2channels, node_name=node_name
        )
    else:
        raise NotImplementedError(name)


__all__ = ["build_neck"]
