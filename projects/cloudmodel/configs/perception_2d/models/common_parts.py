from typing import Dict, Optional

from projects.cloudmodel.configs.perception_2d.models.backbones import (
    build_backbone,
)
from projects.cloudmodel.configs.perception_2d.models.necks import build_neck

__all__ = ["build_backbone_neck"]


def build_backbone_neck(
    backbone_cfg: Optional[Dict[str, str]] = None,
    neck_cfg: Optional[Dict[str, str]] = None,
    out_stride2channels: Optional[Dict[int, int]] = None,
):
    if backbone_cfg is None:
        backbone_cfg = {}
    if neck_cfg is None:
        neck_cfg = {}
    if out_stride2channels is None:
        out_stride2channels = {s: 256 for s in [4, 8, 16, 32, 64]}

    # build the backbone
    backbone = build_backbone(
        name=backbone_cfg.get("name", "hat_efficientnet"),
        arch=backbone_cfg.get("arch", "b5"),
        node_name=backbone_cfg.get("node_name", "backbone"),
    )

    # build the neck
    neck = build_neck(
        name=neck_cfg.get("name", "bifpn"),
        in_stride2channels=backbone["stride2channels"],
        out_stride2channels=out_stride2channels,
        neck_stack=neck_cfg.get("neck_stack", 3),
        node_name=neck_cfg.get("node_name", "neck"),
    )

    return backbone, neck
