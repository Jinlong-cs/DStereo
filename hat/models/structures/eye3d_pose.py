# Copyright (c) Horizon Robotics. All rights reserved.

import logging

import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.base_modules.extend_container import ExtSequential
from hat.registry import OBJECT_REGISTRY

__all__ = ["Eye3dPoseModel"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class Eye3dPoseModel(nn.Module):
    """The network of Eye3dPose model.

    Args:
        num_mod: num of input mods and model groups
        input_channels: num of input channels
        backbone_scale_list: list of filter nums for backbone
        neck_scale_list: list of filter nums for necks
        losses: loss function. Defaults to None.
        deploy: is deploy model. Defaults to False.
        bn_kwargs: bn args. Defaults to {}.
    """

    def __init__(
        self,
        num_mod: int,
        input_channels: int,
        backbone_scale_list: list,
        neck_scale_list: list,
        losses: nn.Module = None,
        deploy: bool = False,
        deploy_wo_pose: bool = False,
        bn_kwargs: dict = None,
    ):
        super().__init__()
        self.num_mod = num_mod
        self.quant = QuantStub()
        if bn_kwargs is None:
            bn_kwargs = {}
        self._init_backbone(
            backbone_scale_list, bn_kwargs, input_channels, num_mod
        )
        self._init_head(
            backbone_scale_list, neck_scale_list, bn_kwargs, num_mod
        )

        self.dequant_rot = DeQuantStub()
        self.dequant_pos = DeQuantStub()
        self.losses = losses
        self.deploy = deploy
        self.deploy_wo_pose = deploy_wo_pose

    def _init_backbone(
        self, backbone_scale_list, bn_kwargs, input_channels, num_mod
    ):
        backbone = []
        for i, _ in enumerate(backbone_scale_list):
            if i:
                backbone.append(
                    ConvModule2d(
                        backbone_scale_list[i - 1],
                        backbone_scale_list[i],
                        kernel_size=1,
                        groups=num_mod,
                        norm_layer=nn.BatchNorm2d(
                            backbone_scale_list[i], **bn_kwargs
                        ),
                        act_layer=nn.ReLU(inplace=True),
                    )
                )
            else:
                backbone.append(
                    ConvModule2d(
                        input_channels,
                        backbone_scale_list[i],
                        kernel_size=1,
                        groups=num_mod,
                        norm_layer=nn.BatchNorm2d(
                            backbone_scale_list[i], **bn_kwargs
                        ),
                        act_layer=nn.ReLU(inplace=True),
                    )
                )
        self.backbone = ExtSequential(backbone)

    def _init_head(
        self, backbone_scale_list, neck_scale_list, bn_kwargs, num_mod
    ):
        head_rot, head_pos = [], []
        for i, _ in enumerate(neck_scale_list):
            if i:
                head_rot.append(
                    ConvModule2d(
                        neck_scale_list[i - 1],
                        neck_scale_list[i],
                        kernel_size=1,
                        groups=num_mod,
                        norm_layer=nn.BatchNorm2d(
                            neck_scale_list[i], **bn_kwargs
                        ),
                        act_layer=nn.ReLU(inplace=True),
                    )
                )
                head_pos.append(
                    ConvModule2d(
                        neck_scale_list[i - 1],
                        neck_scale_list[i],
                        kernel_size=1,
                        groups=num_mod,
                        norm_layer=nn.BatchNorm2d(
                            neck_scale_list[i], **bn_kwargs
                        ),
                        act_layer=nn.ReLU(inplace=True),
                    )
                )
            else:
                head_rot.append(
                    ConvModule2d(
                        backbone_scale_list[-1],
                        neck_scale_list[i],
                        kernel_size=1,
                        groups=num_mod,
                        norm_layer=nn.BatchNorm2d(
                            neck_scale_list[i], **bn_kwargs
                        ),
                        act_layer=nn.ReLU(inplace=True),
                    )
                )
                head_pos.append(
                    ConvModule2d(
                        backbone_scale_list[-1],
                        neck_scale_list[i],
                        kernel_size=1,
                        groups=num_mod,
                        norm_layer=nn.BatchNorm2d(
                            neck_scale_list[i], **bn_kwargs
                        ),
                        act_layer=nn.ReLU(inplace=True),
                    )
                )
        head_pos.append(
            ConvModule2d(
                neck_scale_list[-1],
                3 * num_mod,
                kernel_size=1,
                groups=num_mod,
            )
        )
        head_rot.append(
            ConvModule2d(
                neck_scale_list[-1],
                3 * num_mod,
                kernel_size=1,
                groups=num_mod,
            )
        )
        self.head_pos = ExtSequential(head_pos)
        self.head_rot = ExtSequential(head_rot)

    def forward(self, data: dict):
        x = self.quant(data["img"])
        target = data.get("gt_eye3d_pose", None)

        base = self.backbone(x)
        rot, pos = self.head_rot(base), self.head_pos(base)
        rot, pos = self.dequant_rot(rot), self.dequant_pos(pos)
        preds = {"pred_pose": rot, "pred_eye3d": pos}
        if self.deploy:
            if self.deploy_wo_pose:
                return pos
            return (rot, pos)
        if not self.training or self.losses is None:
            return preds, target
        loss_dict = {
            "loss_pose": self.losses(preds["pred_pose"], target["gt_pose"]),
            "loss_eye3d": self.losses(preds["pred_eye3d"], target["gt_eye3d"]),
        }
        preds["loss_dict"] = loss_dict
        total_loss = 0.0
        for _, value in loss_dict.items():
            total_loss += value
        return preds, total_loss

    def fuse_model(self):
        for module in [self.backbone, self.head_pos, self.head_rot]:
            if module is not None and hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        for module in [self.backbone, self.head_pos, self.head_rot]:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()
                else:
                    module.qconfig = None
        if self.losses is not None:
            self.losses.qconfig = None

    def set_calibration_qconfig(self):
        from hat.utils import qconfig_manager

        self.calibration_qconfig = (
            qconfig_manager.get_default_calibration_qconfig()
        )
        if self.losses is not None:
            self.losses.qconfig = None
