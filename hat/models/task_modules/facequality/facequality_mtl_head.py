# Copyright (c) Horizon Robotics. All rights reserved.
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY

__all__ = ["FacequalityMultiHead"]


@OBJECT_REGISTRY.register
class FacequalityMultiHead(nn.Module):
    """Head Module for the task of face quality.

    Args:
        input_channels: Channels of each input feature map.
        feat_channels: Channels for the module.
        output_dim: Output dimension of the head.
        task_name: The task selected.
        flat_output: Whether to view the output tensor.
        loss: Loss module. Defaults to None. Only needed in training process.
        bn_kwargs: Extra keyword arguments for bn layers. Default: None.
    """

    def __init__(
        self,
        in_channels: int,
        feat_channels: List[int],
        output_dim: int,
        task_name: str,
        flat_output: bool = True,
        loss: Optional[nn.Module] = None,
        bn_kwargs: Optional[Dict] = None,
    ):
        super(FacequalityMultiHead, self).__init__()
        if bn_kwargs is None:
            bn_kwargs = {}
        self.feature = nn.Sequential(
            ConvModule2d(
                in_channels=in_channels,
                out_channels=feat_channels[0],
                kernel_size=3,
                stride=2,
                padding=1,
                norm_layer=nn.BatchNorm2d(feat_channels[0], **bn_kwargs),
                act_layer=nn.ReLU(inplace=True),
            ),
            ConvModule2d(
                in_channels=feat_channels[0],
                out_channels=feat_channels[1],
                kernel_size=2,
                stride=1,
                padding=0,
                norm_layer=nn.BatchNorm2d(feat_channels[1], **bn_kwargs),
                act_layer=nn.ReLU(inplace=True),
            ),
        )

        self.task_head = ConvModule2d(
            in_channels=feat_channels[1],
            out_channels=output_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_layer=None,
            act_layer=None,
        )

        self.loss = loss
        self.task_name = task_name
        self.dequant = DeQuantStub()
        self.output_dim = output_dim
        self.flat_output = flat_output

    def forward(
        self, feat_map: torch.Tensor, data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, torch.Tensor]:
        feat_map = self.feature(feat_map)
        pred = self.task_head(feat_map)

        pred = self.dequant(pred)

        if self.flat_output:
            if self.output_dim == 1:
                pred = pred.flatten()
            else:
                pred = pred.squeeze()

        out_dict = OrderedDict()
        out_dict["pred"] = pred

        if self.loss:
            target_key = "gt_" + self.task_name
            assert target_key in data.keys(), (
                "Data does not contain the passed"
                f"task_name: {self.task_name}, it is also possible that "
                "the key passed in data forgot to include the prefix `gt_`, "
                "please check."
            )
            if self.output_dim == 1:
                loss = self.loss(pred, data[target_key])
            else:
                loss = self.loss(pred, data[target_key].long())
            out_dict["loss"] = loss
        return out_dict

    def fuse_model(self):
        modules = [self.feature, self.task_head]
        for module in modules:
            for m in module:
                if hasattr(m, "fuse_model"):
                    m.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        # disable output quantization for last quanti layer.
        self.task_head.qconfig = qconfig_manager.get_default_qat_out_qconfig()
        if self.loss is not None:
            self.loss.qconfig = None
