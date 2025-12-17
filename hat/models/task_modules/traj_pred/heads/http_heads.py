# Copyright (c) Horizon Robotics. All rights reserved.

from typing import List, Optional, Tuple

import horizon_plugin_pytorch as horizon
import torch
import torch.nn as nn
from horizon_plugin_pytorch.quantization import (
    get_default_calib_qconfig,
    get_default_qat_out_qconfig,
    get_default_qat_qconfig,
)
from torch.nn import BatchNorm2d as BN

from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY
from ..utils import module_list_forward

__all__ = [
    "TrajRegHead",
    "HeatmapHead",
]


@OBJECT_REGISTRY.register
class TrajRegHead(nn.Module):
    """
    Trajectory regression head, the model structure is as follow:   \
    (agent_feature) -> 2D convs -> (feature with 1x1 spacial size)  \
                                                |                   \
    (trajectory) <- traj_layer <- fcs <- concat with state_feature  \
                                  | \
    (confidence) <- conf_layer <---.

    Args:
        bn_kwargs: Kwargs for batch normalization layers.
        input_channel: Channel of the input agent_feature.
        output_channel: Channel of the output trajectory.
        endpoint_feature_channel: Channel of the endpoint_feature.
        state_feature_channel: Channel of the state_feature.
        conv_channels: Channels of the convolutions for down-scale.
        paddings:  Padding sizes of the convolutions for down-scale.
        fc_channels: Channels of the full connect layers.
        pooling_size: Max pooling size between 2D convs and fcs.
        confidence: Whether to output confidence.

    Shaple:
        - Input:
        * agent_feature: agent_num x C_a x S x S.
        * endpoint_features: [agent_num x C_e x 1 x 1] * endpoint_num.
        * state_feature: agent_num x C_s x 1 x 1.
        - Output:
        * trajectory: (agent_num x endpoint_num) x output_channel x 1 x 1.
        * confidence: (agent_num x endpoint_num) x 1 x 1 x 1.
    """

    def __init__(
        self,
        bn_kwargs: dict,
        input_channel: int,
        output_channel: int,
        endpoint_feature_channel: int,
        state_feature_channel: int,
        conv_channels: List[int],
        paddings: List[int],
        fc_channels: List[int],
        pooling_size: int = -1,
        confidence: bool = False,
    ):
        super(TrajRegHead, self).__init__()

        channels = [input_channel] + conv_channels
        self.convs = nn.ModuleList()
        for i in range(len(paddings)):
            self.convs.append(
                ConvModule2d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=3,
                    padding=paddings[i],
                    stride=1,
                    bias=False,
                    act_layer=nn.ReLU(inplace=True),
                    norm_layer=BN(channels[i + 1], **bn_kwargs),
                )
            )
        if pooling_size > 1:
            self.convs.append(nn.MaxPool2d(pooling_size))

        fc_channels.insert(
            0, channels[-1] + endpoint_feature_channel + state_feature_channel
        )
        self.fcs = nn.ModuleList()
        for i in range(len(fc_channels) - 1):
            self.fcs.append(
                ConvModule2d(
                    fc_channels[i],
                    fc_channels[i + 1],
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    bias=False,
                    act_layer=nn.ReLU(inplace=True),
                    norm_layer=BN(fc_channels[i + 1], **bn_kwargs),
                )
            )
        self.traj_layer = nn.Conv2d(
            fc_channels[-1],
            output_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=True,
        )
        if confidence:
            self.conf_layer = nn.Conv2d(
                fc_channels[-1],
                1,
                kernel_size=1,
                padding=0,
                stride=1,
                bias=True,
            )

        self.cat_op = nn.quantized.FloatFunctional()
        self.confidence = confidence

    def forward(
        self,
        agent_feature: torch.Tensor,
        endpoint_features: List[torch.Tensor],
        state_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        agent_feature_vector = module_list_forward(self.convs, agent_feature)

        features = []
        for epf in endpoint_features:
            features.append(
                self.cat_op.cat(
                    [agent_feature_vector, epf, state_feature], dim=1
                )
            )
        feature = self.cat_op.cat(features, dim=0)
        feature = module_list_forward(self.fcs, feature)

        trajectory = self.traj_layer(feature)

        if self.confidence:
            confidence = self.conf_layer(feature)
        else:
            confidence = None

        return trajectory, confidence

    def fuse_model(self):
        for module in [self.convs, self.fcs]:
            for layer in module:
                if hasattr(layer, "fuse_model"):
                    layer.fuse_model()

    def set_qconfig(self):
        self.qconfig = get_default_qat_qconfig()
        self.traj_layer.qconfig = get_default_qat_out_qconfig()
        if self.confidence:
            self.conf_layer.qconfig = get_default_qat_out_qconfig()

    def set_calibration_qconfig(self):
        self.qconfig = get_default_calib_qconfig()
        self.traj_layer.qconfig = get_default_qat_out_qconfig()
        if self.confidence:
            self.conf_layer.qconfig = get_default_qat_out_qconfig()


@OBJECT_REGISTRY.register
class HeatmapHead(nn.Module):
    """
    Heatmap head, implemented with 2D convs and Interpolate. \
    (input) -> [conv2d+conv2d+interpolate] x N -> heatmap_layer -> (heatmap) \
                                               |                             \
                                               ----> offset_layer -> (offset).

    Args:
        bn_kwargs: Kwargs for batch normalization layers.
        input_channel: Channel of the input agent_feature.
        channels: Channels of the 2D convs.
        with_offset: Whether to output offset.

    Shape:
        - Input: agent_num x C x H x W.
        - Output:
        * heatmap: agent_num x 1 x H' x W'.
        * offset: agent_num x 2 x H' x W'.
        where H' = H x 2^N, W' = W x 2^N.
    """

    def __init__(
        self,
        bn_kwargs: dict,
        input_channel: int,
        channels: List[int],
        with_offset: bool = False,
    ):
        super(HeatmapHead, self).__init__()

        channels = [input_channel] + channels
        self.convs = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.convs.extend(
                [
                    ConvModule2d(
                        channels[i],
                        channels[i + 1],
                        kernel_size=3,
                        padding=1,
                        stride=1,
                        bias=False,
                        act_layer=nn.ReLU(inplace=True),
                        norm_layer=BN(channels[i + 1], **bn_kwargs),
                    ),
                    ConvModule2d(
                        channels[i + 1],
                        channels[i + 1],
                        kernel_size=3,
                        padding=1,
                        stride=1,
                        bias=False,
                        act_layer=nn.ReLU(inplace=True),
                        norm_layer=BN(channels[i + 1], **bn_kwargs),
                    ),
                    horizon.nn.Interpolate(
                        scale_factor=2,
                        mode="bilinear",
                        recompute_scale_factor=True,
                    ),
                ]
            )

        self.heatmap_layers = nn.ModuleList()
        self.heatmap_layers.append(
            ConvModule2d(
                channels[-1],
                32,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
                act_layer=nn.ReLU(inplace=True),
                norm_layer=BN(32, **bn_kwargs),
            )
        )
        self.heatmap_layers.append(
            nn.Conv2d(32, 1, kernel_size=3, padding=1, stride=1, bias=True)
        )

        self.with_offset = with_offset
        if with_offset:
            self.offset_layers = nn.ModuleList()
            self.offset_layers.append(
                ConvModule2d(
                    channels[-1],
                    32,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    bias=False,
                    act_layer=nn.ReLU(inplace=True),
                    norm_layer=BN(32, **bn_kwargs),
                )
            )
            self.offset_layers.append(
                nn.Conv2d(32, 2, kernel_size=3, padding=1, stride=1, bias=True)
            )

    def forward(
        self, agent_feature: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        feature_map = module_list_forward(self.convs, agent_feature)
        heatmap = module_list_forward(self.heatmap_layers, feature_map)

        if self.with_offset:
            offset = module_list_forward(self.offset_layers, feature_map)
        else:
            offset = None
        return heatmap, offset

    def fuse_model(self):
        for module in [self.convs, self.heatmap_layers]:
            for layer in module:
                if hasattr(layer, "fuse_model"):
                    layer.fuse_model()
        if self.with_offset:
            for layer in self.offset_layers:
                if hasattr(layer, "fuse_model"):
                    layer.fuse_model()

    def set_qconfig(self):
        self.qconfig = get_default_qat_qconfig()
        self.heatmap_layers[-1].qconfig = get_default_qat_out_qconfig()
        if self.with_offset:
            self.offset_layers[-1].qconfig = get_default_qat_out_qconfig()

    def set_calibration_qconfig(self):
        self.qconfig = get_default_calib_qconfig()
        self.heatmap_layers[-1].qconfig = get_default_qat_out_qconfig()
        if self.with_offset:
            self.offset_layers[-1].qconfig = get_default_qat_out_qconfig()
