from typing import Mapping

import horizon_plugin_pytorch as horizon
import torch.nn as nn
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.base_modules.separable_conv_module import SeparableConvModule2d
from hat.models.utils import _take_features
from hat.models.weight_init import bias_init_with_prob, normal_init
from hat.registry import OBJECT_REGISTRY

__all__ = ["ANCBEVParkingRodHead"]


@OBJECT_REGISTRY.register
class ANCBEVParkingRodHead(nn.Module):
    """Head module for bev parking rod detection task.

    Args:
        num_parkingrod_class: Number of categories.
        feature_name: Name of features from neck
            in input dict.
        in_strides: strides for input
        out_stride: the stride of selected downsampling
        in_channels: the channels of input feature
        sep_conv: Use separable convolution. default is False.
        stack: The number of prepred conv block stacked together.
        use_bias: Use bias. default is True.
        aux_branch: Aux supervision of slot to parkingrod.
        mid_channels(int): the channels of middle feature, default is the
            same as in_channels.
    """

    def __init__(
        self,
        num_parkingrod_class: int,
        in_strides: list,
        out_strides: int,
        in_channels: int,
        feature_name: str = None,
        sep_conv: bool = False,
        stack: int = 1,
        use_bias: bool = True,
        aux_branch: bool = False,
        mid_channels: int = -1,
    ):
        super(ANCBEVParkingRodHead, self).__init__()
        self.num_parkingrod_class = num_parkingrod_class
        self.feature_name = feature_name
        self.in_strides = in_strides
        self.out_strides = out_strides
        self.in_channels = in_channels
        self.mid_channels = mid_channels if mid_channels > 0 else in_channels
        self.sep_conv = sep_conv
        self.stack = stack
        self.use_bias = use_bias
        self.aux_branch = aux_branch
        self.dequant = DeQuantStub()

        self._init_layers()
        self._init_weight()

    def _init_module(self, out_channels):
        convs = []
        for _ in range(self.stack):
            if self.sep_conv:
                prepred_conv_block = SeparableConvModule2d(
                    in_channels=self.in_channels,
                    out_channels=self.mid_channels,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    pw_norm_layer=nn.BatchNorm2d(self.mid_channels),
                    pw_act_layer=nn.ReLU(inplace=True),
                )
                convs.append(prepred_conv_block)
            else:
                prepred_conv_block = ConvModule2d(
                    in_channels=self.in_channels,
                    out_channels=self.mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=self.use_bias,
                    norm_layer=nn.BatchNorm2d(self.mid_channels),
                    act_layer=nn.ReLU(inplace=True),
                )
                convs.append(prepred_conv_block)
        convs.append(
            nn.Conv2d(
                in_channels=self.mid_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
            )
        )
        return nn.Sequential(*convs)

    def _init_layers(self):
        self.classification_conv = self._init_module(out_channels=1)
        self.endpoint_offset_conv = self._init_module(out_channels=4)
        self.slot_01offset_conv = self._init_module(out_channels=4)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.classification_conv[-1], std=0.01, bias=bias_cls)

    def forward(self, data):
        data = data[self.feature_name] if isinstance(data, Mapping) else data
        # input_features = (
        #     data[self.feature_name][0] if isinstance(data, Mapping) else data
        # )
        feature = _take_features(data[0], self.in_strides, self.out_strides)[0]

        classification_pred = self.classification_conv(feature)
        endpoint_offset_pred = self.endpoint_offset_conv(feature)

        classification_pred = self.dequant(classification_pred)
        endpoint_offset_pred = self.dequant(endpoint_offset_pred)

        if self.aux_branch:
            slot_01offset_pred = self.slot_01offset_conv(feature)
            slot_01offset_pred = self.dequant(slot_01offset_pred)
            return [
                classification_pred,
                endpoint_offset_pred,
                slot_01offset_pred,
            ]

        return [
            classification_pred,
            endpoint_offset_pred,
        ]

    def set_qconfig(self):
        self.qconfig = horizon.quantization.get_default_qat_qconfig()

    def fuse_model(self):
        for m in self.modules():
            if isinstance(m, ConvModule2d):
                m.fuse_model()
