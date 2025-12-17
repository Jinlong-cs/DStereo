from collections import OrderedDict
from typing import Dict, Tuple

import horizon_plugin_pytorch as horizon
import torch
import torch.nn as nn
from numpy import iterable
from torch import Tensor
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_factory import get_conv_module
from hat.models.weight_init import bias_init_with_prob, normal_init
from hat.registry import OBJECT_REGISTRY

__all__ = ["CenterNetHead"]


@OBJECT_REGISTRY.register
class CenterNetHead(nn.Module):
    """CenterNetHead.

    Args:
        num_classes: Number of categories excluding the background category.
        strides2levels: A stride to levels of feature map dict.
        in_strides: the stride of used feature maps from neck.
        feat_channels: Number of channel in the input feature map.
        in_channel: Number of channel in the intermediate feature map.
        local_max_kernel: Max pooling kernel for extracting local maximum
            pixels. Default is 3.
        conv_type: type of convolution module, option is [sep_conv, varg_conv,
            conv].
        num_conv_stage: number of convolution modules before predictor
            convolution.
        int8_output: This decides whether to use int8 or int 32 output format.
        wh_relu: Whether to apply relu to wh regression outputs.
    """

    def __init__(
        self,
        num_classes: int,
        strides2levels: Dict[int, int],
        in_strides: int = 4,
        feat_channels: int = 32,
        in_channels: int = 32,
        local_max_kernel: int = 3,
        conv_type: str = "sep_conv",
        num_conv_stage: int = 2,
        int8_output: bool = True,
        wh_relu: bool = False,
    ):
        super(CenterNetHead, self).__init__()
        self.num_classes = num_classes
        self.in_strides = in_strides
        self.strides2levels = strides2levels
        self.feat_channels = feat_channels
        self.in_channels = in_channels
        self.local_max_kernel = local_max_kernel
        self.conv_type = conv_type
        self.num_conv_stage = num_conv_stage
        self.int8_output = int8_output
        self.wh_relu = wh_relu

        self._init_layers()
        self._init_weight()

    def forward(self, feats: Tuple[Tensor, ...]) -> Dict[str, Tensor]:
        """Forward features. Notice CenterNet head only use one level feature.

        Args:
            feats: Features from the upstream network, each is a 4D-tensor.

        Returns:
            heatmap_pred: Center predict heatmaps for single level.
            wh_pred: Wh predicts for single level.
            offset_pred: Offset predicts for single level.
            mp_heat_pred: Heatmap + maxpool output.

        """
        level = self.strides2levels.get(self.in_strides)
        feat = feats[level]
        heatmap_pred = self.heatmap_conv(feat)
        wh_pred = self.wh_conv(feat)
        offset_pred = self.offset_conv(feat)

        mp_heat_pred = self.qmax_pool(heatmap_pred)  # for post-processing

        heatmap_pred = self.dequant(heatmap_pred)
        wh_pred = self.dequant(wh_pred)
        offset_pred = self.dequant(offset_pred)
        mp_heat_pred = self.dequant(mp_heat_pred)

        output = OrderedDict(
            heatmap_pred=heatmap_pred,
            wh_pred=wh_pred,
            offset_pred=offset_pred,
            mp_heat_pred=mp_heat_pred,
        )
        return output

    def _init_pre_pred_conv(self):
        convs = []
        for ii in range(self.num_conv_stage):
            in_channels = self.in_channels if ii == 0 else self.feat_channels
            convs.append(
                get_conv_module(
                    in_channels,
                    self.feat_channels,
                    kernel_size=3,
                    padding=1,
                    conv_method=self.conv_type,
                    dw_activation=None,
                    pw_activation=nn.ReLU,
                    dw_norm_method=None,
                    pw_norm_method=nn.BatchNorm2d,
                )
            )
        return nn.Sequential(*convs)

    def _init_module(self, out_channels, out_relu=None):
        return nn.Sequential(
            self._init_pre_pred_conv(),
            get_conv_module(
                self.feat_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                conv_method="conv",
                dw_activation=out_relu,
            ),
        )

    def _init_layers(self):
        """Initialize layers of the head."""
        self.heatmap_conv = self._init_module(self.num_classes)
        self.wh_conv = self._init_module(
            2, out_relu=nn.ReLU if self.wh_relu else None
        )
        self.offset_conv = self._init_module(2)
        self.qmax_pool = torch.nn.MaxPool2d(
            self.local_max_kernel,
            stride=1,
            padding=(self.local_max_kernel - 1) // 2,
        )
        self.dequant = DeQuantStub()

    def _init_weight(self):
        """Initialize weights of the head."""
        bias_init = bias_init_with_prob(0.1)
        self.heatmap_conv[-1][0].bias.data.fill_(bias_init)
        for head in [
            self.heatmap_conv[0],
            self.wh_conv[0],
            self.offset_conv[0],
        ]:
            for m in head:
                if not iterable(m):
                    continue
                for n in m:
                    for a in n:
                        if isinstance(a, nn.Conv2d):
                            normal_init(a, std=0.01)

    def fuse_model(self):
        for head in [
            self.heatmap_conv[0],
            self.wh_conv[0],
            self.offset_conv[0],
        ]:
            for m in head:
                for n in m:
                    n.fuse_model()

    def set_qconfig(self):
        self.qconfig = horizon.quantization.get_default_qat_qconfig()

        # disable output quantization for last quanti layer.
        if not self.int8_output:
            from hat.utils import qconfig_manager

            self.heatmap_conv[
                -1
            ].qconfig = qconfig_manager.get_default_qat_out_qconfig()
            self.wh_conv[
                -1
            ].qconfig = qconfig_manager.get_default_qat_out_qconfig()
            self.offset_conv[
                -1
            ].qconfig = qconfig_manager.get_default_qat_out_qconfig()
