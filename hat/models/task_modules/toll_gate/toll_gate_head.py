from collections import OrderedDict
from collections.abc import Mapping
from typing import Dict, List, Union

import torch.nn as nn
from torch import Tensor
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "TollGateHead",
]


@OBJECT_REGISTRY.register
class TollGateHead(nn.Module):
    """
    TollGateHead module.

    Args:
        in_channels (dict): A list of to indicates the input channels of the
            block.
        hm_channels (dict): A list of to indicates the heatmap output channels
            of the block.
        offset_channels (dict): A list of to indicates the offset output
            channels of the block.
        use_bias (bool): Use bias.
    """

    def __init__(
        self,
        in_channels: int,
        hm_channels: int,
        offset_channels: int,
        use_bias: bool = False,
        int8_output: bool = True,
    ):
        super(TollGateHead, self).__init__()
        self.use_bias = use_bias
        self.int8_output = int8_output
        self.heatmap_head = ConvModule2d(
            in_channels=in_channels,
            out_channels=hm_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.use_bias,
            norm_layer=nn.BatchNorm2d(hm_channels),
            act_layer=nn.ReLU(inplace=True),
        )
        self.offset_head = ConvModule2d(
            in_channels=in_channels,
            out_channels=offset_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.use_bias,
            norm_layer=None,
            act_layer=None,
        )
        self.dequant = DeQuantStub()

    def forward(self, data: Union[Dict, List]) -> Dict[str, Tensor]:
        input_features = data["feats"] if isinstance(data, Mapping) else data
        feat_heatmap = self.heatmap_head(input_features[0])
        feat_offset = self.offset_head(input_features[0])
        out = OrderedDict()
        out["hm"] = self.dequant(feat_heatmap)
        out["offset"] = self.dequant(feat_offset)
        return out

    def fuse_model(self):
        for head in [self.heatmap_head, self.offset_head]:
            head.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        # Low precision
        if self.int8_output:
            self.qconfig = qconfig_manager.get_default_qat_qconfig()
        else:  # int16 precision
            self.offset_head.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
            self.heatmap_head.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
