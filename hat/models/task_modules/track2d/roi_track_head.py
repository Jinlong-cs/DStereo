from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY

__all__ = ["RCNNTrackSplitHead"]


@OBJECT_REGISTRY.register
class RCNNTrackSplitHead(nn.Module):
    """RCNN track split head is not shared within a task group.

    The output of this module is the track feature.

    Args:
        in_channel: Number of channels of input feature maps.
        pw_num_filter2: Number of filters of second pw conv.
        feat_len: The length of output feature.
        bn_kwargs: Kwargs of BN layer.
        int8_output: Determines whether the dtype of output is int8.
    """

    def __init__(
        self,
        in_channel: int,
        pw_num_filter2: int,
        feat_len: int,
        bn_kwargs: Dict,
        int8_output: bool = False,
    ):
        super().__init__()
        self.int8_output = int8_output
        self.dequant = DeQuantStub()

        # rcnn split fc
        self.conv1 = ConvModule2d(
            in_channels=in_channel,
            out_channels=pw_num_filter2,
            kernel_size=4,
            stride=1,
            padding=0,
            norm_layer=nn.BatchNorm2d(pw_num_filter2, **bn_kwargs),
            act_layer=nn.ReLU(inplace=True),
        )

        self.feat = ConvModule2d(
            in_channels=pw_num_filter2,
            out_channels=feat_len,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_layer=None,
            act_layer=None,
        )

    def forward(self, x: torch.Tensor) -> Dict:
        x = self.conv1(x)
        track_feat = self.dequant(self.feat(x))
        output = OrderedDict(track_feat=track_feat)
        return output

    def fuse_model(self):
        self.conv1.fuse_model()
        # Don't need to fuse output ops since no norm and act
        # ops presented

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        # disable output quantization for last quanti layer.
        if not self.int8_output:
            self.feat.qconfig = qconfig_manager.get_default_qat_out_qconfig()
