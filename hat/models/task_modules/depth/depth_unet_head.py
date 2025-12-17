import logging
from collections import OrderedDict
from typing import Dict, List

import horizon_plugin_pytorch as horizon
from horizon_plugin_pytorch.nn import Interpolate
from torch import nn
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.task_modules.ipm_seg.mask_cat_feat import base_conv_module
from hat.models.weight_init import normal_init
from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class DepthUnetHead(nn.Module):
    """Head Module for depth prediction task.

    Args:
        in_strides: The strides corresponding to the inputs of
            depth_head, the inputs usually come from backbone or neck.
        out_strides: List of output strides.
        in_channels: The channels of inputs .
        out_channels:
            The channels of hidden features (of each output stride).
        pred_out_channel: Final prediction output channel. Default: 2
        stacked_convs: Number of stacking convs of head.  Default: 3
        start_level: Begining index of neck features input to head.
            Default: 0
        end_level: End index of neck features input to head. Default: 2
        group_base: Group param used in conv. Default: 8
        conv_method: Choice of convolution method used in head.
            Default: varg_conv
        use_auxi_loss: Whether to use auxi loss in head. Default: False
        with_refine: Whether to use depth refine module. Default: False
        last_with_relu: Whether to user relu at last conv layer.
            Default: False
        dequant_output : Whether to dequant output. Default: True
        int8_output: If True, output int8, otherwise output int32.
            Default: True
        bn_kwargs: Extra keyword arguments for bn layers.

    """

    def __init__(
        self,
        in_strides: List[int],
        out_strides: List[int],
        in_channels: List[int],
        out_channels: List[int],
        head_out_names: Dict,
        pred_out_channel: int = 2,
        stacked_convs: int = 3,
        start_level: int = 0,
        end_level: int = 2,
        group_base: int = 8,
        conv_method: str = "varg_conv",
        use_auxi_loss: bool = False,
        with_refine: bool = False,
        last_with_relu: bool = False,
        dequant_output: bool = True,
        int8_output: bool = True,
        bn_kwargs: bool = None,
    ):
        super(DepthUnetHead, self).__init__()

        assert conv_method in ["varg_conv", "conv2d", "sep_conv"]
        assert start_level >= 0 and start_level <= end_level
        assert in_strides[start_level] >= out_strides[0]

        self.conv_method = conv_method
        self.group_base = group_base

        self.in_strides = in_strides
        self.out_strides = out_strides
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pred_out_channel = pred_out_channel

        self.stacked_convs = stacked_convs
        self.start_level = start_level
        self.end_level = end_level

        self.use_auxi_loss = use_auxi_loss

        self.bn_kwargs = bn_kwargs

        # TODO: add refine module of depth head
        self.with_refine = with_refine

        self.last_with_relu = last_with_relu

        self.int8_output = int8_output
        self.dequant_output = dequant_output
        self.dequant = DeQuantStub()

        self._init_decode_layers()
        self._init_depth_layers()
        self._init_pred_layers()

        # need to be improved
        self.scale_factor = in_strides[start_level] // out_strides[0]
        self.upsample = Interpolate(
            scale_factor=self.scale_factor,
            align_corners=False,
            recompute_scale_factor=True,
        )

        # pred_output_names
        self.head_out_names = head_out_names

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

    def _init_pred_layers(self):
        self.conv_pred_layers = nn.ModuleList()
        out_channels_rev = self.out_channels[1:][::-1]
        for i in range(len(out_channels_rev)):
            self.conv_pred_layers.append(
                ConvModule2d(
                    in_channels=out_channels_rev[i],
                    out_channels=self.pred_out_channel,
                    kernel_size=1,
                    norm_layer=None,
                    act_layer=nn.ReLU(inplace=True)
                    if self.last_with_relu
                    else None,
                )
            )
            if not self.use_auxi_loss:
                break

    def _init_depth_layers(self):
        self.conv_depth_layers = nn.ModuleList()
        out_channels_rev = self.out_channels[1:][::-1]
        for i in range(len(out_channels_rev)):
            convs_per_level = nn.Sequential()
            for j in range(self.stacked_convs):
                one_conv = base_conv_module(
                    out_channels_rev[i],
                    out_channels_rev[i],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    dilation=1,
                    bn_kwargs=self.bn_kwargs,
                    group_base=self.group_base,
                    conv_method=self.conv_method,
                )
                convs_per_level.add_module("conv" + str(i) + str(j), one_conv)
            self.conv_depth_layers.append(convs_per_level)
            if not self.use_auxi_loss:
                break

    def _init_decode_layers(self):
        in_channels = self.in_channels[self.start_level : self.end_level + 1]

        self.decode_conv_layers = nn.ModuleList()
        self.cat_layers = nn.ModuleList()
        self.cat_conv_layers = nn.ModuleList()
        in_channels_rev = in_channels[::-1]
        in_channel_lowest = in_channels_rev[0]
        out_channel_lowest = self.out_channels[0]
        for i in range(len(in_channels_rev) - 1):
            convs_per_level = nn.Sequential()
            in_channel = in_channel_lowest if i == 0 else self.out_channels[i]
            out_channel = (
                out_channel_lowest if i == 0 else self.out_channels[i]
            )
            one_conv = base_conv_module(
                in_channel,
                out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                bn_kwargs=self.bn_kwargs,
                group_base=self.group_base,
                conv_method=self.conv_method,
            )
            one_upsample = Interpolate(
                scale_factor=2,
                mode="bilinear",
                align_corners=None,
                recompute_scale_factor=True,
            )
            convs_per_level.add_module("decode_conv" + str(i) + "0", one_conv)
            convs_per_level.add_module("decode_up" + str(i), one_upsample)
            self.decode_conv_layers.append(convs_per_level)

            self.cat_layers.append(nn.quantized.FloatFunctional())

            in_channel = out_channel + in_channels_rev[i + 1]
            cat_conv = base_conv_module(
                in_channel,
                self.out_channels[i + 1],
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                bn_kwargs=self.bn_kwargs,
                group_base=self.group_base,
                conv_method=self.conv_method,
            )
            self.cat_conv_layers.append(cat_conv)

    def forward(self, inputs):
        assert isinstance(inputs, List)
        assert len(inputs) >= (self.end_level - self.start_level + 1)

        inputs = inputs[self.start_level : self.end_level + 1]

        # decode layers
        decode_feat = inputs[-1]
        len_decode_layer = len(self.decode_conv_layers)
        decode_feats = []
        for i in range(len_decode_layer):
            decode_feat = self.decode_conv_layers[i](decode_feat)
            decode_feat = self.cat_layers[i].cat(
                [decode_feat, inputs[len(inputs) - i - 2]], dim=1
            )
            decode_feat = self.cat_conv_layers[i](decode_feat)
            if self.use_auxi_loss and i < len_decode_layer - 1:
                decode_feats.insert(0, decode_feat)
        decode_feats.insert(0, decode_feat)

        # depth layers
        depth_feats = []
        depth_feat = self.conv_depth_layers[0](decode_feats[0])
        depth_feats.append(depth_feat)
        if self.use_auxi_loss:
            for i in range(1, len_decode_layer):
                depth_feats.append(self.conv_depth_layers[i](decode_feats[i]))

        # depth prediction
        preds = []

        depth_feat_0 = self.upsample(depth_feats[0])
        pred = self.conv_pred_layers[0](depth_feat_0)
        pred = self.dequant(pred)
        preds.append(pred)
        if self.use_auxi_loss:
            for i in range(1, len(decode_feats)):
                preds.append(
                    self.dequant(self.conv_pred_layers[i](depth_feats[i]))
                )

        head_preds = OrderedDict()
        head_preds[self.head_out_names["depth"]] = tuple(preds)
        return head_preds

    def fuse_model(self):
        modules = [
            self.decode_conv_layers,
            self.cat_conv_layers,
            self.conv_depth_layers,
            self.conv_pred_layers,
        ]
        for module in modules:
            if module is not None:
                for m in module:
                    if hasattr(m, "fuse_model"):
                        m.fuse_model()
                    elif isinstance(m, nn.Sequential):
                        for op in m:
                            if hasattr(op, "fuse_model"):
                                op.fuse_model()

    def set_qconfig(self):
        self.qconfig = horizon.quantization.get_default_qat_qconfig()
        if not self.int8_output:
            self.conv_pred_layers[
                0
            ].qconfig = horizon.quantization.get_default_qat_out_qconfig()
