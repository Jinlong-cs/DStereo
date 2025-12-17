# Copyright (c) Horizon Robotics. All rights reserved.

from collections import OrderedDict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import horizon_plugin_pytorch as horizon
import horizon_plugin_pytorch.nn as hnn
import numpy as np
import torch
import torch.nn as nn
from horizon_plugin_pytorch.quantization.stubs import QuantStub as HQuantStub
from torch.quantization import DeQuantStub

from hat.models.base_modules.basic_vargnet_module import BasicVarGBlock
from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.base_modules.separable_conv_module import (
    SeparableConvModule2d,
    SeparableGroupConvModule2d,
)
from hat.models.task_modules.dddv.head import OutputBlock
from hat.models.utils import _take_features
from hat.models.weight_init import bias_init_with_prob, normal_init
from hat.registry import OBJECT_REGISTRY
from hat.utils import qconfig_manager
from hat.utils.apply_func import _as_list, _is_increasing_sequence

__all__ = [
    "ANCBEV3DHead",
    "ANCBEVStaticHead",
    "ANCBevDiscobjHead",
]


@OBJECT_REGISTRY.register
class ANCBEV3DHead(nn.Module):
    """Bev3DHead module.

    Args:
        in_strides: A list contains the strides of feature maps
            from backbone or neck.
        out_strides: A list contains the strides of this head
            will output.
        in_channels: A list of to indicates the input channels of the
            block.
        head_channels: A dictionary contains output heads and
            corresponding channels.
        feature_name: Name of features from backbone(or neck)
            in input dict.
        sep_conv: Use separable convolution.
        use_varg: Use VarGNet block.
        stack: The number of VarGNet black stacked together.
        use_bias: Use bias.
        bn_kwargs: Batch norm arguments.
        dw_with_relu: A param for VarGNet block.
        pw_with_relu: A param for VarGNet block.
        factor: A param for VarGNet block.
        group_base: A param for VarGNet block.
        last_conv_kernel_size: Kernel size of last conv, default is 1.
        forward_frame_idx: when input multi frame features,
            select which frame idx to forward, default is 0.
            e.g., for bev3d training, backbone will output features of
            frame t and t+1, and bev3d head only output result for t,
            so we should set forward_frame_idx to 0.
        merge_block: Whether to merge block of different branchs,
            default is False.
        merge_output: Merge output_block at `int_infer` stage, only work at
            int_infer stage and merge_block=True.
        dequant_out: Whether dequant output or not, default is True.
            NOTE: dequant_out is False when some operation need to do in
            model after head, such as max in model after head, dequant_out
            need to be setted False at int_infer stage, the dequant is done
            after max.
        quant_config: Set the qconfig of output, like {"bev3d_hm": "qint16"},
            default is None.
        mid_channels: Feature channels of middel conv, default to in_channels.
    """

    def __init__(
        self,
        in_strides: List[int],
        out_strides: List[int],
        in_channels: int,
        head_channels: Dict,
        feature_name: str = None,
        sep_conv: bool = False,
        use_varg: bool = True,
        stack: int = 1,
        use_bias: bool = False,
        bn_kwargs: Optional[Dict] = None,
        dw_with_relu: bool = True,
        pw_with_relu: bool = False,
        factor: int = 2,
        group_base: int = 8,
        last_conv_kernel_size: int = 1,
        forward_frame_idx: int = 0,
        merge_block: bool = False,
        merge_output: bool = False,
        dequant_out: bool = True,
        quant_config: Dict[str, str] = None,
        mid_channels: int = -1,
    ):
        super(ANCBEV3DHead, self).__init__()

        self.in_strides = in_strides
        self.out_strides = out_strides
        self.head_channels = head_channels
        self.merge_block = merge_block
        bn_kwargs = (
            bn_kwargs
            if bn_kwargs is not None
            else {"eps": 1e-5, "momentum": 0.1}
        )
        self.in_channels = in_channels
        mid_channels = mid_channels if mid_channels > 0 else in_channels
        self.mid_channels = mid_channels
        self.last_conv_kernel_size = last_conv_kernel_size

        if merge_output:
            assert merge_block, "merge_block must be True."
        self.merge_output = merge_output
        self.merge_conv = None
        self.dequant_out = dequant_out
        if self.dequant_out:
            self.dequant = DeQuantStub()
        self.quant_config = quant_config

        if use_varg:
            self.head_block = BasicVarGBlock(
                in_channels=in_channels,
                mid_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                bn_kwargs=bn_kwargs,
                factor=factor,
                group_base=group_base,
                merge_branch=False,
                dw_with_relu=dw_with_relu,
                pw_with_relu=pw_with_relu,
            )
        else:
            group_num = int(in_channels / group_base)
            self.head_block = SeparableGroupConvModule2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=3,
                groups=group_num,
                padding=1,
                stride=1,
                pw_norm_layer=nn.BatchNorm2d(mid_channels),
                pw_act_layer=nn.ReLU(inplace=True),
            )

        if merge_block:
            encode_block = []
            if sep_conv:
                for _ in range(stack):
                    block = SeparableConvModule2d(
                        in_channels=mid_channels,
                        out_channels=mid_channels,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                        pw_norm_layer=nn.BatchNorm2d(mid_channels),
                        pw_act_layer=nn.ReLU(inplace=True),
                    )
                    encode_block.append(block)
            else:
                for _ in range(stack):
                    block = ConvModule2d(
                        mid_channels,
                        out_channels=mid_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=use_bias,
                        norm_layer=nn.BatchNorm2d(mid_channels),
                        act_layer=nn.ReLU(inplace=True),
                    )
                    encode_block.append(block)
            self.encode_block = nn.Sequential(*encode_block)

            for name, num_channel in head_channels.items():
                block = []
                sub_block = nn.Conv2d(
                    mid_channels,
                    num_channel,
                    last_conv_kernel_size,
                    1,
                    (last_conv_kernel_size - 1) // 2,
                    groups=1,
                    bias="hm" in name,
                )
                block.append(sub_block)
                block = nn.Sequential(*block)
                if "hm" in name:
                    bias = bias_init_with_prob(0.01)
                    normal_init(block[-1], std=0.01, bias=bias)
                setattr(self, "out_block_{}".format(name), block)
        else:
            for name, num_channel in head_channels.items():
                block = []
                for _ in range(stack):
                    if sep_conv:
                        sub_block = SeparableConvModule2d(
                            in_channels=mid_channels,
                            out_channels=mid_channels,
                            kernel_size=3,
                            padding=1,
                            stride=1,
                            pw_norm_layer=nn.BatchNorm2d(mid_channels),
                            pw_act_layer=nn.ReLU(inplace=True),
                        )
                    else:
                        sub_block = ConvModule2d(
                            mid_channels,
                            out_channels=mid_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False,
                            norm_layer=nn.BatchNorm2d(mid_channels),
                            act_layer=nn.ReLU(inplace=True),
                        )
                    block.append(sub_block)

                block.append(
                    nn.Conv2d(
                        mid_channels,
                        num_channel,
                        last_conv_kernel_size,
                        1,
                        (last_conv_kernel_size - 1) // 2,
                        groups=1,
                        bias="hm" in name,
                    )
                )
                block = nn.Sequential(*block)

                if "hm" in name:
                    bias = bias_init_with_prob(0.01)
                    normal_init(block[-1], std=0.01, bias=bias)
                setattr(self, "out_block_{}".format(name), block)

        self.feature_name = feature_name
        self.forward_frame_idx = forward_frame_idx

    def forward(
        self, data: Union[Mapping, Sequence]
    ) -> Mapping[str, torch.Tensor]:
        """Forward head layers."""
        input_features = (
            data[self.feature_name][self.forward_frame_idx]
            if isinstance(data, Mapping)
            else data[self.forward_frame_idx]
        )

        feat = _take_features(
            input_features, self.in_strides, self.out_strides
        )[0]
        feat = self.head_block(feat)
        if self.merge_block:
            feat = self.encode_block(feat)

        if self.merge_output:
            self._create_int_merge_conv()
            merge_out = self.dequant(self.merge_conv(feat))
            return merge_out

        out = OrderedDict()
        for name in self.head_channels.keys():
            block_name = "out_block_{}".format(name)
            block = getattr(self, block_name)
            if self.dequant_out:
                out[name] = self.dequant(block(feat))
            else:
                out[name] = block(feat)

        return out

    def _create_int_merge_conv(self):
        if self.merge_conv is not None:
            return
        for name in self.head_channels.keys():
            block_name = "out_block_{}".format(name)
            block = getattr(self, block_name)[0]
            assert (
                type(block) == horizon.nn.quantized.conv2d.Conv2d
            ), "merge_output only work at int_infer stage and \
                the op must be Conv2d"

        sum_channel = sum(self.head_channels.values())
        conv = nn.Conv2d(
            self.mid_channels,
            sum_channel,
            self.last_conv_kernel_size,
            stride=1,
            padding=(self.last_conv_kernel_size - 1) // 2,
            groups=1,
            bias=True,
        )
        nn.init.constant_(conv.bias, 0.0)
        conv.qconfig = qconfig_manager.get_default_qat_out_qconfig()

        conv = nn.Sequential(*[conv])
        horizon.quantization.prepare_qat(conv, inplace=True)
        horizon.quantization.convert(conv.eval(), inplace=True)
        conv = conv[0]

        idx = 0
        for name, chn in self.head_channels.items():
            block_name = "out_block_{}".format(name)
            block = getattr(self, block_name)[0]

            conv.weight.data[idx : (idx + chn), ...] = block.weight.data
            conv.weight_scale[idx : (idx + chn), ...] = block.weight_scale.data
            if hasattr(block, "bias") and block.bias is not None:
                conv.bias.data[idx : (idx + chn)] = block.bias.data
                conv.bias_scale.data[idx : (idx + chn)] = block.bias_scale.data
            idx += chn

        conv.to(block.weight.device)
        self.merge_conv = conv

    def fuse_model(self):
        self.head_block.fuse_model()
        if self.merge_block:
            assert isinstance(self.encode_block, nn.Sequential)
            for block in self.encode_block:
                if hasattr(block, "fuse_model"):
                    block.fuse_model()
        for name in self.head_channels.keys():
            block_name = "out_block_{}".format(name)
            out_block = getattr(self, block_name)
            assert isinstance(out_block, nn.Sequential)
            for block in out_block:
                if hasattr(block, "fuse_model"):
                    block.fuse_model()

    def set_qconfig(self):
        # disable output quantization for last quanti layer.
        for name in self.head_channels.keys():
            block_name = "out_block_{}".format(name)
            block = getattr(self, block_name)
            if self.quant_config and name in self.quant_config:
                block[
                    -1
                ].qconfig = horizon.quantization.get_default_qat_qconfig(
                    self.quant_config[name]
                )
            else:
                block[
                    -1
                ].qconfig = qconfig_manager.get_default_qat_out_qconfig()
            if self.merge_output:
                assert (
                    block[-1].qconfig.activation is None
                ), "Only support int32 output."


@OBJECT_REGISTRY.register
class ANCIdentityHead(nn.Module):
    """
    A head that outputs the input as-is.

    Args:
        dequant_out: Whether to dequant the output.
        feature_name: Name of features in input dict.
    """

    def __init__(
        self,
        dequant_out: bool = False,
        feature_name: str = None,
        **kwargs,
    ):
        super(ANCIdentityHead, self).__init__(**kwargs)
        self.dequant_out = dequant_out
        if self.dequant_out:
            self.dequant = DeQuantStub()
            self.feature_name = feature_name

    def forward(self, data):
        if self.dequant_out:
            if isinstance(data, (list, tuple)):
                output = []
                for i in range(len(data)):
                    output.append(self.dequant(data[i]))
            else:
                output = self.dequant(data)
            return output
        else:
            return data


@OBJECT_REGISTRY.register
class ANCBEVStaticHead(nn.Module):
    """
    BEVStaticHead Module.

    e.g. BEVseg, OnlineMapping ..

    Migrated from Pixelhead.
    Set align_channels to adjust the channels of the input feature
    Set merge_branches to share the same output_block for output

    Args:
        in_strides: a list contains the strides of feature maps
            from backbone or neck.
        out_strides: a list contains the strides of output.
        align_channels: A list containing the output channels.It can be used
        to adjust the channels of the input feature.
        stride2channels: a stride to channels dict.
        quanti_last_conv: whether output quantize result in last conv.
        dequant_out: whether dequanti output.
        out_nums: output dim of result.
        output_name: Name of result in result dict.
        feature_name: Name of features from backbone(or neck)
            in input dict.
        merge_branches: Whether to merge branches.
            If True, different branches share the same output_block,
            with outputs separated by out_nums.
            If False, different branches correspond to different output_blocks.
        bn_kwargs: kwargs of bn layer.
        use_bias: whether to use bias.
        factor: factor of Group Separable Conv.
        group_base: group_base of output_block, if > 1, use vargnet
            block.
    """

    def __init__(
        self,
        in_strides: int,
        out_strides: int,
        stride2channels: Mapping,
        bn_kwargs: Mapping,
        quanti_last_conv: bool = False,
        dequant_out: bool = True,
        dropout_ratio: float = 0.0,
        out_nums: Union[Sequence[int], int] = 1,
        output_name: Union[Sequence[str], str] = "pred_depths",
        feature_name: str = "feats",
        merge_branches: bool = False,
        align_channels: Mapping = None,
        factor: int = 2,
        use_bias: bool = False,
        group_base: int = 1,
        **kwargs,
    ):
        super(ANCBEVStaticHead, self).__init__(**kwargs)

        assert _is_increasing_sequence(in_strides), in_strides
        self.in_strides = _as_list(in_strides)

        assert _is_increasing_sequence(out_strides), out_strides
        self.out_strides = _as_list(out_strides)
        assert out_strides[0] >= self.in_strides[0], "%d vs. %d" % (
            out_strides[0],
            self.in_strides[0],
        )
        assert out_strides[-1] <= self.in_strides[-1], "%d vs. %d" % (
            out_strides[-1],
            self.in_strides[-1],
        )
        self.align_channels = align_channels
        self.do_align = True if align_channels else False
        self.stride2channels = stride2channels
        self.use_bias = use_bias
        self.bn_kwargs = bn_kwargs
        self.factor = factor
        self.dropout_ratio = dropout_ratio
        self.factor = factor
        self.group_base = group_base
        self.quanti_last_conv = quanti_last_conv
        self.dequant_out = dequant_out
        self.out_nums = _as_list(out_nums)
        self.output_name = _as_list(output_name)
        self.feature_name = feature_name
        self.merge_branches = merge_branches

        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self._init_align_convs()
        self._init_predictor()

    def _init_align_convs(self):
        """Initialize align layers of the head.It can be used\
        to adjust the channels of the input feature."""

        if self.do_align:
            assert len(self.align_channels) == len(self.out_strides)
            self.align_blocks = nn.ModuleList()
            for outstride, target_channel in zip(
                self.out_strides, self.align_channels
            ):
                old_channel = self.stride2channels[outstride]
                if old_channel != target_channel:
                    self.align_blocks.append(
                        ConvModule2d(
                            in_channels=old_channel,
                            out_channels=target_channel,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=self.use_bias,
                            norm_layer=nn.BatchNorm2d(
                                target_channel, **self.bn_kwargs
                            ),
                        )
                    )
                else:
                    self.align_blocks.append(nn.Identity())

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.output_blocks = nn.ModuleList()
        for idx, stride in enumerate(self.out_strides):
            in_channels = (
                self.stride2channels[stride]
                if not self.do_align
                else self.align_channels[idx]
            )
            out_channels = in_channels
            sub_output_blocks = nn.ModuleList()
            if self.merge_branches:
                out_nums_sum = sum(self.out_nums)
                sub_output_blocks.append(
                    OutputBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        dropout_ratio=self.dropout_ratio,
                        bn_kwargs=self.bn_kwargs,
                        use_bias=self.use_bias,
                        out_nums=out_nums_sum,
                        factor=self.factor,
                        group_base=self.group_base,
                        quanti_last_conv=self.quanti_last_conv,
                        dequant_out=self.dequant_out,
                    )
                )
            else:
                for out_nums_i in self.out_nums:
                    sub_output_blocks.append(
                        OutputBlock(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            dropout_ratio=self.dropout_ratio,
                            bn_kwargs=self.bn_kwargs,
                            use_bias=self.use_bias,
                            out_nums=out_nums_i,
                            factor=self.factor,
                            group_base=self.group_base,
                            quanti_last_conv=self.quanti_last_conv,
                            dequant_out=self.dequant_out,
                        )
                    )
            self.output_blocks.append(sub_output_blocks)

    def forward_single(self, feature, index, result):

        if self.do_align:
            align_conv = self.align_blocks[index]
            feature = align_conv(feature)

        blocks = self.output_blocks[index]
        if self.merge_branches:
            for sub_blocks in blocks:
                out = sub_blocks(feature)
                if len(self.output_name) == 1:
                    result[self.output_name[0]].append(out)
                else:
                    presum = np.cumsum([0] + self.out_nums)
                    begin_channel = presum[:-1]
                    end_channel = presum[1:]
                    for output_name, _begin, _end in zip(
                        self.output_name, begin_channel, end_channel
                    ):
                        result[output_name].append(out[:, _begin:_end])
        else:
            for output_name, sub_blocks in zip(self.output_name, blocks):
                out = sub_blocks(feature)
                result[output_name].append(out)

    def forward(self, data: Mapping):
        data = data[self.feature_name] if isinstance(data, Mapping) else data
        res = OrderedDict()
        features = _take_features(data[0], self.in_strides, self.out_strides)
        result = {output_name: [] for output_name in self.output_name}

        for index, feature in enumerate(features):
            self.forward_single(feature, index, result)

        for output_name in self.output_name:
            res["%s_frame0" % (output_name)] = result[output_name]
        return res

    def fuse_model(self):
        for m in self.output_blocks:
            for m_i in m:
                m_i.fuse_model()

    def set_qconfig(self):
        for m in self.output_blocks:
            for m_i in m:
                m_i.set_qconfig()


@OBJECT_REGISTRY.register
class ANCBevDiscobjHead(ANCBEV3DHead):
    """BevDiscobjHead module with feature mask.

    Args:
        feat_size: (h, w) feature size used to generate pre-definded feat mask.
        num_classes: Classes number.
        valid_range_percls: Defined the valid bev range for every class, the
            key is class id. such as.

            .. code-block:: none

                {
                    0: (0, 0, 110, 96), # (top, left, bottom, right)

                    1: (20, 40, 110, 80), # (top, left, bottom, right)

                }

        use_feat_mask: Whether use feature mask in heatmap. If the bev_range of
            each category is different, it is recommended to set this value to
            True, and feat mask will be generated by valid_range_percls which
            will be used to mask the class heatmap feature.

    """

    def __init__(
        self,
        feat_size: Tuple[int],
        num_classes: int,
        valid_range_percls: Optional[Dict] = None,
        use_feat_mask: bool = False,
        **kwargs,
    ):
        super(ANCBevDiscobjHead, self).__init__(**kwargs)
        self.use_feat_mask = (
            True if valid_range_percls is not None and use_feat_mask else False
        )
        if self.use_feat_mask:
            # mask_mul is used to set the out range feature to 0,
            # value in this mask is fixed
            self.feat_mask_mul = nn.Parameter(
                torch.zeros(
                    [1, num_classes, feat_size[0], feat_size[1]]
                ).float(),
                requires_grad=False,
            )
            # mask_add is used to set the out range feature to a mini value,
            # let the sigmoid probability value tend to 0
            self.feat_mask_add = nn.Parameter(
                torch.full(
                    [1, num_classes, feat_size[0], feat_size[1]], -10.0
                ).float(),
                requires_grad=False,
            )
            self._init_mask(valid_range_percls, num_classes, feat_size)
            self.mask_mul = hnn.quantized.FloatFunctional()
            self.mask_mul_quant = HQuantStub(scale=1.0)
            self.mask_add = hnn.quantized.FloatFunctional()
            self.mask_add_quant = HQuantStub(scale=1.0)

    def _init_mask(self, valid_range_percls, num_cls, feat_size):
        for cls_id in range(num_cls):
            # top, left, bottom, right
            lt_y, lt_x, rb_y, rb_x = valid_range_percls.get(
                cls_id, (0, 0, feat_size[0], feat_size[1])
            )
            # init mul feat mask, out range set 0, in range set 1
            self.feat_mask_mul[:, cls_id, lt_y:rb_y, lt_x:rb_x] = 1

            # init add feat mask, out range set -10 to make sigmoid
            # scale to 4.5e-5, smaller than 1e-4 which is the minimum
            # value in sigmoid_and_clip used in decoder
            self.feat_mask_add[:, cls_id, lt_y:rb_y, lt_x:rb_x] = 0.0

    def forward(
        self, data: Union[Mapping, Sequence]
    ) -> Mapping[str, torch.Tensor]:
        """Forward head layers."""
        input_features = (
            data[self.feature_name][self.forward_frame_idx]
            if isinstance(data, Mapping)
            else data[self.forward_frame_idx]
        )

        feat = _take_features(
            input_features, self.in_strides, self.out_strides
        )[0]
        if isinstance(feat, list):
            feat = feat[0]
        feat = self.head_block(feat)
        if self.merge_block:
            feat = self.encode_block(feat)

        if self.merge_output:
            self._create_int_merge_conv()
            merge_out = self.dequant(self.merge_conv(feat))
            return merge_out

        out = OrderedDict()
        for name in self.head_channels.keys():
            block_name = "out_block_{}".format(name)
            block = getattr(self, block_name)
            out_feat = block(feat)
            if "hm" in name and self.use_feat_mask:
                feat_mul_mask = self.mask_mul_quant(self.feat_mask_mul)
                feat_add_mask = self.mask_add_quant(self.feat_mask_add)
                out_feat = self.mask_add.add(
                    self.mask_mul.mul(out_feat, feat_mul_mask),
                    feat_add_mask,
                )
            if self.dequant_out:
                out_feat = self.dequant(out_feat)
            out[name] = out_feat

        return out

    def set_qconfig(self):
        if self.use_feat_mask:
            # FloatFunctional should set to qint16 for hm operation.
            self.mask_mul.qconfig = (
                horizon.quantization.get_default_qat_qconfig("qint16")
            )
            self.mask_add.qconfig = (
                horizon.quantization.get_default_qat_qconfig("qint16")
            )
        # disable output quantization for last quanti layer.
        for name in self.head_channels.keys():
            block_name = "out_block_{}".format(name)
            block = getattr(self, block_name)
            if self.quant_config and name in self.quant_config:
                block[
                    -1
                ].qconfig = horizon.quantization.get_default_qat_qconfig(
                    self.quant_config[name]
                )
            else:
                block[
                    -1
                ].qconfig = qconfig_manager.get_default_qat_out_qconfig()
            if self.merge_output:
                assert (
                    block[-1].qconfig.activation is None
                ), "Only support int32 output."


@OBJECT_REGISTRY.register
class ANCBEVOMHead(nn.Module):
    """
    BEVOMHead Module.

    Migrated from BEVStatichead.
    Set align_channels to adjust the channels of the input feature

    Args:
        in_strides: a list contains the strides of feature maps
            from backbone or neck.
        select_strides: a list contains the strides of selected feature maps.
        out_strides: a list contains the strides of output.
        stride2channels: a stride to channels dict.
        align_channels: used to adjust the channels of the input feature.
        quanti_last_conv: whether output quantize result in last conv.
        dequant_out: whether dequanti output.
        dropout_ratio: the ratio of dropout last output.
        out_nums: output dim of result.
        output_name: Name of result in result dict.
        feature_name: Name of features from backbone(or neck) in input dict.
        bn_kwargs: kwargs of bn layer.
        use_bias: whether to use bias.
        factor: factor of Group Separable Conv.
        group_base: group_base of output_block, if > 1, use vargnet block.
        repeat_times: the repeat times of OutputBlock.
        repeat_channels: the output channels of repeated layers.
        enhance_repeat_times: the repeat times of enhance_blocks.
    """

    def __init__(
        self,
        in_strides: Sequence[int],
        select_strides: Sequence[int],
        out_stride: int,
        stride2channels: Mapping,
        bn_kwargs: Mapping,
        quanti_last_conv: bool = False,
        dequant_out: bool = True,
        dropout_ratio: float = 0.0,
        out_nums: Union[Sequence[int], int] = 1,
        output_name: Union[Sequence[str], str] = "pred_depths",
        feature_name: Sequence[str] = None,
        align_channels: int = None,
        factor: int = 2,
        use_bias: bool = False,
        group_base: int = 1,
        repeat_times: int = 1,
        repeat_channels: int = None,
        enhance_repeat_times: int = 0,
        **kwargs,
    ):
        super(ANCBEVOMHead, self).__init__(**kwargs)
        self.in_strides = _as_list(in_strides)
        self.select_strides = _as_list(select_strides)
        self.out_stride = out_stride

        self.align_channels = align_channels
        self.do_align = True if align_channels else False
        self.stride2channels = stride2channels
        self.use_bias = use_bias
        self.bn_kwargs = bn_kwargs
        self.factor = factor
        self.dropout_ratio = dropout_ratio
        self.group_base = group_base
        self.quanti_last_conv = quanti_last_conv
        self.dequant_out = dequant_out
        self.out_nums = _as_list(out_nums)
        self.output_name = _as_list(output_name)
        self.feature_name = _as_list(feature_name)
        self.repeat_times = repeat_times
        self.repeat_channels = repeat_channels
        self.enhance_repeat_times = enhance_repeat_times

        if not isinstance(self.repeat_times, (list, tuple)):
            self.repeat_times = [self.repeat_times] * len(self.output_name)

        if len(self.select_strides) > 1:
            self.cat_op = nn.quantized.FloatFunctional()
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self._init_interpolate()
        self._init_align_convs()
        self._init_enhance_convs()
        self._init_predictor()

    def _init_interpolate(self):
        """Initialize interpolate layers of the head."""
        self.sample_layers = nn.ModuleList()
        out_stride = self.out_stride
        for stride in self.select_strides:
            upsample_factor = stride // out_stride
            if upsample_factor > 1:
                self.sample_layers.append(
                    hnn.Interpolate(
                        scale_factor=upsample_factor,
                        recompute_scale_factor=True,
                    )
                )
            elif upsample_factor == 1:
                self.sample_layers.append(nn.Identity())
            else:
                downsample_factor = out_stride // stride
                in_channel = self.stride2channels[stride]
                self.sample_layers.append(
                    ConvModule2d(
                        in_channels=in_channel,
                        out_channels=in_channel,
                        kernel_size=1,
                        stride=downsample_factor,
                        padding=0,
                        bias=self.use_bias,
                        norm_layer=nn.BatchNorm2d(
                            in_channel, **self.bn_kwargs
                        ),
                    )
                )

    def _init_align_convs(self):
        """Initialize align layers of the head.It can be used\
        to adjust the channels of the input feature."""

        if self.do_align:
            self.align_blocks = nn.ModuleList()
            old_channel = 0
            for stride in self.select_strides:
                old_channel += self.stride2channels[stride]
            self.align_blocks.append(
                ConvModule2d(
                    in_channels=old_channel,
                    out_channels=self.align_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=self.use_bias,
                    norm_layer=nn.BatchNorm2d(
                        self.align_channels, **self.bn_kwargs
                    ),
                    act_layer=nn.ReLU(inplace=True),
                )
            )

    def _init_enhance_convs(self):
        """Initialize enhance layers of the head.It can be used\
        to enhance the input feature."""

        enhance_blocks = nn.ModuleList()
        if self.do_align:
            in_channels = self.align_channels
        else:
            in_channels = 0
            for stride in self.select_strides:
                in_channels += self.stride2channels[stride]

        for _ in range(self.enhance_repeat_times):
            enhance_blocks.append(
                ConvModule2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=self.use_bias,
                    norm_layer=nn.BatchNorm2d(in_channels, **self.bn_kwargs),
                    act_layer=nn.ReLU(inplace=True),
                )
            )
            enhance_blocks.append(
                ConvModule2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=self.use_bias,
                    norm_layer=nn.BatchNorm2d(in_channels, **self.bn_kwargs),
                    act_layer=nn.ReLU(inplace=True),
                )
            )
        self.enhance_block = nn.Sequential(*enhance_blocks)

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.output_blocks = nn.ModuleList()
        if self.do_align:
            in_channels = self.align_channels
        else:
            in_channels = 0
            for stride in self.select_strides:
                in_channels += self.stride2channels[stride]
        sub_output_blocks = nn.ModuleList()
        for out_nums_i, repeat_i in zip(self.out_nums, self.repeat_times):
            out_channels = in_channels
            if repeat_i > 1 and self.repeat_channels is not None:
                out_channels = self.repeat_channels
            repeat_block = nn.ModuleList()
            for i in range(repeat_i - 1):
                in_channels_i = in_channels if i == 0 else out_channels
                repeat_block.append(
                    SeparableConvModule2d(
                        in_channels=in_channels_i,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=self.use_bias,
                        pw_norm_layer=nn.BatchNorm2d(
                            out_channels, **self.bn_kwargs
                        ),
                        pw_act_layer=nn.ReLU(inplace=True),
                    )
                )
            repeat_block.append(
                OutputBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    dropout_ratio=self.dropout_ratio,
                    bn_kwargs=self.bn_kwargs,
                    use_bias=self.use_bias,
                    out_nums=out_nums_i,
                    factor=self.factor,
                    group_base=self.group_base,
                    quanti_last_conv=self.quanti_last_conv,
                    dequant_out=self.dequant_out,
                )
            )
            if repeat_i > 1:
                repeat_block = nn.Sequential(*repeat_block)
            else:
                repeat_block = repeat_block[0]
            sub_output_blocks.append(repeat_block)
        self.output_blocks.append(sub_output_blocks)

    def fuse_feature(self, data: Mapping):
        features = []
        if isinstance(data, Mapping):
            if len(self.feature_name) > 1:
                for i in range(len(self.select_strides)):
                    stride = self.select_strides[i]
                    feature_name = self.feature_name[i]
                    feature = _take_features(
                        data[feature_name][0], stride, stride
                    )
                    features.append(self.sample_layers[i](feature[0]))
            else:
                feature_name = self.feature_name[0]
                feature_list = _take_features(
                    data[feature_name][0], self.in_strides, self.select_strides
                )
                for i in range(len(self.select_strides)):
                    features.append(self.sample_layers[i](feature_list[i]))
        else:
            feature_list = (
                _take_features(data[0], self.in_strides, self.select_strides)
                if len(data[0]) != len(self.select_strides)
                else data[0]
            )
            for i in range(len(self.select_strides)):
                stride = self.select_strides[i]
                features.append(self.sample_layers[i](feature_list[i]))

        if len(features) > 1:
            feature = self.cat_op.cat(features, dim=1)
        else:
            feature = features[0]
        return feature

    def forward(self, data: Mapping):
        res = OrderedDict()

        # fuse multi features with different scales
        feature = self.fuse_feature(data)
        if self.do_align:
            feature = self.align_blocks[0](feature)
        if self.enhance_repeat_times > 0:
            feature = self.enhance_block(feature)

        for output_name, sub_block in zip(
            self.output_name, self.output_blocks[0]
        ):
            out = sub_block(feature)
            res["%s_frame0" % (output_name)] = [out]

        return res

    def fuse_model(self):
        for m in self.output_blocks:
            for m_i in m:
                if isinstance(m_i, nn.Sequential):
                    for m_i_j in m_i:
                        m_i_j.fuse_model()
                else:
                    m_i.fuse_model()
        if self.do_align:
            for m in self.align_blocks:
                if isinstance(m, ConvModule2d):
                    m.fuse_model()
        if self.enhance_repeat_times > 0:
            for m in self.enhance_block:
                m.fuse_model()

    def set_qconfig(self):
        for m in self.output_blocks:
            for m_i in m:
                if isinstance(m_i, nn.Sequential):
                    for m_i_j in m_i:
                        if hasattr(m_i_j, "set_qconfig"):
                            m_i_j.set_qconfig()
                else:
                    m_i.set_qconfig()
