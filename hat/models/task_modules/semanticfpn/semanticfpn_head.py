# Copyright (c) Horizon Robotics. All rights reserved.

from numbers import Integral, Real
from typing import Dict, List, Optional, Union

import horizon_plugin_pytorch as horizon
import numpy as np
import torch
import torch.nn as nn
from horizon_plugin_pytorch.march import March, get_march

from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.base_modules.separable_conv_module import SeparableConvModule2d
from hat.models.weight_init import normal_init
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list


@OBJECT_REGISTRY.register
class SemanticFPNplusHead(nn.Module):
    """SemanticFPN head.

        Interpolate_C op in seg_convs is an equivalent replacement for
        horizon.nn.interporate, but it's implemented to be supported
        by torchdynamo.

    Args:
        num_classes: Number of classes.
        in_strides: The strides corresponding to the inputs, the inputs
            usually come from backbone or neck.
        in_channels: The channels corresponding to the inputs.
        out_stride: The stride corresponding to the output.
        feat_channels: Number of hidden channels (of each output stride).
            Default is 256.
        bn_kwargs: Extra keyword arguments for bn layers.
        use_dw_conv: Whether use depthwise sparable convolution module.
            Default is True.
        use_gn: Use Group Normalization instead of Batch Normalization.
            Default is True.
        dropout_ratio: Drop out ratio before seg_pred.
        upscale: If True, the first feature map is upsampled by 2x.
            Default is False.
        pixel_shuffle_factor: Upscale factor with pixel shuffle operator.
            Default is -1.
        argmax_output: Whether conduct argmax on output. Default: False
        upsample_output_scale: Output upsample scale, default is None.
        use_native_op: Use torch's native operators which is implemented to
            support torchdynamo when compiling models with trt and torchdynamo,
            since torch.nn.functional.interpoate is overridden by
            horizon_plugin_pytorch. Default: False.
    """

    def __init__(
        self,
        num_classes: int,
        in_strides: List[int],
        in_channels: List[int],
        out_stride: Optional[int] = None,
        feat_channels: Union[int, List[int]] = 256,
        bn_kwargs: Optional[Dict] = None,
        use_dw_conv: bool = True,
        use_gn: bool = True,
        dropout_ratio: float = -1,
        upscale: bool = False,
        pixel_shuffle_factor: int = -1,
        argmax_output: bool = False,
        upsample_output_scale: Optional[int] = None,
        use_native_op: bool = False,
    ):
        super(SemanticFPNplusHead, self).__init__()
        self.in_strides = sorted(_as_list(in_strides))
        self.in_channels = in_channels
        if isinstance(feat_channels, list):
            assert len(feat_channels) == 1, feat_channels
            feat_channels = feat_channels[0]
        self.feat_channel = feat_channels

        self.num_classes = num_classes
        self.bn_kwargs = bn_kwargs or {}

        self.upscale = upscale
        if self.upscale:
            if use_native_op:
                self.upsample = Interpolate_C(
                    scale_factor=(2, 2),
                    align_corners=False,
                    recompute_scale_factor=True,
                )
            else:
                self.upsample = horizon.nn.Interpolate(
                    scale_factor=2,
                    align_corners=False,
                    recompute_scale_factor=True,
                )
            self.in_strides = [self.in_strides[0] // 2] + self.in_strides
            self.in_channels = [self.in_channels[0]] + self.in_channels
        self.out_stride = out_stride or self.in_strides[0]
        self.pixel_shuffle_factor = pixel_shuffle_factor
        self.pixel_shuffle = pixel_shuffle_factor > 1

        self.strides_prefix = [f"stride{i}" for i in self.in_strides]
        self.use_dw_conv = use_dw_conv
        self.use_gn = use_gn
        self.dropout_ratio = dropout_ratio

        self.argmax_output = argmax_output
        self.upsample_output_scale = upsample_output_scale
        if upsample_output_scale:
            if use_native_op:
                self.resize = Interpolate_C(
                    scale_factor=self.upsample_output_scale,
                    align_corners=False,
                    recompute_scale_factor=True,
                )
            else:
                self.resize = horizon.nn.Interpolate(
                    scale_factor=self.upsample_output_scale,
                    align_corners=None,
                    recompute_scale_factor=True,
                )
        self.use_native_op = use_native_op
        self._init_layers()
        self._init_weights()

    def _make_conv(
        self,
        chn_in,
        chn_out,
        kernel_size,
        padding,
        stride: int = 1,
        inplace: bool = True,
    ):
        if self.use_dw_conv:
            return SeparableConvModule2d(
                chn_in,
                chn_out,
                kernel_size,
                stride=stride,
                padding=padding,
                pw_norm_layer=nn.GroupNorm(32, chn_out)
                if self.use_gn
                else nn.BatchNorm2d(self.feat_channel, **self.bn_kwargs),
                pw_act_layer=nn.ReLU(inplace=inplace),
            )
        else:
            return ConvModule2d(
                chn_in,
                chn_out,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
                norm_layer=nn.GroupNorm(32, chn_out)
                if self.use_gn
                else nn.BatchNorm2d(self.feat_channel, **self.bn_kwargs),
                act_layer=nn.ReLU(inplace=inplace),
            )

    def _init_layers(self):
        """Initialize layers of the head."""
        self._init_seg_convs()
        self._init_predictor()

    def _init_seg_convs(self):
        self.seg_convs = nn.ModuleDict()
        for ii, in_stride in enumerate(self.in_strides):
            seg_convs = nn.ModuleList()
            head_length = max(
                1, int(np.log2(in_stride) - np.log2(self.out_stride))
            )
            for i in range(head_length):
                chn = self.in_channels[ii] if i == 0 else self.feat_channel
                seg_convs.append(
                    self._make_conv(
                        chn,
                        self.feat_channel,
                        3,
                        padding=1,
                    )
                )
                if in_stride != self.out_stride:
                    if self.use_native_op:
                        seg_convs.append(
                            Interpolate_C(
                                scale_factor=(2, 2),
                                align_corners=False,
                                recompute_scale_factor=True,
                            )
                        )
                    else:
                        seg_convs.append(
                            horizon.nn.Interpolate(
                                scale_factor=2,
                                align_corners=False,
                                recompute_scale_factor=True,
                            )
                        )
            self.seg_convs[self.strides_prefix[ii]] = seg_convs

    def _init_predictor(self):
        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout2d(self.dropout_ratio)
        else:
            self.dropout = None

        if self.pixel_shuffle:
            self.cls_seg = nn.Sequential(
                ConvModule2d(
                    self.feat_channel,
                    self.num_classes * (self.pixel_shuffle_factor ** 2),
                    1,
                    norm_layer=None,
                    act_layer=None,
                ),
                nn.PixelShuffle(self.pixel_shuffle_factor),
            )
        else:
            self.cls_seg = ConvModule2d(
                self.feat_channel,
                self.num_classes,
                1,
                norm_layer=None,
                act_layer=None,
            )

    def _init_weights(self):
        """Initialize weights of the head."""
        for m in self.seg_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.cls_seg.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)

    def forward_single(self, x, stride_index: int = 0):
        """Forward features of a single scale level.

        Args:
            x (Tensor): feature maps of the specified stride.
            stride_index (int): stride index of input feature map.

        Returns:
            tuple: seg predictions of input feature maps.
        """
        seg_convs = self.seg_convs[self.strides_prefix[stride_index]]
        for seg_layer in seg_convs:
            x = seg_layer(x)

        return x

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.upscale:
            feats.insert(0, self.upsample(feats[0]))
        feats = feats[: len(self.in_strides)]

        for i, feat_i in enumerate(feats):
            if i == 0:
                seg_feat = self.forward_single(feat_i, i)
            else:
                seg_feat = seg_feat + self.forward_single(feat_i, i)

        if self.dropout is not None:
            seg_feat = self.dropout(seg_feat)
        seg_pred = self.cls_seg(seg_feat)

        if self.upsample_output_scale:
            seg_pred = self.resize(seg_pred)
        if self.argmax_output:
            seg_pred = seg_pred.argmax(dim=1, keepdim=True)

        return [seg_pred]


class Interpolate_C(torch.nn.Module):
    r"""Resize for float training.

    Support bilinear and nearest interpolate method and NCHW input.
    The behaviour is same as torch.nn.functional.interpolate except the default
    mode is 'bilinear', and it's used with torchdynamo

    Parameters
    ----------
    size : int or tuple of int, optional
        the output shape of resize: if int, the output shape is (size, size)
        else the output shape is (out_height, out_width), by default None
        size and scale_factor shouldn't be set at the same time
    scale_factor : float or tuple of float, optional
        the ratio of output shape to input shape, ie. out_shape / in_shape,
        or (out_height / in_height, out_width / in_width), by default None
        size and scale_factor shouldn't be set at the same time
    mode : str, optional
        the interpolate method, by default "bilinear",
        support "bilinear" and "nearest"
    align_corners : bool, optional
    recompute_scale_factor : bool, optional
        did not support, by default None
    antialias: bool, optional
        flag to apply anti-aliasing, not supported yet
    """

    def __init__(
        self,
        size=None,
        scale_factor=None,
        mode="bilinear",
        align_corners=None,
        recompute_scale_factor=None,
        antialias=False,
    ):
        super(Interpolate_C, self).__init__()
        assert not antialias, "antialias is not supported"
        assert isinstance(size, (Integral, type(None))) or (
            isinstance(size, (tuple, list))
            and len(size) == 2
            and isinstance(size[0], Integral)
            and isinstance(size[1], Integral)
        ), "param 'size' must be int or tuple of two int or None"
        assert isinstance(scale_factor, (Real, type(None))) or (
            isinstance(scale_factor, (tuple, list))
            and len(scale_factor) == 2
            and isinstance(scale_factor[0], Real)
            and isinstance(scale_factor[1], Real)
        ), "param 'scale_factor' must be real or tuple of two real or None"
        assert mode in (
            "bilinear",
            "nearest",
        ), "mode only support 'bilinear' and 'nearest'"
        if mode == "nearest":
            assert (
                align_corners is None
            ), "align_corners option can only be set with 'bilinear' mode"
        else:
            if align_corners is None:
                align_corners = False
            assert isinstance(
                align_corners, bool
            ), "param 'align_corners' must be bool or None"

        # only support align_corners=True on BAYES

        if get_march() == March.BERNOULLI2:
            assert (
                not align_corners
            ), "only support align_corners = True on BAYES"

        assert isinstance(
            recompute_scale_factor, (bool, type(None))
        ), "param 'recompute_scale_factor' must be bool or None"
        if scale_factor:
            assert (
                size is None
            ), "only one of size or scale_factor should be defined"
            assert recompute_scale_factor, (
                "only support recompute_scale_factor=True "
                + "when using scale_factor"
            )

        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def forward(self, data):
        return torch._C._nn.upsample_bilinear2d(
            data,
            output_size=self.size,
            scale_factors=self.scale_factor,
            align_corners=self.align_corners,
        )
