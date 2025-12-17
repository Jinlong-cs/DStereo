# Copyright (c) Horizon Robotics. All rights reserved.
# Source code reference to mmdetection

import warnings
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.base_modules.separable_conv_module import SeparableConvModule2d
from hat.models.task_modules.semanticfpn.semanticfpn_head import Interpolate_C
from hat.models.weight_init import bias_init_with_prob, normal_init
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import multi_apply
from .utils import generate_coordinate


class MaskFeatModule(nn.Module):
    """mask feature map branch.

    Args:
        in_channels: Number of channels in the input feature map.
        feat_channels: Number of hidden channels of the mask feature branch.
            Default: 128.
        start_level: The starting feature map level from FPN that will be used
            to predict the mask feature map. Default: 0.
        end_level: The ending feature map level from FPN that will be used to
            predict the mask feature map. Default: 3.
        out_channels: Number of output channels of the mask feature branch.
            This is the channel count of the mask feature map that to be
            dynamically convolved with the predicted kernel.
            Default: 256.
        upsample_mask_feat: 2x upsampling the first level feature map.
            Default: False.
        use_dw_conv: Whether use depthwise sparable convolution module.
            Default: True.
        stacked_convs: Number of stacking convs after the start_level.
            Default: 1.
    """

    def __init__(
        self,
        in_channels: int,
        feat_channels: int = 128,
        start_level: int = 0,
        end_level: int = 3,
        out_channels: int = 256,
        upsample_mask_feat: bool = False,
        use_dw_conv: bool = True,
        stacked_convs: int = 1,
        use_native_op: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.start_level = start_level
        self.end_level = end_level
        assert start_level >= 0 and end_level >= start_level
        self.upsample_mask_feat = upsample_mask_feat
        if self.upsample_mask_feat:
            assert start_level == 0, start_level
        self.stacked_convs = stacked_convs
        self.out_channels = out_channels
        self.use_dw_conv = use_dw_conv
        self.use_native_op = use_native_op
        self._init_layers()
        self._init_weights()

    def _init_layers(self):
        def make_conv(
            chn_in: int,
            chn_out: int,
            kernel_size: int,
            padding: int,
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
                    pw_norm_layer=nn.GroupNorm(32, chn_out),
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
                    norm_layer=nn.GroupNorm(32, chn_out),
                    act_layer=nn.ReLU(inplace=inplace),
                )

        self.convs_all_levels = nn.ModuleList()
        for i in range(self.start_level, self.end_level + 1):
            convs_per_level = nn.Sequential()
            if i == 0:
                if self.stacked_convs == 1:
                    chn_in = self.in_channels
                    chn_out = self.feat_channels
                    convs_per_level.add_module(
                        f"conv{i}",
                        make_conv(
                            chn_in,
                            chn_out,
                            3,
                            padding=1,
                        ),
                    )
                    self.convs_all_levels.append(convs_per_level)
                    continue
                else:
                    for k in range(self.stacked_convs):
                        if k == 0:
                            chn_in = self.in_channels + 2
                        else:
                            chn_in = self.feat_channels
                        chn_out = self.feat_channels
                        convs_per_level.add_module(
                            f"conv{i}_{k}",
                            make_conv(
                                chn_in,
                                chn_out,
                                3,
                                padding=1,
                            ),
                        )
                    self.convs_all_levels.append(convs_per_level)
                    continue

            for j in range(i):
                if j == 0:
                    if i == self.end_level:
                        chn = self.in_channels + 2
                    else:
                        chn = self.in_channels
                    convs_per_level.add_module(
                        f"conv{j}",
                        make_conv(
                            chn,
                            self.feat_channels,
                            3,
                            padding=1,
                        ),
                    )
                    if self.use_native_op:
                        convs_per_level.add_module(
                            f"upsample{j}",
                            Interpolate_C(
                                scale_factor=(2, 2),
                                align_corners=False,
                                recompute_scale_factor=True,
                            ),
                        )
                    else:
                        convs_per_level.add_module(
                            f"upsample{j}",
                            nn.Upsample(
                                scale_factor=2,
                                mode="bilinear",
                                align_corners=False,
                            ),
                        )
                    continue

                convs_per_level.add_module(
                    f"conv{j}",
                    make_conv(
                        self.feat_channels,
                        self.feat_channels,
                        3,
                        padding=1,
                    ),
                )
                if self.use_native_op:
                    convs_per_level.add_module(
                        f"upsample{j}",
                        Interpolate_C(
                            scale_factor=(2, 2),
                            align_corners=False,
                            recompute_scale_factor=True,
                        ),
                    )
                else:
                    convs_per_level.add_module(
                        f"upsample{j}",
                        nn.Upsample(
                            scale_factor=2,
                            mode="bilinear",
                            align_corners=False,
                        ),
                    )

            self.convs_all_levels.append(convs_per_level)
        if self.upsample_mask_feat:
            if self.use_native_op:
                self.upsample = Interpolate_C(
                    scale_factor=(2, 2),
                    align_corners=False,
                    recompute_scale_factor=True,
                )
            else:
                self.upsample = nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=False
                )

        self.conv_pred = make_conv(
            self.feat_channels,
            self.out_channels,
            1,
            padding=0,
        )

    def _init_weights(self):
        """Initialize weights of the head."""
        for layer in [self.convs_all_levels, self.conv_pred]:
            for m in layer.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        if self.upsample_mask_feat:
            upsampled_feature = self.upsample(feats[0])
            feats.insert(0, upsampled_feature)

        inputs = feats[self.start_level : self.end_level + 1]
        assert len(inputs) == (self.end_level - self.start_level + 1), (
            len(inputs),
            self.end_level,
            self.start_level,
        )

        for i in range(len(inputs)):
            input_p = inputs[i]
            if i == len(inputs) - 1:
                coord_feat = generate_coordinate(
                    input_p.size(), input_p.device
                )
                input_p = torch.cat([input_p, coord_feat], 1)

            if i == 0:
                feature_add_all_level = self.convs_all_levels[i](input_p)
            else:
                feature_add_all_level = (
                    feature_add_all_level + self.convs_all_levels[i](input_p)
                )

        feature_pred = self.conv_pred(feature_add_all_level)
        return feature_pred


@OBJECT_REGISTRY.register
class SOLOV2Head(nn.Module):
    """SOLOV2 with attribute prediction branches.

    Args:
        num_classes: Number of categories excluding the background category.
        in_channels: Number of channels in the input feature map.
        mask_feature_head_updater: Cfg dict, including configurations of the
            mask feature generation branch.
        mask_stride: Downsample factor of the mask feature map output.
            Default: 4.
        upsample_mask_logit: Learning the sampling factors for upsampling the
            mask logits to stride=1.
        feat_channels: Number of hidden channels. Used in child classes.
            Default: 256.
        feat_channels_attr: Number of hidden channels. Used in attribute
            classification branches. Default: 128.
        stacked_convs: Number of stacking convs of the head. Default: 3.
        num_levels: Number of FPN levels. Default: 5.
        num_grids: Divided image into a uniform grids, each feature map has a
            different grid value. The number of output channels is grid ** 2.
            Default: [40, 36, 24, 16, 12].
        kernel_out_channels: Number of output channels of the mask feature map
            branch. This is the channel count of the mask feature map that to
            be dynamically convolved with the predicted kernel.
        attr_name2num: Instance attributes to be predicted in parallel with
            classification and their corresponding numbers.
        use_dw_conv: Whether use depthwise sparable convolution module.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 256,
        mask_feature_head_updater: Optional[Dict] = None,
        mask_stride: int = 4,
        upsample_mask_logit: bool = False,
        feat_channels: int = 256,
        feat_channels_attr: int = 128,
        stacked_convs: int = 3,
        num_levels: int = 5,
        num_grids: Optional[Sequence[int]] = None,
        kernel_out_channels: int = 256,
        attr_name2num: Optional[Dict[str, int]] = None,
        use_dw_conv: bool = True,
        use_native_op: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cls_out_channels = self.num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.feat_channels_attr = feat_channels_attr
        self.stacked_convs = stacked_convs
        self.num_grids = num_grids if num_grids else [40, 36, 24, 16, 12]
        self.kernel_out_channels = kernel_out_channels
        self.attr_name2num = attr_name2num if attr_name2num else {}
        self.use_dw_conv = use_dw_conv
        self.upsample_mask_logit = upsample_mask_logit
        self.mask_stride = mask_stride
        self.use_native_op = use_native_op
        # number of FPN feats
        self.num_levels = num_levels
        self._init_layers()
        self._init_weights()

        mask_feature_head = {
            "feat_channels": 128,
            "start_level": 0,
            "end_level": 3,
            "out_channels": 256,
            "use_dw_conv": use_dw_conv,
        }
        if mask_feature_head_updater:
            mask_feature_head.update(mask_feature_head_updater)
        # update the in_channels of mask_feature_head
        if mask_feature_head.get("in_channels", None) is not None:
            if mask_feature_head["in_channels"] != self.in_channels:
                warnings.warn(
                    "The `in_channels` of SOLOV2MaskFeatHead and "
                    "SOLOV2Head should be same, changing "
                    "mask_feature_head.in_channels to "
                    f"{self.in_channels}"
                )
                mask_feature_head.update(in_channels=self.in_channels)
        else:
            mask_feature_head.update(in_channels=self.in_channels)

        self.mask_feature_head = MaskFeatModule(**mask_feature_head)

    def _init_layers(self):
        def make_conv(
            chn_in: int,
            chn_out: int,
            kernel_size: int,
            padding: int,
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
                    pw_norm_layer=nn.GroupNorm(32, chn_out),
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
                    norm_layer=nn.GroupNorm(32, chn_out),
                    act_layer=nn.ReLU(inplace=inplace),
                )

        self.cls_convs = nn.ModuleList()
        self.kernel_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels + 2 if i == 0 else self.feat_channels
            self.kernel_convs.append(
                make_conv(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                )
            )

            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                make_conv(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                )
            )

        self.conv_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1
        )

        self.conv_kernel = nn.Conv2d(
            self.feat_channels, self.kernel_out_channels, 3, padding=1
        )

        self.attr_convs = nn.ModuleDict()
        for attr_name, attr_num in self.attr_name2num.items():
            attr_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels_attr
                attr_convs.append(
                    make_conv(
                        chn,
                        self.feat_channels_attr,
                        3,
                        stride=1,
                        padding=1,
                    )
                )
            attr_convs.append(
                nn.Conv2d(
                    self.feat_channels_attr,
                    attr_num,
                    3,
                    padding=1,
                )
            )
            self.attr_convs[attr_name] = attr_convs

        if self.upsample_mask_logit:
            self.upsample_factor = nn.Sequential(
                nn.Conv2d(self.in_channels, self.feat_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    self.feat_channels,
                    9 * (self.mask_stride ** 2),
                    1,
                    padding=0,
                ),
            )

    def _init_weights(self):
        """Initialize weights of the head."""
        for layer in [self.cls_convs, self.kernel_convs, self.attr_convs]:
            for m in layer.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)

        if self.upsample_mask_logit:
            for m in self.upsample_factor.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        normal_init(self.conv_kernel, std=0.01)

    def resize_feats(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """Downsample the first feat and upsample last feat in feats."""
        out = []
        for i in range(len(feats)):
            if i == 0:
                if self.use_native_op:
                    out.append(
                        torch._C._nn.upsample_bilinear2d(
                            feats[0],
                            scale_factors=None,
                            output_size=feats[i + 1].shape[-2:],
                            align_corners=False,
                        )
                    )
                else:
                    out.append(
                        F.interpolate(
                            feats[0],
                            size=feats[i + 1].shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    )
            elif i == len(feats) - 1:
                if self.use_native_op:
                    out.append(
                        torch._C._nn.upsample_bilinear2d(
                            feats[i],
                            scale_factors=None,
                            output_size=feats[i - 1].shape[-2:],
                            align_corners=False,
                        )
                    )
                else:
                    out.append(
                        F.interpolate(
                            feats[i],
                            size=feats[i - 1].shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    )
            else:
                out.append(feats[i])
        return out

    def forward_attr_single(
        self, attr_name: str, attr_feat: torch.Tensor
    ) -> torch.Tensor:
        for attr_conv in self.attr_convs[attr_name]:
            attr_feat = attr_conv(attr_feat)
        return attr_feat

    def forward(self, feats: List[torch.Tensor]) -> Sequence[Any]:
        mask_feats = self.mask_feature_head(feats)
        (mlvl_kernel_preds, mlvl_cls_preds, mlvl_attr_preds) = multi_apply(
            self.forward_single,
            feats[: self.num_levels],
            self.num_grids,
        )
        if self.upsample_mask_logit:
            upsample_factors = 0.25 * self.upsample_factor(feats[0])
            return (
                mlvl_kernel_preds,
                mlvl_cls_preds,
                mlvl_attr_preds,
                mask_feats,
                upsample_factors,
            )
        else:
            return (
                mlvl_kernel_preds,
                mlvl_cls_preds,
                mlvl_attr_preds,
                mask_feats,
                [None] * len(mask_feats),
            )

    def forward_single(
        self, ins_kernel_feat: torch.Tensor, num_grid: int
    ) -> Sequence[Optional[torch.Tensor]]:
        # ins branch
        if self.use_native_op:
            lvl_feat = torch._C._nn.upsample_bilinear2d(
                ins_kernel_feat,
                scale_factors=None,
                output_size=(num_grid, num_grid),
                align_corners=False,
            )
        else:
            lvl_feat = F.interpolate(
                ins_kernel_feat,
                size=num_grid,
                mode="bilinear",
                align_corners=False,
            )

        # concat coord
        coord_feat = generate_coordinate(lvl_feat.size(), lvl_feat.device)
        kernel_feat = torch.cat([lvl_feat, coord_feat], 1)

        # kernel branch
        kernel_feat = kernel_feat.contiguous()
        for kernel_conv in self.kernel_convs:
            kernel_feat = kernel_conv(kernel_feat)
        kernel_pred = self.conv_kernel(kernel_feat)

        # cate branch
        cate_feat = lvl_feat.contiguous()
        for cls_conv in self.cls_convs:
            cate_feat = cls_conv(cate_feat)
        cate_pred = self.conv_cls(cate_feat)

        # attr branches
        if self.attr_name2num:
            lvl_attr_preds = multi_apply(
                self.forward_attr_single,
                self.attr_name2num.keys(),
                attr_feat=lvl_feat.contiguous(),
            )
        else:
            lvl_attr_preds = []

        return kernel_pred, cate_pred, lvl_attr_preds
