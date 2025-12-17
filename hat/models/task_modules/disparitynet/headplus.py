# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Dict, List

import horizon_plugin_pytorch as hpp
import torch
import torch.nn as nn
import torch.nn.functional as F
from horizon_plugin_pytorch.dtype import qint16, qint8
from horizon_plugin_pytorch.quantization import QuantStub
from horizon_plugin_pytorch import quantization
import horizon_plugin_pytorch.nn as hnn

from torch import Tensor
from torch.quantization import DeQuantStub

from hat.models.base_modules.basic_resnet_module import BasicResBlock
from hat.models.base_modules.conv_module import (
    ConvModule2d,
)
from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

__all__ = ["DisparityNetHead"]


# class DeConvResModule(nn.Module):
#     """
#     A basic module for deconv shortcut.

#     Args:
#         in_channels: The channels of inputs.
#         out_channels:  The channels of outputs.
#         bn_kwargs: Dict for BN layer.
#         kernel: The kernel_size of deconv.
#     """

#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         bn_kwargs: Dict = None,
#         kernel: int = 4,
#     ):
#         super(DeConvResModule, self).__init__()

#         self.conv1 = ConvTransposeModule2d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=kernel,
#             stride=2,
#             padding=1,
#             bias=False,
#             norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
#             act_layer=nn.ReLU(inplace=True),
#         )

#         self.conv2 = nn.Sequential(
#             ConvModule2d(
#                 in_channels=out_channels,
#                 out_channels=out_channels,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#                 bias=False,
#                 norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
#                 act_layer=nn.ReLU(inplace=True),
#             )
#         )
#         self.func_op = hpp.nn.quantized.FloatFunctional()

#     def forward(self, x: Tensor, rem: Tensor) -> Tensor:
#         """Perform the forward pass of the model."""

#         x = self.conv1(x)
#         x = self.func_op.add(x, rem)
#         x = self.conv2(x)
#         return x


# class UnfoldConv(nn.Module):
#     """
#     A unfold module using conv.

#     Args:
#         in_channels: The channels of inputs.
#         kernel_size: The kernel_size of unfold.
#     """

#     def __init__(self, in_channels: int = 1, kernel_size: int = 2):
#         super(UnfoldConv, self).__init__()
#         self.kernel_size = kernel_size
#         self.conv = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=self.kernel_size ** 2,
#             kernel_size=self.kernel_size,
#             stride=1,
#             bias=False,
#         )
#         self.pad = nn.ZeroPad2d(padding=(1, 0, 1, 0))
#         self.init_weights()

#     def init_weights(self) -> None:
#         """Initialize the weights of head module."""

#         weight_new = torch.zeros(
#             self.conv.weight.size(), dtype=self.conv.weight.dtype
#         )
#         for i in range(self.kernel_size ** 2):
#             wx = i % self.kernel_size
#             wy = i // self.kernel_size

#             if wx < self.kernel_size / 2:
#                 if wy < self.kernel_size / 2:
#                     weight_new[i, :, 0, 0] = 1
#                 else:
#                     weight_new[i, :, 1, 0] = 1
#             else:
#                 if wy < self.kernel_size / 2:
#                     weight_new[i, :, 0, 1] = 1
#                 else:
#                     weight_new[i, :, 1, 1] = 1

#         self.conv.weight = torch.nn.Parameter(weight_new, requires_grad=False)

#     def forward(self, x: Tensor) -> Tensor:
#         """Perform the forward pass of the model."""

#         x = self.pad(x)
#         x = self.conv(x)
#         return x

#     def set_qconfig(self) -> None:
#         """Set the quantization configuration."""

#         from hat.utils import qconfig_manager

#         self.pad.qconfig = qconfig_manager.get_qconfig(
#             activation_qat_qkwargs={"dtype": qint16},
#             activation_calibration_qkwargs={
#                 "dtype": qint16,
#             },
#             activation_calibration_observer="mix",
#         )
#         self.conv.qconfig = qconfig_manager.get_qconfig(
#             activation_fake_quant=None,
#             activation_qat_observer=None,
#             activation_qat_qkwargs=None,
#             activation_calibration_observer=None,
#             activation_calibration_qkwargs=None,
#             weight_qat_qkwargs={
#                 "qscheme": torch.per_channel_symmetric,
#                 "ch_axis": 0,
#                 "averaging_constant": 1,
#             },
#             weight_calibration_qkwargs={
#                 "qscheme": torch.per_channel_symmetric,
#                 "ch_axis": 0,
#                 "averaging_constant": 1,
#             },
#         )

#     def fix_weight_qscale(self) -> None:
#         """Fix the qscale of conv weight when calibration or qat stage."""

#         self.conv.weight_fake_quant.disable_observer()
#         self.conv.weight_fake_quant.set_qparams(
#             torch.ones(
#                 self.conv.weight.shape[0], device=self.conv.weight.device
#             )
#         )


class EdgeAwareRefinement(nn.Module):
    """
    A Refinement module of Stereonet.

    Args:
        in_channel: Channels of featmap.
        bn_kwargs: Dict for BN layer.
        num_res: Number of res block.
        is_last: Whether is the last refinement layer.
    """

    def __init__(
        self,
        in_channel: int,
        bn_kwargs: Dict = None,
        num_res: int = 6,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.conv2d_feature = nn.Sequential(
            ConvModule2d(
                in_channels=in_channel,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm_layer=nn.BatchNorm2d(32, **bn_kwargs),
                act_layer=nn.ReLU(inplace=True),
            )
        )
        self.residual_astrous_blocks = nn.ModuleList()
        self.num_res = num_res
        for _ in range(self.num_res):
            self.residual_astrous_blocks.append(
                BasicResBlock(
                    32,
                    32,
                    stride=1,
                    bias=False,
                    bn_kwargs=bn_kwargs,
                )
            )

        self.conv2d_out = ConvModule2d(
            in_channels=32,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm_layer=nn.BatchNorm2d(1, **bn_kwargs),
            act_layer=None,
        )
        self.act = nn.ReLU(inplace=True)
        self.cat_img = hpp.nn.quantized.FloatFunctional()
        self.res_add = hpp.nn.quantized.FloatFunctional()
        self.mul_s = hpp.nn.quantized.FloatFunctional()

    def forward(
        self,
        low_disparity: Tensor,
        corresponding_rgb: Tensor,
    ) -> Tensor:
        """
        Forward pass of the module to get disparity or offsets.

        Args:
            low_disparity: Input low-resolution disparity map.
            corresponding_rgb: Corresponding left image.

        """
        low_disparity = self.mul_s.mul_scalar(low_disparity, 1.0)

        twice_disparity = F.interpolate(
            low_disparity,
            size=corresponding_rgb.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        cat_disp = self.cat_img.cat(
            [twice_disparity, corresponding_rgb], dim=1
        )
        output = self.conv2d_feature(cat_disp)
        for astrous_block in self.residual_astrous_blocks:
            output = astrous_block(output)
        offset = self.conv2d_out(output)

        if self.is_last:
            return offset
        else:
            new_disp = self.res_add.add(offset, twice_disparity)
            return self.act(new_disp)

    def fuse_model(self) -> None:
        """Perform model fusion on the specified modules within the class."""

        modules = [self.conv2d_feature, self.residual_astrous_blocks]
        for m in modules:
            for m_ in m:
                if hasattr(m_, "fuse_model"):
                    m_.fuse_model()
        if self.is_last:
            torch.quantization.fuse_modules(
                self,
                ["conv2d_out.0", "conv2d_out.1"],
                inplace=True,
                fuser_func=quantization.fuse_known_modules,
            )
        else:
            torch.quantization.fuse_modules(
                self,
                ["conv2d_out.0", "conv2d_out.1", "res_add", "act"],
                inplace=True,
                fuser_func=quantization.fuse_known_modules,
            )

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""

        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        m_list = [
            self.cat_img,
            self.conv2d_out,
            self.res_add,
            self.act,
        ]

        for m in m_list:
            m.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={
                    "dtype": qint16,
                },
                activation_calibration_observer="mix",
            )
        self.conv2d_feature.qconfig = qconfig_manager.get_qconfig(
            activation_qat_qkwargs={"dtype": qint8},
            activation_calibration_qkwargs={
                "dtype": qint8,
            },
            activation_calibration_observer="mix",
        )
        for m in self.residual_astrous_blocks:
            m.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint8},
                activation_calibration_qkwargs={
                    "dtype": qint8,
                },
                activation_calibration_observer="mix",
            )
        if self.is_last:
            self.conv2d_out.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
            

class AdaptiveAggregationModule(nn.Module):
    """
    Adaptive aggregation module for optimizing disparity.

    Args:
        num_scales: The num of cost volume.
        num_output_branches:  The num branch for outputs.
        max_disp: The max value of disparity.
        num_blocks: The num of block.
    """

    def __init__(
        self,
        num_scales: int,
        num_output_branches: int,
        max_disp: int,
        num_blocks: int = 1,
    ):
        super(AdaptiveAggregationModule, self).__init__()

        self.num_scales = num_scales
        self.num_output_branches = num_output_branches
        self.max_disp = max_disp
        self.num_blocks = num_blocks

        self.branches = nn.ModuleList()

        # Adaptive intra-scale aggregation
        for i in range(self.num_scales):
            num_candidates = max_disp // (2 ** i)
            branch = nn.ModuleList()
            for _ in range(num_blocks):
                # if simple_bottleneck:
                branch.append(
                    BasicResBlock(num_candidates, num_candidates, bn_kwargs={})
                )
            self.branches.append(nn.Sequential(*branch))

        self.fuse_layers = nn.ModuleList()

        # Adaptive cross-scale aggregation
        # For each output branch
        for i in range(self.num_output_branches):
            self.fuse_layers.append(nn.ModuleList())
            # For each branch (different scale)
            for j in range(self.num_scales):
                if i == j:
                    # Identity
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                            ConvModule2d(
                                in_channels=max_disp // (2 ** j),
                                out_channels=max_disp // (2 ** i),
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False,
                                norm_layer=nn.BatchNorm2d(
                                    max_disp // (2 ** i)
                                ),
                                act_layer=None,
                            )
                        ),
                    )
                elif i > j:
                    layers = nn.ModuleList()
                    for _ in range(i - j - 1):
                        layers.append(
                            nn.Sequential(
                                ConvModule2d(
                                    in_channels=max_disp // (2 ** j),
                                    out_channels=max_disp // (2 ** j),
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    bias=False,
                                    norm_layer=nn.BatchNorm2d(
                                        max_disp // (2 ** j)
                                    ),
                                    act_layer=nn.ReLU(inplace=True),
                                )
                            )
                        )

                    layers.append(
                        nn.Sequential(
                            ConvModule2d(
                                in_channels=max_disp // (2 ** j),
                                out_channels=max_disp // (2 ** i),
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False,
                                norm_layer=nn.BatchNorm2d(
                                    max_disp // (2 ** i)
                                ),
                                act_layer=None,
                            )
                        )
                    )
                    self.fuse_layers[-1].append(nn.Sequential(*layers))

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.fuse_add = nn.ModuleList()
        for _ in range(len(self.fuse_layers) * len(self.branches)):
            self.fuse_add.append(hpp.nn.quantized.FloatFunctional())

    @fx_wrap()
    def update_idx(self, idx: int) -> int:
        """Update the idx."""

        return idx + 1

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        """Perform the forward pass of the model.

        Args:
            x: The inputs pyramid costvolume.

        Returns:
            x_fused: The fused pyramid costvolume.
        """

        assert len(self.branches) == len(x)

        for i in range(len(self.branches)): # 多个level共享的
            branch = self.branches[i]
            for j in range(self.num_blocks):
                dconv = branch[j]
                x[i] = dconv(x[i])          # residual,上一步是concate

        x_fused = []
        idx = 0
        for i in range(len(self.fuse_layers)):
            for j in range(len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    exchange = self.fuse_layers[i][j](x[j])
                    x_fused[i] = self.interpolate_exchange(
                        x_fused, exchange, i, idx
                    )
                    idx = self.update_idx(idx)
        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])

        return x_fused

    @fx_wrap()
    def interpolate_exchange(
        self, x_fused: Tensor, exchange: Tensor, i: int, idx: int
    ) -> Tensor:
        """Unsample costvolume and fuse."""

        if exchange.size()[2:] != x_fused[i].size()[2:]:
            exchange = F.interpolate(
                exchange,
                size=x_fused[i].size()[2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.fuse_add[idx].add(exchange, x_fused[i])


@OBJECT_REGISTRY.register
class DisparityNetHead(nn.Module):
    """
    An advanced head for StereoNet.

    Args:
        maxdisp: The max value of disparity.
        refine_levels:  Number of refinement layers.
        bn_kwargs: Dict for BN layer.
        max_stride: The max stride for model input.
        num_costvolume: The number of pyramid costvolume.
        num_fusion: The number of fusion module.
        hidden_dim: The hidden dim.
        in_channels: The channels of input features.
    """

    def __init__(
        self,
        maxdisp: int = 320,
        refine_levels: int = 3,
        bn_kwargs: Dict = None,
        max_stride: int = 32,
        num_costvolume: int = 3,
        num_fusion: int = 6,
        hidden_dim: int = 16,
        in_channels: List[int] = (32, 32, 16, 16, 16),
    ):
        super(DisparityNetHead, self).__init__()
        self.maxdisp = maxdisp
        self.refine_levels = refine_levels
        self.num_costvolume = num_costvolume
        self.D = self.maxdisp // max_stride
        self.num_fusion = num_fusion
        self.gc_pad = nn.ModuleList()
        self.gc_mean = nn.ModuleList()
        self.gc_mul = nn.ModuleList()
        self.hidden_dim = hidden_dim
        for k in range(num_costvolume):
            scale_tmp = pow(2, k)
            for i in range(self.D * scale_tmp):
                self.gc_pad.append(nn.ZeroPad2d(padding=(i, 0, 0, 0)))
                self.gc_mean.append(hpp.nn.quantized.FloatFunctional())
                self.gc_mul.append(hpp.nn.quantized.FloatFunctional())

        self.gc_cat_final = nn.ModuleList()

        for _ in range(self.refine_levels):
            self.gc_cat_final.append(hpp.nn.quantized.FloatFunctional())

        # self.softmax2 = nn.Softmax(dim=1)
        # self.softmax2.min_sub_out = -12.0

        low_disp_max = self.D * pow(2, self.num_costvolume - 1)

        self.downsample = nn.ModuleList()
        for i in range(self.refine_levels - 1, -1, -1):
            self.downsample.append(
                hnn.Interpolate(
                    scale_factor=1 / pow(2, i),
                    mode="bilinear",
                    recompute_scale_factor=True,
                )
            )

        self.edge_aware_refinements = nn.ModuleList()
        for i in range(self.refine_levels):
            if i == self.refine_levels - 1:
                self.edge_aware_refinements.append(
                    EdgeAwareRefinement(4, bn_kwargs, is_last=True)
                )
            else:
                self.edge_aware_refinements.append(
                    EdgeAwareRefinement(4, bn_kwargs)
                )

        self.fusions = nn.ModuleList()
        for i in range(self.num_fusion):
            num_out_branches = 1 if i == self.num_fusion - 1 else 3
            self.fusions.append(
                AdaptiveAggregationModule(
                    self.num_costvolume, num_out_branches, low_disp_max
                )
            )

        self.final_conv = ConvModule2d(
            low_disp_max,
            low_disp_max,
            kernel_size=1,
        )
        self.disp_mul_op = hpp.nn.quantized.FloatFunctional()
        self.disp_sum_op = hpp.nn.quantized.FloatFunctional()
        self.quant_img = QuantStub(scale=1.0 / 128.0)
        self.quant_dispvalue = QuantStub()
        self.softmax = nn.Softmax(dim=1)
        self.softmax.min_sub_out = -12.0
        self.disp_values = nn.Parameter(
            torch.range(0, low_disp_max - 1).view(1, low_disp_max, 1, 1),
            requires_grad=False,
        )
        # self.spx_8 = nn.Sequential(
        #     ConvModule2d(
        #         in_channels=self.hidden_dim,
        #         out_channels=self.hidden_dim,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         bias=False,
        #         norm_layer=nn.BatchNorm2d(self.hidden_dim, **bn_kwargs),
        #         act_layer=nn.ReLU(inplace=True),
        #     ),
        #     ConvModule2d(
        #         in_channels=self.hidden_dim,
        #         out_channels=self.hidden_dim,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         bias=False,
        #         norm_layer=nn.BatchNorm2d(self.hidden_dim, **bn_kwargs),
        #     ),
        # )

        # self.spx_4 = DeConvResModule(
        #     self.hidden_dim, self.hidden_dim, bn_kwargs
        # )
        # self.spx_2 = DeConvResModule(
        #     self.hidden_dim, self.hidden_dim, bn_kwargs
        # )
        # self.spx = ConvTransposeModule2d(
        #     self.hidden_dim,
        #     4,
        #     kernel_size=4,
        #     stride=2,
        #     padding=1,
        #     norm_layer=nn.BatchNorm2d(4, **bn_kwargs),
        #     act_layer=nn.ReLU(inplace=True),
        # )
        # self.spx_conv3x3 = ConvModule2d(
        #     in_channels=4,
        #     out_channels=4,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1,
        #     bias=False,
        # )
        self.mod1 = ConvModule2d(
            in_channels=in_channels[0],
            out_channels=self.hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
            norm_layer=nn.BatchNorm2d(self.hidden_dim, **bn_kwargs),
        )
        self.mod2 = ConvModule2d(
            in_channels=in_channels[1],
            out_channels=self.hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
            norm_layer=nn.BatchNorm2d(self.hidden_dim, **bn_kwargs),
        )
        # self.unfold = UnfoldConv()
        self.dequant = DeQuantStub()

    @fx_wrap()
    def get_l_img(self, img: Tensor, B: int) -> Tensor:
        """Get left featuremaps.

        Args:
            img: The inputs featuremaps.
            B: Batchsize.

        """

        return img[: B // 2]

    @fx_wrap()
    def dis_mul(self, x: Tensor) -> Tensor:
        """Mul weight to the disparity."""

        disp_values = self.quant_dispvalue(self.disp_values)
        return self.disp_mul_op.mul(x, disp_values)

    @fx_wrap()
    def dis_sum(self, x: Tensor) -> Tensor:
        """Get the low disparity."""
        return self.disp_sum_op.sum(x, dim=1, keepdim=True)

    @fx_wrap()
    def build_aanet_volume(self, refimg_fea, maxdisp, offset, idx):
        """
        Build the cost volume using the same approach as AANet.

        Args:
            refimg_fea: Featuremaps.
            maxdisp: Maximum disparity value.
            offset: The offset of gc_mul and gc_mean floatFunctional.
            idx: The idx of cat floatFunctional.

        Returns:
            volume: Costvolume.
        """

        B, C, H, W = refimg_fea.shape
        num_sample = B // 2
        tmp_volume = []
        for i in range(maxdisp):
            if i > 0:
                cost = self.gc_mul[i + offset].mul(
                    refimg_fea[:num_sample, :, :, i:],
                    refimg_fea[num_sample:, :, :, :-i],
                )
                tmp = self.gc_mean[i + offset].mean(cost, dim=1)
                tmp_volume.append(self.gc_pad[i + offset](tmp))
            else:
                cost = self.gc_mul[i + offset].mul(
                    refimg_fea[:num_sample, :, :, :],
                    refimg_fea[num_sample:, :, :, :],
                )
                tmp = self.gc_mean[i + offset].mean(cost, dim=1)
                tmp_volume.append(tmp)

        volume = (
            self.gc_cat_final[idx]
            .cat(tmp_volume, dim=1)
            .view(num_sample, maxdisp, H, W)
        )
        return volume

    @fx_wrap()
    def get_offset(self, offset: int, idx: int) -> int:
        """Get offset of floatFunctional."""
        return offset + self.D * (2 ** idx)

    def forward(self, features_inputs: List[Tensor], imgs: Tensor, early_return=False) -> List[Tensor]:
        """Perform the forward pass of the model.

        Args:
            features: The inputs featuremaps.

        Returns:
            pred0: The low disparity.
            pred0_unfold: The low disparity after unfold.
            spx_pred: The weight of each point.
        """

        #refinement
        B, _, _, _ = imgs.shape
        imgs = self.quant_img(imgs)
        imgl = self.get_l_img(imgs, B)
        # print(imgl.size())

        # features_inputs[0] = self.mod1(features_inputs[0])  # 没用到
        # features_inputs[1] = self.mod2(features_inputs[1])

        # Build cost volume as AANet
        features = features_inputs[-3:][::-1]
        aanet_volumes = []
        offset = 0
        for i in range(len(features)):
            aanet_volume = self.build_aanet_volume(
                features[i], self.D * (2 ** i), offset, i
            )
            offset = self.get_offset(offset, i)
            aanet_volumes.append(aanet_volume)

        # Fusion costvolume as AANet
        aanet_volumes = aanet_volumes[::-1]
        for i in range(len(self.fusions)):
            fusion = self.fusions[i]
            aanet_volumes = fusion(aanet_volumes)

        cost0 = self.final_conv(aanet_volumes[0])

        # Obtain low disparity and unfold it.
        pred0 = self.softmax(cost0)
        pred0 = self.dis_mul(pred0)
        pred0 = self.dis_sum(pred0)
        if early_return:
            return cost0, pred0
        # pred0_unfold = self.unfold(pred0)

        img_pyramid_list = []
        for i in range(self.refine_levels):
            img_pyramid_list.append(self.downsample[i](imgl))

        pred_pyramid_list = [pred0]

        # Refinement disparity
        for i in range(self.refine_levels):
            pred_new = self.edge_aware_refinements[i](
                pred_pyramid_list[i], img_pyramid_list[i]
            )
            pred_pyramid_list.append(pred_new)

        length_all = len(pred_pyramid_list)

        for i in range(length_all):
            pred_pyramid_list[i] = self.dequant(pred_pyramid_list[i])
        return pred_pyramid_list

        # Obtain weight of each point as Coex.

        # B, _, _, _ = features_inputs[0].shape
        # xspx = self.spx_8(self.get_l_img(features_inputs[2], B))

        # feature1_l = self.get_l_img(features_inputs[1], B)

        # xspx = self.spx_4(xspx, feature1_l)

        # feature0_l = self.get_l_img(features_inputs[0], B)

        # xspx = self.spx_2(xspx, feature0_l)
        # spx_pred = self.spx(xspx)
        # spx_pred = self.spx_conv3x3(spx_pred)
        # spx_pred = self.softmax2(spx_pred)

        # return (
        #     self.dequant(pred0),
        #     self.dequant(pred0_unfold),
        #     self.dequant(spx_pred),
        # )

    def upsample_disp(self, pred0, imgl):
        img_pyramid_list = []
        for i in range(self.refine_levels):
            img_pyramid_list.append(self.downsample[i](imgl))

        pred_pyramid_list = [pred0]

        # Refinement disparity
        for i in range(self.refine_levels):
            pred_new = self.edge_aware_refinements[i](
                pred_pyramid_list[i], img_pyramid_list[i]
            )
            pred_pyramid_list.append(pred_new)

        length_all = len(pred_pyramid_list)

        for i in range(length_all):
            pred_pyramid_list[i] = self.dequant(pred_pyramid_list[i])
        # return pred_pyramid_list
        disp_tmp = F.interpolate(
            pred_pyramid_list[-2],
            size=pred_pyramid_list[-1].shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        pred_disps = F.relu(disp_tmp + pred_pyramid_list[-1])
        return pred_disps.squeeze(1)
        pred_pyramid_list[-1] = F.relu(disp_tmp + pred_pyramid_list[-1])
        for i in range(len(pred_pyramid_list)):
            pred_disp_w = pred_pyramid_list[i].size()[-1]
            gt_disp_w = imgl.size()[-1]
            # pred_pyramid_list[i] = pred_pyramid_list[i] * self.maxdisp
            if pred_disp_w != gt_disp_w:
                pred_pyramid_list[i] = F.interpolate(
                    pred_pyramid_list[i], size=imgl.shape[2:], mode="bilinear"
                )
            pred_pyramid_list[i] = pred_pyramid_list[i].squeeze(1)

        return pred_pyramid_list
    
    def fuse_model(self) -> None:
        """Perform model fusion on the specified modules within the class."""

        for module in self.edge_aware_refinements:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""

        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        module_list = [
            self.disp_mul_op,
            self.disp_sum_op,
            self.final_conv,
            self.quant_dispvalue,
            self.softmax,
        ]
        for m in module_list:
            m.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={
                    "dtype": qint16,
                },
                activation_calibration_observer="mix",
            )
        # self.unfold.set_qconfig()

        for m in self.gc_mul:
            m.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={
                    "dtype": qint16,
                },
                activation_calibration_observer="mix",
            )

        for m in self.gc_mean:
            m.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={
                    "dtype": qint16,
                },
                activation_calibration_observer="mix",
            )
        for m in self.gc_pad:
            m.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={
                    "dtype": qint16,
                },
                activation_calibration_observer="mix",
            )
        
        for module in self.edge_aware_refinements:
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()


class BasicMotionEncoder(nn.Module):
    def __init__(self, corr_radius=4, corr_levels=4):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = 40
        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 64, 128 - 1, 3, padding=1)

    def forward(self, disp, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        disp_ = F.relu(self.convd1(disp))
        disp_ = F.relu(self.convd2(disp_))

        cor_disp = torch.cat([cor, disp_], dim=1)
        out = F.relu(self.conv(cor_disp))
        return torch.cat([out, disp], dim=1)

def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)

def pool4x(x):
    return F.avg_pool2d(x, 5, stride=4, padding=1)

def interp(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[2:], **interp_args)

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)

    def forward(self, h, cz, cr, cq, *x_list):  # h: pre_state
        x = torch.cat(x_list, dim=1)            #     
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx) + cz)
        r = torch.sigmoid(self.convr(hx) + cr)
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)) + cq)

        h = (1-z) * h + z * q
        return h

class DispHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=1):
        super(DispHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class BasicMultiUpdateBlock(nn.Module):
    def __init__(self, n_gru_layers=3, n_downsample=2, hidden_dims=[]):
        super().__init__()
        self.n_gru_layers = n_gru_layers
        self.encoder = BasicMotionEncoder()
        encoder_output_dim = 128

        self.gru04 = ConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (self.n_gru_layers > 1))
        self.gru08 = ConvGRU(hidden_dims[1], hidden_dims[0] * (self.n_gru_layers == 3) + hidden_dims[2])
        self.gru16 = ConvGRU(hidden_dims[0], hidden_dims[1])
        self.disp_head = DispHead(hidden_dims[2], hidden_dim=64, output_dim=1)
        factor = 2**n_downsample

        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 32, 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, net, inp, corr=None, disp=None, iter04=True, iter08=True, iter16=True, update=True):

        if iter16:
            net[2] = self.gru16(net[2], *(inp[2]), pool2x(net[1]))
        if iter08:
            if self.n_gru_layers > 2:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]))
        if iter04:
            motion_features = self.encoder(disp, corr)
            if self.n_gru_layers > 1:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features)

        if not update:
            return net

        delta_disp = self.disp_head(net[0])
        mask_feat_4 = self.mask_feat_4(net[0])
        return net, mask_feat_4, delta_disp

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)

class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x)  # , inplace=True)
        return x
    
class BasicConv_IN(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, IN=True, relu=True, **kwargs):
        super(BasicConv_IN, self).__init__()

        self.relu = relu
        self.use_in = IN
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_in:
            x = self.IN(x)
        if self.relu:
            x = nn.LeakyReLU()(x)  # , inplace=True)
        return x


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, bn=True,
                 relu=True, keep_dispc=False):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.is_3d = is_3d
        if deconv and is_3d:
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel,
                                   stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel,
                                   stride=2, padding=1)

        if self.concat:
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(out_channels * 2, out_channels * mul, False, is_3d, bn, relu, kernel_size=3,
                                   stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1,
                                   padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(
                x,
                size=(rem.shape[-2], rem.shape[-1]),
                mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x

class MultiBasicEncoder(nn.Module):
    def __init__(self, output_dim=[16], norm_fn='batch', dropout=0.0, downsample=3):
        super(MultiBasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample

        # self.norm_111 = nn.BatchNorm2d(128, affine=False, track_running_stats=False)
        # self.norm_222 = nn.BatchNorm2d(128, affine=False, track_running_stats=False)

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(output_dim[0][0])

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, output_dim[0][0], kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = output_dim[0][0]
        self.layer1 = self._make_layer(32, stride=1)
        self.layer2 = self._make_layer(32, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(64, stride=1 + (downsample > 0))
        self.layer_down = self._make_layer(64, stride=1 + (downsample > 0))
        self.layer4 = self._make_layer(64, stride=2)
        self.layer5 = self._make_layer(96, stride=2)

        output_list = []
        
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(64, 32, self.norm_fn, stride=1),
                nn.Conv2d(32, dim[2], 3, padding=1))
            output_list.append(conv_out)

        self.outputs04 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(64, 32, self.norm_fn, stride=1),
                nn.Conv2d(32, dim[1], 3, padding=1))
            output_list.append(conv_out)

        self.outputs08 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Conv2d(96, dim[0], 3, padding=1)
            output_list.append(conv_out)

        self.outputs16 = nn.ModuleList(output_list)

        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, dual_inp=False, num_layers=3):

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer_down(x)
        if dual_inp:
            v = x
            x = x[:(x.shape[0]//2)]

        outputs04 = [f(x) for f in self.outputs04]
        if num_layers == 1:
            return (outputs04, v) if dual_inp else (outputs04,)

        y = self.layer4(x)
        outputs08 = [f(y) for f in self.outputs08]

        if num_layers == 2:
            return (outputs04, outputs08, v) if dual_inp else (outputs04, outputs08)

        z = self.layer5(y)
        outputs16 = [f(z) for f in self.outputs16]

        return (outputs04, outputs08, outputs16, v) if dual_inp else (outputs04, outputs08, outputs16)
    
@OBJECT_REGISTRY.register
class DisparityGRUHead(DisparityNetHead):
    def __init__(self, maxdisp: int = 320, refine_levels: int = 3, bn_kwargs: Dict = None, max_stride: int = 32, num_costvolume: int = 3, num_fusion: int = 6, hidden_dim: int = 16, in_channels: List[int] = (32, 32, 16, 16, 16)):
        super().__init__(maxdisp, refine_levels, bn_kwargs, max_stride, num_costvolume, num_fusion, hidden_dim, in_channels)
        self.n_downsample = 2
        self.n_gru_layers = 3
        self.slow_fast_gru = True
        self.norm_fn = 'batch'
        self.corr_radius = 4
        self.corr_levels = 2
        self.train_iters = 4
        self.valid_iters = 4
        context_dims=[16, 16, 16]
        self.update_block = BasicMultiUpdateBlock(hidden_dims=context_dims)
        self.cnet = MultiBasicEncoder(output_dim=[context_dims, context_dims], norm_fn="batch", downsample=2)
        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], context_dims[i]*3, 3, padding=3//2) for i in range(self.n_gru_layers)])
        # low_disp_max = self.D * pow(2, self.num_costvolume - 1)
        # self.disp_values = nn.Parameter(
        #     torch.range(0, low_disp_max - 1).view(1, low_disp_max, 1, 1)
        #     / low_disp_max,
        #     requires_grad=False,
        # )

    def forward(self, features_inputs: List[Tensor], imgs: Tensor) -> List[Tensor]:
        """Perform the forward pass of the model.

        Args:
            features: The inputs featuremaps.

        Returns:
            pred0: The low disparity.
            pred0_unfold: The low disparity after unfold.
            spx_pred: The weight of each point.
        """

        #refinement
        B, _, _, _ = imgs.shape
        imgs = self.quant_img(imgs)
        imgl = self.get_l_img(imgs, B)

        # Build cost volume as AANet
        features = features_inputs[-3:][::-1]
        aanet_volumes = []
        offset = 0
        for i in range(len(features)):
            aanet_volume = self.build_aanet_volume(
                features[i], self.D * (2 ** i), offset, i
            )
            offset = self.get_offset(offset, i)
            aanet_volumes.append(aanet_volume)

        # Fusion costvolume as AANet
        aanet_volumes = aanet_volumes[::-1]
        for i in range(len(self.fusions)):
            fusion = self.fusions[i]
            aanet_volumes = fusion(aanet_volumes)

        cost0 = self.final_conv(aanet_volumes[0])
        # Obtain low disparity and unfold it.
        pred0 = self.softmax(cost0)
        pred0 = self.dis_mul(pred0)
        pred0 = self.dis_sum(pred0)
        # pred0_unfold = self.unfold(pred0)

        # add ConvFRU
        cnet_list = self.cnet(imgl, num_layers=self.n_gru_layers)
        net_list = [torch.tanh(x[0]) for x in cnet_list]
        inp_list = [torch.relu(x[1]) for x in cnet_list]
        inp_list = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1)) for i, conv in
                    zip(inp_list, self.context_zqr_convs)]
        disp = pred0
        for itr in range(6):
            # disp = disp.detach()
            # geo_feat = geo_fn(disp, coords)
            if self.n_gru_layers == 3 and self.slow_fast_gru:  # Update low-res ConvGRU
                net_list = self.update_block(net_list, inp_list, iter16=True, iter08=False, iter04=False, update=False)
            if self.n_gru_layers >= 2 and self.slow_fast_gru:  # Update low-res ConvGRU and mid-res ConvGRU
                net_list = self.update_block(net_list, inp_list, iter16=self.n_gru_layers == 3, iter08=True,
                                             iter04=False, update=False)
            net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, cost0, disp,
                                                                  iter16=self.n_gru_layers == 3,
                                                                  iter08=self.n_gru_layers >= 2)

            disp = disp + delta_disp


        img_pyramid_list = []
        for i in range(self.refine_levels):
            img_pyramid_list.append(self.downsample[i](imgl))

        pred_pyramid_list = [disp]

        # Refinement disparity
        for i in range(self.refine_levels):
            pred_new = self.edge_aware_refinements[i](
                pred_pyramid_list[i], img_pyramid_list[i]
            )
            pred_pyramid_list.append(pred_new)

        length_all = len(pred_pyramid_list)

        for i in range(length_all):
            pred_pyramid_list[i] = self.dequant(pred_pyramid_list[i])
        return pred_pyramid_list