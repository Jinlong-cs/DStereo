# Copyright (c) Horizon Robotics. All rights reserved.

import math
from typing import List, Tuple, Union

import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from hat.models.backbones.mixvargenet import IdentityConfig, MixVarGENetConfig
from hat.models.base_modules.basic_mixvargenet_module import MixVarGEBlock
from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.base_modules.extend_container import ExtSequential
from hat.models.base_modules.inverted_residual import InvertedResidual
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "EyeClsHead",
    "EyeMultiBranchHead",
    "EyeMultiBinBranchHead",
    "EyeMixVarGEClsHead",
    "EyeMixVarGEBinClsHead",
    "EyeMultiBranchSingleHead",
]


@OBJECT_REGISTRY.register
class EyeMultiBranchSingleHead(nn.Module):
    """
    Multi-branch for eye status and use one cls.

    Args:
        l_cls_head: Single eye cls module.
        r_cls_head: Single eye cls module.
        num_classes: Class num.
        bn_kwargs: Dict BN layer.
        in_channels: In_channels of the module.
        out_channels: Out_channels of the last_conv.
        pool_size: Pooling size.
        bias: Whether to use bias in module.
        flat_output: Whether to view the output tensor.
        dropout_rate: Dropout ratio.
    """

    def __init__(
        self,
        l_cls_head: nn.Module,
        r_cls_head: nn.Module,
        num_classes: int,
        bn_kwargs: dict,
        in_channels: int,
        out_channels: int,
        pool_size: Union[int, Tuple[int, int]],
        bias: bool = False,
        flat_output: bool = True,
        dropout_rate: float = 0.2,
    ):
        super(EyeMultiBranchSingleHead, self).__init__()
        self.r_branch = r_cls_head
        self.l_branch = l_cls_head
        self.num_classes = num_classes
        self.flat_output = flat_output

        self.dequant = DeQuantStub()
        self.output = ExtSequential(
            [
                ConvModule2d(
                    in_channels,
                    out_channels,
                    1,
                    bias=bias,
                    norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
                    act_layer=nn.ReLU(inplace=True),
                ),
                nn.AvgPool2d(pool_size[0], pool_size[1]),
                nn.Dropout(p=dropout_rate, inplace=True),
                ConvModule2d(
                    out_channels,
                    num_classes,
                    kernel_size=1,
                    bias=True,
                ),
            ]
        )
        self.cat_op = nn.quantized.FloatFunctional()

    def forward(self, x):
        l_branch_feat = self.l_branch(x)
        r_branch_feat = self.r_branch(x)
        feats = self.cat_op.cat([l_branch_feat, r_branch_feat], dim=0)
        x = self.output(feats)
        x = self.dequant(x)
        if self.flat_output:
            x = x.view(-1, self.num_classes)

        return x

    def fuse_model(self):
        for module in [self.l_branch, self.r_branch, self.output]:
            module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        getattr(
            self.output, "3"
        ).qconfig = qconfig_manager.get_default_qat_out_qconfig()


@OBJECT_REGISTRY.register
class EyeMultiBranchHead(nn.Module):
    """
    Multi-branch for eye status cls.

    Args:
        l_cls_head: Single eye cls module.
        r_cls_head: Single eye cls module.
    """

    def __init__(
        self,
        l_cls_head: nn.Module,
        r_cls_head: nn.Module,
    ):
        super(EyeMultiBranchHead, self).__init__()
        self.r_branch = r_cls_head
        self.l_branch = l_cls_head

    def forward(self, x):
        l_branch_feat = self.l_branch(x)
        r_branch_feat = self.r_branch(x)
        return l_branch_feat, r_branch_feat

    def fuse_model(self):
        for module in [self.l_branch, self.r_branch]:
            module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        for module in [self.l_branch, self.r_branch]:
            module.set_qconfig()


@OBJECT_REGISTRY.register
class EyeMultiBinBranchHead(nn.Module):
    """
    Multi-branch for binary eye status cls.

    Args:
        l_cls_head: Single eye cls module.
        r_cls_head: Single eye cls module.
    """

    def __init__(
        self,
        l_cls_head: nn.Module,
        r_cls_head: nn.Module,
    ):
        super(EyeMultiBinBranchHead, self).__init__()
        self.r_branch = r_cls_head
        self.l_branch = l_cls_head

    def forward(self, x):
        l_branch_feat = self.l_branch(x)
        r_branch_feat = self.r_branch(x)
        return l_branch_feat, r_branch_feat

    def fuse_model(self):
        for module in [self.l_branch, self.r_branch]:
            module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        for module in [self.l_branch, self.r_branch]:
            module.set_qconfig()


@OBJECT_REGISTRY.register
class EyeMixVarGEBinClsHead(nn.Module):
    """
    Single-branch for binary eye status cls.

    Args:
        bn_kwargs: Dict for BN layer.
        in_channels: In_channels of the module.
        out_channels: Out_channels of the last_conv.
        bias: Whether to use bias in module.
        pool_size: Pooling size.
        include_top: Whether to include output layer.
        flat_output: Whether to view the output tensor.
        use_dw_as_avgpool: Whether to replace AvgPool with DepthWiseConv
    """

    def __init__(
        self,
        bn_kwargs: dict,
        in_channels: int,
        out_channels: int,
        pool_size: Union[int, Tuple[int, int]],
        bias: bool = False,
        flat_output: bool = True,
        dropout_rate: float = 0.2,
        groups: int = 16,
    ):
        super(EyeMixVarGEBinClsHead, self).__init__()
        self.bias = bias
        self.bn_kwargs = bn_kwargs
        self.flat_output = flat_output

        self.dequant = DeQuantStub()

        self.pool_size = pool_size

        self.output = ExtSequential(
            [
                ConvModule2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    bias=self.bias,
                    norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
                    padding=1,
                    groups=groups,
                    act_layer=nn.ReLU(inplace=True),
                ),
                nn.AvgPool2d(pool_size[0], pool_size[1]),
                nn.Dropout(p=dropout_rate, inplace=True),
                ConvModule2d(
                    out_channels,
                    1,
                    kernel_size=1,
                    padding=0,
                    bias=True,
                ),
            ]
        )

    def forward(self, x):
        x = self.output(x)
        x = self.dequant(x)
        if self.flat_output:
            x = x.view(-1, 1)
        return x

    def fuse_model(self):

        for m in self.output:
            if hasattr(m, "fuse_model"):
                m.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        # disable output quantization for last quanti layer.
        getattr(
            self.output, "3"
        ).qconfig = qconfig_manager.get_default_qat_out_qconfig()


@OBJECT_REGISTRY.register
class EyeMixVarGEClsHead(nn.Module):
    """
    Single-branch for eye status cls use Mixvargenet block.

    Args:
        net_config: network setting.
        num_classes: Num classes of output layer.
        bn_kwargs: Dict for BN layer.
        in_channels: In_channels of the module.
        out_channels: Out_channels of the last_conv.
        pool_size: Pooling size.
        alpha: Alpha for channels.
        include_top: Whether to include output layer.
        bias: Whether to use bias in module.
        flat_output: Whether to view the output tensor.
        dropout_rate: Dropout ratio.
    """

    def __init__(
        self,
        net_config: List[MixVarGENetConfig],
        num_classes: int,
        bn_kwargs: dict,
        in_channels: int,
        out_channels: int,
        pool_size: Union[int, Tuple[int, int]],
        include_top: bool = True,
        bias: bool = False,
        flat_output: bool = True,
        dropout_rate: float = 0.2,
    ):
        super(EyeMixVarGEClsHead, self).__init__()
        self.bias = bias
        self.bn_kwargs = bn_kwargs
        self.num_classes = num_classes
        self.include_top = include_top
        self.flat_output = flat_output

        self.dequant = DeQuantStub()
        stage_config = net_config

        self.mod1 = self._make_stage(stage_config[-1])

        if self.include_top:
            self.output = ExtSequential(
                [
                    ConvModule2d(
                        in_channels,
                        out_channels,
                        1,
                        bias=bias,
                        norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
                        act_layer=nn.ReLU(inplace=True),
                    ),
                    nn.AvgPool2d(pool_size[0], pool_size[1]),
                    nn.Dropout(p=dropout_rate, inplace=True),
                    ConvModule2d(
                        out_channels,
                        num_classes,
                        1,
                        bias=True,
                    ),
                ]
            )
        else:
            self.output = None

    def _make_stage(self, stage_config):
        def _get_fusion_channels(fusion_strides):
            if len(fusion_strides) == 0:
                return []
            strides_ids = map(
                lambda stride: int(math.log2(stride) - 1), fusion_strides
            )
            fusion_channels = map(
                lambda idx: self.net_config[idx][0].out_channels, strides_ids
            )
            return list(fusion_channels)

        layers = []
        for config_i in stage_config:
            if isinstance(config_i, IdentityConfig):
                layers.append(nn.Identity())
            else:
                layers.append(
                    MixVarGEBlock(
                        in_ch=config_i.in_channels,
                        block_ch=config_i.out_channels,
                        head_op=config_i.head_op,
                        stack_ops=config_i.stack_ops,
                        stack_factor=config_i.stack_factor,
                        stride=config_i.stride,
                        bias=self.bias,
                        fusion_channels=_get_fusion_channels(
                            config_i.fusion_strides
                        ),
                        downsample_num=config_i.extra_downsample_num,
                        bn_kwargs=self.bn_kwargs,
                    )
                )
        return ExtSequential(layers)

    def forward(self, x):

        for module in [self.mod1]:
            module_out = module(x)
            if isinstance(module_out, tuple) and len(module_out) != 1:
                x, _ = module_out
            else:
                x = module_out

        if not self.include_top:
            return x

        x = self.output(x)
        x = self.dequant(x)
        if self.flat_output:
            x = x.view(-1, self.num_classes)
        return x

    def fuse_model(self):

        self.mod1.fuse_model()

        if self.include_top:
            for m in self.output:
                if hasattr(m, "fuse_model"):
                    m.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        if self.include_top:
            # disable output quantization for last quanti layer.
            getattr(
                self.output, "3"
            ).qconfig = qconfig_manager.get_default_qat_out_qconfig()


@OBJECT_REGISTRY.register
class EyeClsHead(nn.Module):
    """
    Single-branch for eye status cls use mobilenetv2 last two stage.

    Args:
        num_classes: Num classes of output layer.
        bn_kwargs: Dict for BN layer.
        in_channels: In_channels of the module.
        pool_size: Pooling size.
        alpha: Alpha for mobilenetv1.
        bias: Whether to use bias in module.
        include_top: Whether to include output layer.
        flat_output: Whether to view the output tensor.
        use_dw_as_avgpool (bool): Whether to replace AvgPool with DepthWiseConv
        dropout_rate: Dropout ratio.
    """

    def __init__(
        self,
        num_classes: int,
        bn_kwargs: dict,
        in_channels: int,
        pool_size: Union[int, Tuple[int, int]],
        alpha: float = 1.0,
        bias: bool = True,
        include_top: bool = True,
        flat_output: bool = True,
        use_dw_as_avgpool: bool = False,
        dropout_rate: float = 0.2,
    ):
        super(EyeClsHead, self).__init__()
        self.alpha = alpha
        self.bias = bias
        self.bn_kwargs = bn_kwargs
        self.num_classes = num_classes
        self.include_top = include_top
        self.flat_output = flat_output
        self.use_dw_as_avgpool = use_dw_as_avgpool

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        in_chls = [
            [in_channels] + [160] * 3,
        ]
        out_chls = [
            [160] * 3 + [320],
        ]

        self.mod1 = self._make_stage(in_chls[0], out_chls[0], 6)

        if self.use_dw_as_avgpool:
            pool_layer = ConvModule2d(
                in_channels=int(out_chls[0][-1] * alpha),
                out_channels=max(1280, int(1280 * alpha)),
                kernel_size=(pool_size[0], pool_size[1]),
                stride=1,
                padding=0,
                groups=max(1280, int(1280 * alpha)),
            )
        else:
            pool_layer = nn.AvgPool2d(pool_size[0], pool_size[1])

        if self.include_top:
            self.output = nn.Sequential(
                ConvModule2d(
                    int(out_chls[0][-1] * alpha),
                    max(1280, int(1280 * alpha)),
                    1,
                    bias=self.bias,
                    norm_layer=nn.BatchNorm2d(
                        max(1280, int(1280 * alpha)), **bn_kwargs
                    ),
                    act_layer=nn.ReLU(inplace=True),
                ),
                pool_layer,
                nn.Dropout(p=dropout_rate, inplace=True),
                ConvModule2d(
                    max(1280, int(1280 * alpha)),
                    num_classes,
                    1,
                    bias=True,
                ),
            )
        else:
            self.output = None

    def _make_stage(self, in_chls, out_chls, expand_t):
        layers = []
        in_chls = [int(chl * self.alpha) for chl in in_chls]
        out_chls = [int(chl * self.alpha) for chl in out_chls]
        stride = 1
        for _, in_chl, out_chl in zip(range(len(in_chls)), in_chls, out_chls):
            layers.append(
                InvertedResidual(
                    in_chl,
                    out_chl,
                    stride,
                    expand_t,
                    self.bn_kwargs,
                    self.bias,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        output = []
        for module in [self.mod1]:
            x = module(x)
            output.append(x)
        if not self.include_top:
            return output
        x = self.output(x)
        x = self.dequant(x)
        if self.flat_output:
            x = x.view(-1, self.num_classes)
        return x

    def fuse_model(self):
        modules = [self.mod1]
        if self.include_top:
            modules += [self.output]
        for module in modules:
            for m in module:
                if hasattr(m, "fuse_model"):
                    m.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        if self.include_top:
            # disable output quantization for last quanti layer.
            getattr(
                self.output, "2"
            ).qconfig = qconfig_manager.get_default_qat_out_qconfig()
