import torch
import torch.nn as nn

from hat.models.base_modules.separable_conv_module import (
    SeparableGroupConvModule2d,
)


class BasicVarGBlockA(nn.Module):
    """
    A basic block for VarGFaceNet.

    Args:
        in_channels : Input channels.
        mid_channels : Mid channels.
        out_channels : Output channels.
        stride : Stride of basic block.
        bn_kwargs : Dict for BN layer.
        kernel_size : Kernel size of basic block.
        padding : Padding of basic block.
        bias : Whether to use bias in basic block.
        factor : Factor for channels expansion.
        group_base : Group base for group conv.
        merge_branch : Whether to merge branch.
        dw_with_relu : Whether to use relu in dw conv.
        pw_with_relu : Whether to use relu in pw conv.
        inplace : Whether to use inplace op.
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        stride: int,
        bn_kwargs: dict,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = True,
        factor: int = 2,
        group_base: int = 8,
        merge_branch: bool = False,
        dw_with_relu: bool = True,
        pw_with_relu: bool = True,
        inplace: bool = True,
    ):
        super(BasicVarGBlockA, self).__init__()
        self.merge_branch = merge_branch
        self.downsample = None
        if not (stride == 1 and in_channels == out_channels):
            self.downsample = SeparableGroupConvModule2d(
                in_channels,
                out_channels,
                kernel_size,
                dw_channels=in_channels,
                bias=bias,
                padding=padding,
                factor=factor,
                groups=int(in_channels // group_base),
                stride=stride,
                dw_norm_layer=nn.BatchNorm2d(
                    int(in_channels * factor), **bn_kwargs
                ),
                dw_act_layer=nn.ReLU(inplace=inplace)
                if dw_with_relu
                else None,
                pw_norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
            )

        if self.merge_branch:
            self.body_conv = SeparableGroupConvModule2d(
                in_channels,
                out_channels,
                kernel_size,
                dw_channels=in_channels,
                bias=bias,
                padding=padding,
                factor=factor,
                groups=int(in_channels // group_base),
                stride=stride,
                dw_norm_layer=nn.BatchNorm2d(
                    int(in_channels * factor), **bn_kwargs
                ),
                dw_act_layer=nn.ReLU(inplace=inplace)
                if dw_with_relu
                else None,
                pw_norm_layer=nn.BatchNorm2d(mid_channels, **bn_kwargs),
            )
        self.conv = SeparableGroupConvModule2d(
            in_channels,
            mid_channels,
            kernel_size,
            dw_channels=in_channels,
            bias=bias,
            padding=padding,
            factor=factor,
            groups=int(in_channels // group_base),
            stride=stride,
            dw_norm_layer=nn.BatchNorm2d(
                int(in_channels * factor), **bn_kwargs
            ),
            dw_act_layer=nn.ReLU(inplace=inplace) if dw_with_relu else None,
            pw_norm_layer=nn.BatchNorm2d(mid_channels, **bn_kwargs),
        )
        self.out_conv = SeparableGroupConvModule2d(
            mid_channels,
            out_channels,
            kernel_size,
            dw_channels=mid_channels,
            bias=bias,
            padding=padding,
            factor=factor,
            groups=int(mid_channels // group_base),
            stride=1,
            dw_norm_layer=nn.BatchNorm2d(
                int(mid_channels * factor), **bn_kwargs
            ),
            dw_act_layer=nn.ReLU(inplace=inplace) if dw_with_relu else None,
            pw_norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
        )

        self.merge_add = nn.quantized.FloatFunctional()
        self.merge_act = nn.ReLU(inplace=True)
        self.shortcut_add = nn.quantized.FloatFunctional()
        self.out_act = nn.ReLU(inplace=True) if pw_with_relu else None

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        if self.merge_branch:
            body = self.body_conv(x)
            x = self.conv(x)
            x = self.merge_add.add(x, body)
            x = self.merge_act(x)
        else:
            x = self.conv(x)
            x = self.merge_act(x)

        out = self.out_conv(x)
        out = self.shortcut_add.add(out, identity)
        if self.out_act is not None:
            out = self.out_act(out)
        return out

    def fuse_model(self):
        from horizon_plugin_pytorch import quantization

        if self.downsample is not None:
            self.downsample.fuse_model()
        if self.merge_branch:
            self.body_conv.fuse_model()
            getattr(self.conv, "0").fuse_model()
            torch.quantization.fuse_modules(
                self,
                ["conv.1.0", "conv.1.1", "merge_add", "merge_act"],
                inplace=True,
                fuser_func=quantization.fuse_known_modules,
            )
        else:
            getattr(self.conv, "0").fuse_model()
            torch.quantization.fuse_modules(
                self,
                ["conv.1.0", "conv.1.1", "merge_act"],
                inplace=True,
                fuser_func=quantization.fuse_known_modules,
            )

        getattr(self.out_conv, "0").fuse_model()
        fuse_list = ["out_conv.1.0", "out_conv.1.1", "shortcut_add"]
        if self.out_act is not None:
            fuse_list += ["out_act"]
        torch.quantization.fuse_modules(
            self,
            fuse_list,
            inplace=True,
            fuser_func=quantization.fuse_known_modules,
        )


class BasicVarGBlockB(nn.Module):
    """
    A basic block for VarGFaceNet.

    Args:
        in_channels : Input channels.
        mid_channels : Mid channels.
        out_channels : Output channels.
        stride : Stride of basic block.
        bn_kwargs : Dict for BN layer.
        kernel_size : Kernel size of basic block.
        padding : Padding of basic block.
        bias : Whether to use bias in basic block.
        factor : Factor for channels expansion.
        group_base : Group base for group conv.
        dw_with_relu : Whether to use relu in dw conv.
        pw_with_relu : Whether to use relu in pw conv.
        inplace : Whether to use inplace op.
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        stride: int,
        bn_kwargs: dict,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = True,
        factor: int = 1,
        group_base: int = 8,
        dw_with_relu: bool = True,
        pw_with_relu: bool = True,
        inplace: bool = True,
    ):
        super(BasicVarGBlockB, self).__init__()
        first_increased_channels = in_channels * factor
        second_increased_channels = in_channels * factor

        self.downsample = SeparableGroupConvModule2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dw_channels=in_channels,
            bias=bias,
            padding=padding,
            factor=factor,
            groups=int(in_channels // group_base),
            stride=stride,
            dw_norm_layer=nn.BatchNorm2d(
                int(in_channels * factor), **bn_kwargs
            ),
            dw_act_layer=nn.ReLU(inplace=inplace) if dw_with_relu else None,
            pw_norm_layer=nn.BatchNorm2d(int(out_channels), **bn_kwargs),
            pw_act_layer=nn.ReLU(inplace=inplace) if dw_with_relu else None,
        )

        self.cardinality1_conv = SeparableGroupConvModule2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dw_channels=in_channels,
            bias=bias,
            padding=padding,
            factor=factor,
            groups=int(in_channels // group_base),
            stride=stride,
            dw_norm_layer=nn.BatchNorm2d(
                int(first_increased_channels), **bn_kwargs
            ),
            dw_act_layer=nn.ReLU(inplace=inplace) if dw_with_relu else None,
            pw_norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
            pw_act_layer=nn.ReLU(inplace=inplace) if pw_with_relu else None,
        )
        self.cardinality2_conv = SeparableGroupConvModule2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dw_channels=in_channels,
            bias=bias,
            padding=padding,
            factor=factor,
            groups=int(in_channels // group_base),
            stride=stride,
            dw_norm_layer=nn.BatchNorm2d(
                int(second_increased_channels), **bn_kwargs
            ),
            dw_act_layer=nn.ReLU(inplace=inplace) if dw_with_relu else None,
            pw_norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
            pw_act_layer=nn.ReLU(inplace=inplace) if pw_with_relu else None,
        )
        self.merge_conv = SeparableGroupConvModule2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            dw_channels=out_channels,
            bias=bias,
            padding=padding,
            factor=2,
            groups=int(out_channels // group_base),
            stride=1,
            dw_norm_layer=nn.BatchNorm2d(int(out_channels * 2), **bn_kwargs),
            dw_act_layer=nn.ReLU(inplace=inplace) if dw_with_relu else None,
            pw_norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
            pw_act_layer=nn.ReLU(inplace=inplace) if pw_with_relu else None,
        )

        self.merge_add = nn.quantized.FloatFunctional()
        self.merge_act = nn.ReLU(inplace=inplace)
        self.shortcut_add = nn.quantized.FloatFunctional()
        self.out_act = nn.ReLU(inplace=inplace) if pw_with_relu else None

    def forward(self, x):
        shortcut = x
        shortcut = self.downsample(shortcut)
        cardi1 = self.cardinality1_conv(x)
        cardi2 = self.cardinality2_conv(x)

        cardi = self.merge_add.add(cardi1, cardi2)
        out = self.merge_conv(cardi)

        out = self.shortcut_add.add(out, shortcut)
        if self.out_act is not None:
            out = self.out_act(out)
        return out

    def fuse_model(self):
        from horizon_plugin_pytorch import quantization

        self.downsample.fuse_model()

        self.cardinality2_conv.fuse_model()
        self.cardinality1_conv.fuse_model()

        getattr(self.merge_conv, "0").fuse_model()
        fuse_list = ["merge_conv.1.0", "merge_conv.1.1", "shortcut_add"]
        if self.out_act is not None:
            fuse_list += ["out_act"]
        torch.quantization.fuse_modules(
            self,
            fuse_list,
            inplace=True,
            fuser_func=quantization.fuse_known_modules,
        )
