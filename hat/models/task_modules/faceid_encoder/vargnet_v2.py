import torch
import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY
from .vargnet_utils import BasicVarGBlockA, BasicVarGBlockB

__all__ = ["FaceIDLargeVargNet"]


class VarGFaceNet(nn.Module):
    """
    A backbone module of FaceID.

    Args:
        embedding_size : output feat dim
        unit : Unit num for each block.
        channels_list : Channels for each block.
        bn_kwargs : Dict for BN layer.
        bias : Whether to use bias in module.
        disable_quanti_input : Whether to distable quanti input.
        input_channels : Input channels of first conv.
        head_factor : Factor for channels expansion of stage1(mod2).
        input_resize_scale : Narrow_model need resize input 0.65 scale,
        dropout : Dropout rate.
        use_fp16 : Whether to use mixed precision training.
        inplace : Whether to use inplace op.
        flat_output : Whether to view the output tensor.
        include_all: Whether to include stage output.
    """

    def __init__(
        self,
        embedding_size: int,
        unit: list,
        channel_list: list,
        bn_kwargs: dict,
        alpha: float = 1.0,
        group_base: int = 8,
        factor: int = 2,
        bias: bool = False,
        disable_quanti_input: bool = False,
        input_channels: int = 3,
        head_factor: int = 1,
        input_resize_scale: int = None,
        dropout: float = 0.0,
        use_fp16: bool = False,
        inplace: bool = True,
        flat_output: bool = True,
        include_all: bool = False,
    ):

        super(VarGFaceNet, self).__init__()
        self.group_base = group_base
        self.factor = factor
        self.head_factor = head_factor
        self.bias = bias
        self.bn_kwargs = bn_kwargs
        self.embedding_size = embedding_size
        self.disable_quanti_input = disable_quanti_input
        self.input_resize_scale = input_resize_scale
        self.dropout = dropout
        self.use_fp16 = use_fp16
        self.inplace = inplace
        self.include_all = include_all
        self.flat_output = flat_output

        channel_list = [int(chls * alpha) for chls in channel_list]
        self.quant = QuantStub(scale=1.0 / 128.0)
        self.dequant = DeQuantStub()
        self.in_channels = channel_list[0]
        self.conv = ConvModule2d(
            input_channels,
            channel_list[0],
            kernel_size=3,
            bias=bias,
            padding=1,
            norm_layer=nn.BatchNorm2d(channel_list[0], **bn_kwargs),
            act_layer=nn.ReLU(inplace=inplace),
        )
        self.mod1 = BasicVarGBlockA(
            in_channels=channel_list[0],
            mid_channels=channel_list[0],
            out_channels=channel_list[0],
            stride=2,
            bn_kwargs=bn_kwargs,
            group_base=8,
            padding=1,
            bias=bias,
            factor=1,
            merge_branch=False,
            dw_with_relu=False,
            pw_with_relu=False,
        )
        head_factor = 2 if self.head_factor == 2 else 8 // group_base
        self.mod2 = self._make_stage(channel_list[1], 1, unit[0], 2, False)
        self.mod3 = self._make_stage(channel_list[2], 2, unit[1], self.factor)
        self.mod4 = self._make_stage(channel_list[3], 2, unit[2], self.factor)
        self.mod5 = self._make_stage(channel_list[4], 2, unit[3], self.factor)

        self.output = nn.Sequential(
            ConvModule2d(
                channel_list[-2],
                channel_list[-1],
                1,
                bias=bias,
                norm_layer=nn.BatchNorm2d(channel_list[-1], **bn_kwargs),
                act_layer=nn.ReLU(inplace=inplace),
            ),
            ConvModule2d(
                channel_list[-1],
                channel_list[-1],
                kernel_size=7,
                bias=bias,
                groups=128,
                norm_layer=nn.BatchNorm2d(channel_list[-1], **bn_kwargs),
                act_layer=nn.ReLU(inplace=inplace),
            ),
        )

        fc_bn_kwargs = {
            "eps": 1e-5,
        }
        self.fc = ConvModule2d(
            channel_list[-1],
            self.embedding_size,
            kernel_size=1,
            bias=bias,
            padding=0,
            norm_layer=nn.BatchNorm2d(self.embedding_size, **fc_bn_kwargs),
        )
        nn.init.constant_(self.fc.conv_list[1].weight, 1.0)
        self.fc.conv_list[1].weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_stage(
        self, channels, strides, repeats, factor, merge_branch=True
    ):
        layers = []
        layers.append(
            BasicVarGBlockB(
                self.in_channels,
                channels,
                channels,
                strides,
                bias=self.bias,
                bn_kwargs=self.bn_kwargs,
                factor=factor,
                group_base=self.group_base,
                inplace=self.inplace,
            )
        )

        self.in_channels = channels
        for _ in range(1, repeats):
            layers.append(
                BasicVarGBlockA(
                    channels,
                    channels,
                    channels,
                    1,
                    bias=self.bias,
                    bn_kwargs=self.bn_kwargs,
                    factor=factor,
                    group_base=self.group_base,
                    merge_branch=False,
                    inplace=self.inplace,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x if self.disable_quanti_input else self.quant(x)
        if self.input_resize_scale is not None:
            x = self.resize(x)
        output = []

        with torch.cuda.amp.autocast(self.use_fp16):
            x = self.conv(x)

            for module in [
                self.mod1,
                self.mod2,
                self.mod3,
                self.mod4,
                self.mod5,
            ]:
                x = module(x)
                output.append(x)
            x = self.output(x)
        x = self.fc(x.float() if self.use_fp16 else x)
        x = self.dequant(x)
        if self.flat_output:
            x = x.view(-1, self.embedding_size)

        if self.include_all:
            return output, x
        return x

    def fuse_model(self):
        self.conv.fuse_model()
        self.mod1.fuse_model()
        modules = [self.mod2, self.mod3, self.mod4, self.mod5]
        modules += [self.output]
        for module in modules:
            for m in module:
                if hasattr(m, "fuse_model"):
                    m.fuse_model()
        self.fc.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        # disable output quantization for last quanti layer.
        self.fc.qconfig = qconfig_manager.get_default_qat_out_qconfig()


@OBJECT_REGISTRY.register
class FaceIDLargeVargNet(VarGFaceNet):
    """
    A backbone module of FaceID.

    Args:
        bn_kwargs : Dict for BN layer.
        bias : Whether to use bias in module.
        embedding_size : output feat dim
        dropout : Dropout rate.
        use_fp16 : Whether to use mixed precision training.
        inplace : Whether to use inplace op.
        flat_output : Whether to view the output tensor.
        include_all: Whether to include stage output.
    """

    def __init__(
        self,
        bn_kwargs: dict,
        bias: bool = False,
        embedding_size: int = 512,
        dropout: float = 0.0,
        use_fp16: bool = True,
        inplace: bool = True,
        flat_output: bool = True,
        include_all: bool = False,
    ):
        unit = [3, 4, 16, 3]
        channel_list = [64, 96, 192, 384, 768, 1024]
        super(FaceIDLargeVargNet, self).__init__(
            embedding_size=embedding_size,
            unit=unit,
            channel_list=channel_list,
            bn_kwargs=bn_kwargs,
            bias=bias,
            dropout=dropout,
            use_fp16=use_fp16,
            inplace=inplace,
            flat_output=flat_output,
            include_all=include_all,
        )
