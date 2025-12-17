import horizon_plugin_pytorch as horizon
import torch.nn as nn
from torch.nn.quantized import FloatFunctional
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import (
    ConvModule2d,
    ConvTransposeModule2d,
)
from hat.registry import OBJECT_REGISTRY


class HGBottleneckDownsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        reduction: int = 4,
    ):
        """Hourglass downsample bottleneck.

        Args:
            in_channels (int): the number of input channel.
            out_channels (int): the number of output channel.
            bias (bool, optional): whether to add learnable bias.
                Defaults to True.
            reduction (int, optional): reduction of channels. Defaults to 4.
        """
        super(HGBottleneckDownsample, self).__init__()
        mid_channels = in_channels // reduction
        if in_channels < reduction:
            mid_channels = in_channels
        self.conv_list = nn.Sequential(
            ConvModule2d(
                in_channels,
                mid_channels,
                3,
                padding=1,
                stride=2,
                bias=bias,
                norm_layer=nn.BatchNorm2d(mid_channels),
                act_layer=nn.ReLU(inplace=True),
            ),
            ConvModule2d(
                mid_channels,
                mid_channels,
                3,
                padding=1,
                stride=1,
                bias=bias,
                norm_layer=nn.BatchNorm2d(mid_channels),
                act_layer=nn.ReLU(inplace=True),
            ),
            ConvModule2d(
                mid_channels,
                out_channels,
                1,
                padding=0,
                stride=1,
                bias=bias,
            ),
        )
        self.activate_layer = nn.ReLU()

    def forward(self, x, non_activation=False):
        out = self.conv_list(x)
        if non_activation:
            return out
        else:
            return self.activate_layer(out)


class HGBottleneckUpsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        reduction: int = 4,
    ):
        """Hourglass upsample bottleneck.

        Args:
            in_channels (int): the number of input channel.
            out_channels (int): the number of output channel.
            bias (bool, optional): whether to add learnable bias.
                Defaults to True.
            reduction (int, optional): reduction of channels. Defaults to 4.
        """
        super(HGBottleneckUpsample, self).__init__()
        mid_channels = in_channels // reduction
        if in_channels < reduction:
            mid_channels = in_channels
        self.conv_list = nn.Sequential(
            ConvTransposeModule2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=bias,
                norm_layer=nn.BatchNorm2d(mid_channels),
                act_layer=nn.ReLU(),
            ),
            ConvModule2d(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
                norm_layer=nn.BatchNorm2d(mid_channels),
                act_layer=nn.ReLU(),
            ),
            ConvModule2d(
                mid_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                stride=1,
                bias=bias,
            ),
        )

    def forward(self, x):
        out = self.conv_list(x)

        return out


class HGBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        reduction: int = 4,
    ):
        """Hourglass bottleneck.

        Args:
            in_channels (int): the number of input channel.
            out_channels (int): the number of output channel.
            bias (bool, optional): whether to add learnable bias.
                Defaults to True.
            reduction (int, optional): reduction of channels. Defaults to 4.
        """
        super(HGBottleneck, self).__init__()
        mid_channels = in_channels // reduction
        if in_channels < reduction:
            mid_channels = mid_channels
        self.conv = nn.Sequential(
            ConvModule2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
                norm_layer=nn.BatchNorm2d(mid_channels),
                act_layer=nn.ReLU(),
            ),
            ConvModule2d(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
                norm_layer=nn.BatchNorm2d(mid_channels),
                act_layer=nn.ReLU(),
            ),
            ConvModule2d(
                mid_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
        )
        self.activate_layer = nn.ReLU()

    def forward(self, x, non_activation=False):
        out = self.conv(x)
        if non_activation:
            return out
        else:
            return self.activate_layer(out)


class HPPOutput(nn.Module):
    def __init__(
        self,
        in_channels: bool,
        out_channels: bool,
        bias: bool = True,
    ):
        """HPP Output module.

        Args:
            in_channels (int): the number of input channel.
            out_channels (int): the number of output channel.
            bias (bool, optional): whether to add learnable bias.
                Defaults to True.
        """
        super(HPPOutput, self).__init__()
        self.conv1 = ConvModule2d(
            in_channels=in_channels,
            out_channels=in_channels // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
            norm_layer=nn.BatchNorm2d(in_channels // 2),
            act_layer=nn.ReLU(),
        )
        self.conv2 = ConvModule2d(
            in_channels=in_channels // 2,
            out_channels=in_channels // 4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
            norm_layer=nn.BatchNorm2d(in_channels // 4),
            act_layer=nn.ReLU(),
        )
        self.conv3 = ConvModule2d(
            in_channels=in_channels // 4,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


class HourglassBasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        num_bottleneck: int = 4,
    ):
        """Hourglass basic block.

        Args:
            in_channels (int): the number of input channel.
            out_channels (int): the number of output channel.
            bias (bool, optional): whether to add learnable bias.
                Defaults to True.
            num_bottleneck (int, optional): number of bottleneck.
                Defaults to 4.
        """
        super(HourglassBasicBlock, self).__init__()
        self.num_bottleneck = num_bottleneck
        self.downsample_stage = self._make_stage(
            in_channels, out_channels, bottleneck_type="downsample"
        )
        self.upsample_stage = self._make_stage(
            out_channels, out_channels, bottleneck_type="upsample"
        )
        self.residual_stage = self._make_stage(
            out_channels, out_channels, bottleneck_type="downsample"
        )
        self.fusion_stage = self._make_stage(
            out_channels, out_channels, bottleneck_type="conv"
        )
        self.normal_layer1 = HGBottleneck(out_channels, out_channels)
        self.normal_layer2 = HGBottleneck(out_channels, out_channels)
        self.normal_layer3 = HGBottleneck(out_channels, out_channels)
        self.normal_layer4 = HGBottleneck(out_channels, out_channels)
        self.normal_act = ConvModule2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            norm_layer=nn.BatchNorm2d(out_channels),
            act_layer=nn.ReLU(),
        )
        self.add_1 = FloatFunctional()
        self.add_2 = FloatFunctional()
        self.add_3 = FloatFunctional()
        self.add_4 = FloatFunctional()

    def _make_stage(
        self,
        in_channels,
        out_channels,
        bottleneck_type,
        bias=True,
    ):
        layer = nn.ModuleList()
        if bottleneck_type == "downsample":
            layer.append(HGBottleneckDownsample(in_channels, out_channels))
            for _ in range(self.num_bottleneck - 1):
                layer.append(
                    HGBottleneckDownsample(out_channels, out_channels)
                )
        elif bottleneck_type == "upsample":
            layer.append(HGBottleneckUpsample(in_channels, out_channels))
            for _ in range(self.num_bottleneck - 1):
                layer.append(HGBottleneckUpsample(out_channels, out_channels))
        elif bottleneck_type == "conv":
            layer.append(
                ConvModule2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=bias,
                    norm_layer=nn.BatchNorm2d(out_channels),
                    act_layer=nn.ReLU(),
                )
            )
            for _ in range(self.num_bottleneck - 1):
                layer.append(
                    ConvModule2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=bias,
                        norm_layer=nn.BatchNorm2d(out_channels),
                        act_layer=nn.ReLU(),
                    )
                )
        else:
            raise TypeError(
                "do not support this `{}` tpye".format(bottleneck_type)
            )
        return layer

    def forward(self, x):
        down_output = [x]
        # [32 x 64] --> [16 x 32] --> [8 x 16] --> [4 x 8] --> [2 x 4]
        for i, down_layer in enumerate(self.downsample_stage):
            x = down_layer(x)
            if i != self.num_bottleneck - 1:
                down_output.append(x)
        # [2 x 4] --> [2 x 4]
        output = self.normal_layer1(x)
        feat = self.normal_layer2(output, non_activation=True)
        output = self.normal_layer3(self.normal_act(feat))
        output = self.normal_layer4(output, True)
        # recover feature shape from [2 x 4] to [32 x 64]
        for i in range(self.num_bottleneck):
            residual_out = self.residual_stage[i](
                down_output[self.num_bottleneck - i - 1], non_activation=True
            )
            add_out = getattr(self, "add_" + str(i + 1)).add(
                residual_out, output
            )
            fusion_output = self.fusion_stage[i](add_out)
            output = self.upsample_stage[i](fusion_output)

        return output, feat


@OBJECT_REGISTRY.register
class HPPBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        instance_num: int = 4,
        residual: bool = True,
    ):
        """HPP block.

        Args:
            in_channels (int): the number of input channel.
            out_channels (int): the number of output channel.
            bias (bool, optional): whether to add learnable bias.
                Defaults to True.
            instance_num (int, optional): the number of instance.
                Defaults to 4.
            residual (bool, optional): whether apply short cut struture
                or not. Defaults to True.
        """
        super(HPPBlock, self).__init__()
        self.residual = residual
        self.hourglass_layer = HourglassBasicBlock(in_channels, out_channels)
        self.bottleneck = nn.Sequential(
            ConvModule2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
                norm_layer=nn.BatchNorm2d(out_channels),
                act_layer=nn.ReLU(),
            ),
            HGBottleneck(out_channels, out_channels),
        )
        self.conv3 = ConvModule2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.residual_layer = ConvModule2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.branch_cls = HPPOutput(out_channels, 1)
        self.branch_reg = HPPOutput(out_channels, 2)
        self.branch_cluster = HPPOutput(out_channels, instance_num)
        self.conv1 = ConvModule2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            norm_layer=nn.BatchNorm2d(out_channels),
            act_layer=nn.ReLU(),
        )
        self.dequant = DeQuantStub()
        self.conv_add_1 = FloatFunctional()
        self.conv_add_2 = FloatFunctional()

    def forward(self, inputs):
        identity = inputs
        outputs = self.conv1(inputs)
        outputs, feature = self.hourglass_layer(outputs)
        outputs_branch = self.bottleneck(outputs)
        # outputs, the input of next HPP block
        outputs = self.conv3(outputs_branch)
        out_confidence = self.branch_cls(outputs_branch)
        out_offset = self.branch_reg(outputs_branch)
        out_instance = self.branch_cluster(outputs_branch)
        residual_out = self.residual_layer(out_confidence)
        # residual
        outputs = self.conv_add_1.add(outputs, residual_out)
        outputs = self.conv_add_2.add(outputs, identity)
        # dequant
        out_confidence = self.dequant(out_confidence)
        out_offset = self.dequant(out_offset)
        out_instance = self.dequant(out_instance)
        feature = self.dequant(feature)
        return out_confidence, out_offset, out_instance, outputs, feature

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvModule2d or type(m) == ConvTransposeModule2d:
                m.fuse_model()

    def set_qconfig(self):
        self.qconfig = horizon.quantization.get_default_qat_qconfig()
