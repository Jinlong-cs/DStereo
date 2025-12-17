import horizon_plugin_pytorch.nn as hnn
import torch.nn as nn

from hat.models.base_modules.conv_module import FixedConvModule2d, FusedConv2d
from hat.registry import OBJECT_REGISTRY

__all__ = ["NISP"]


class NISPBlock(nn.Module):
    """
    A module of Neural ISP group block module.

    Args:
        channels: Input channels of first conv.
        use_bias: Whether to use bias in module.
        bn_kwargs: Dict for BN layer.
        group: group for the second fused conv.
    """

    def __init__(
        self,
        channels=3,
        use_bias: bool = False,
        bn_kwargs: dict = None,
        group: int = 4,
    ):
        super(NISPBlock, self).__init__()
        self.head_conv = FusedConv2d(
            in_channels=channels,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            bn_kwargs=bn_kwargs,
            with_relu=True,
            # add group
            group=group,
        )

        self.conv1 = nn.Conv2d(
            16, channels, kernel_size=1, stride=1, padding=0
        )
        self.conv2 = FusedConv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=use_bias,
            group=group,
            bn_kwargs=bn_kwargs,
            with_relu=True,
        )

        self.conv2_1x1 = FusedConv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=use_bias,
            bn_kwargs=bn_kwargs,
            with_relu=True,
        )
        self.conv3 = nn.Conv2d(
            16, channels, kernel_size=1, stride=1, padding=0
        )

        self.upsampler = hnn.Interpolate(
            scale_factor=2, mode="bilinear", recompute_scale_factor=True
        )
        self.skip_add = nn.quantized.FloatFunctional()

        self.bnrelu = FixedConvModule2d(channels, bn_kwargs=bn_kwargs)

    def forward(self, x):
        x1 = self.head_conv(x)
        out = self.conv1(x1)
        x_down = self.conv2(x1)
        x_down = self.conv2_1x1(x_down)

        out_down = self.conv3(x_down)
        out_up = self.upsampler(out_down)
        out_fusion = self.skip_add.add(out, out_up)
        output = self.bnrelu(out_fusion)

        return output


@OBJECT_REGISTRY.register
class NISP(nn.Module):
    """
    A module of Neural ISP.

    Args:
        in_channels: Input channels of first conv.
        out_channels: Output channels of last conv.
        block_num: Number of NISPBlock.
        use_bias: Whether to use bias in module.
        bn_kwargs: Dict for BN layer.
        group: group for the second fused conv.

    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_num: int = 2,
        use_bias: bool = False,
        bn_kwargs: dict = None,
        group: int = 4,
    ):
        super(NISP, self).__init__()
        self.head_conv = FusedConv2d(
            in_channels=in_channels,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            bn_kwargs=bn_kwargs,
            with_relu=True,
        )

        self.middle_conv = nn.Sequential(
            *[
                NISPBlock(
                    channels=16,
                    use_bias=use_bias,
                    bn_kwargs=bn_kwargs,
                    group=group,
                )
                for _ in range(block_num)
            ]
        )

        self.skip_add = nn.quantized.FloatFunctional()
        self.end_conv = FusedConv2d(
            in_channels=16,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            bn_kwargs=bn_kwargs,
            with_relu=True,
        )

    def forward(self, data):
        x = data["img"]
        x_intro = self.head_conv(x)
        x = self.middle_conv(x_intro)
        x = self.skip_add.add(x_intro, x)
        x = self.end_conv(x)
        return x
