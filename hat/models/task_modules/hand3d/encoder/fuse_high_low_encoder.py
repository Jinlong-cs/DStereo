# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Dict, List

from torch import Tensor, nn

from hat.models.base_modules.basic_resnet_module import BasicResBlock
from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY

__all__ = ["H3DFuseHighLowEncoder", "H3DHeatMapResEncoder"]


@OBJECT_REGISTRY.register
class H3DHeatMapResEncoder(nn.Module):
    """
    A resblock for heatmap encoder.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        bn_kwargs: Dict for Bn layer.
        stride: Stride for first conv.
        bias: Whether to use bias in module.
        num_joints: Number of hand landmarks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_kwargs: dict = None,
        stride: int = 1,
        bias: bool = True,
        num_joints: int = 21,
    ):
        super(H3DHeatMapResEncoder, self).__init__()
        self.res_encoder1 = BasicResBlock(
            in_channels, num_joints, bn_kwargs, stride, bias
        )
        self.res_encoder2 = BasicResBlock(
            num_joints, out_channels, bn_kwargs, stride, bias
        )

    def forward(self, x):
        latents = self.res_encoder1(x)
        encoding = self.res_encoder2(latents)
        return latents, encoding

    def fuse_model(self):
        self.res_encoder1.fuse_model()
        self.res_encoder2.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        for module in [
            self.res_encoder1,
            self.res_encoder2,
        ]:
            if module is not None and hasattr(module, "set_qconfig"):
                module.set_qconfig()


@OBJECT_REGISTRY.register
class H3DFuseHighLowEncoder(nn.Module):
    """H3D Encoder that fuses high and low semantic features.

    Args:
        in_channels: Input channels of last featuremap.
        bifpn_channels: Channels of bifpn.
        encoding_channels: Channels of encoding.
        heatmap_encoder: Encoder network for heatmap.
            Defaults to None.
        bn_kwargs: Dict for BN layer.
            Defaults to None.
        global_avg_size: Params for nn.AvgPool2d, acts as a global pooling
            layer. Defaults to 4.
    """

    def __init__(
        self,
        in_channels: int,
        bifpn_channels: int,
        encoding_channels: int,
        heatmap_encoder: nn.Module = None,
        bn_kwargs: Dict = None,
        global_avg_size: int = 4,
    ):
        super(H3DFuseHighLowEncoder, self).__init__()
        self.heatmap_encoder = heatmap_encoder
        self.conv_shape = ConvModule2d(
            in_channels,
            encoding_channels,
            kernel_size=3,
            padding=1,
            stride=2,
            bias=True,
            norm_layer=nn.BatchNorm2d(encoding_channels, **bn_kwargs),
            act_layer=nn.ReLU(inplace=True),
        )
        self.conv_pose = ConvModule2d(
            bifpn_channels,
            encoding_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=True,
            norm_layer=nn.BatchNorm2d(encoding_channels, **bn_kwargs),
            act_layer=nn.ReLU(inplace=True),
        )
        self.downsample_encoder = nn.Sequential(
            BasicResBlock(
                encoding_channels,
                encoding_channels,
                bn_kwargs,
                stride=4,
                bias=True,
            ),
            BasicResBlock(
                encoding_channels,
                encoding_channels,
                bn_kwargs,
                stride=2,
                bias=True,
            ),
        )
        self.fusion_conv = ConvModule2d(
            encoding_channels,
            encoding_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=True,
            norm_layer=nn.BatchNorm2d(encoding_channels, **bn_kwargs),
            act_layer=nn.ReLU(inplace=True),
        )
        self.short_add = nn.quantized.FloatFunctional()
        self.gap_conv = nn.AvgPool2d(
            global_avg_size
        )  # used as global avg pool

    def forward(self, featuremaps: List[Tensor]):
        # cal shape encoding feature by high-level semantic featuremap
        feat_shape = self.conv_shape(featuremaps[-1])
        shape_encoding = self.gap_conv(feat_shape)

        # cal heatmap latent (for heatmap predict) and encoding feature
        # (for heatmap predict) by low-level semantic featuremap
        heatmap_latent, heatmap_encoding = self.heatmap_encoder(featuremaps[0])

        # cal confusion encoding for pose and camera params predict
        confusion_feature = self.short_add.add(
            self.conv_pose(featuremaps[0]), heatmap_encoding
        )
        deconfusion_feature = self.downsample_encoder(confusion_feature)
        confusion_encoding = self.fusion_conv(
            self.short_add.add(feat_shape, deconfusion_feature)
        )
        confusion_encoding = self.gap_conv(confusion_encoding)

        return (
            shape_encoding,
            confusion_encoding,
            heatmap_latent,
            confusion_feature,
        )

    def fuse_model(self):
        self.heatmap_encoder.fuse_model()

        for m in [
            self.conv_shape,
            self.conv_pose,
            self.gap_conv,
            self.fusion_conv,
        ]:
            if hasattr(m, "fuse_model"):
                m.fuse_model()

        for m in self.downsample_encoder:
            if hasattr(m, "fuse_model"):
                m.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        for module in [
            self.heatmap_encoder,
            self.conv_shape,
            self.conv_pose,
            self.gap_conv,
            self.fusion_conv,
        ]:
            if module is not None and hasattr(module, "set_qconfig"):
                module.set_qconfig()

        for module in self.downsample_encoder:
            if module is not None and hasattr(module, "set_qconfig"):
                module.set_qconfig()
