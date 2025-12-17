# Copyright (c) Horizon Robotics, All rights reserved.

from typing import Optional

import torch
from horizon_plugin_pytorch.quantization import QuantStub
from torch import nn
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "GeneralTemporal",
    "CocktailTopLayer",
    "TriHeadFrequencyOnChannelAudioVideoSpeech",
]


@OBJECT_REGISTRY.register
class GeneralTemporal(nn.Module):
    def __init__(
        self,
        num_classes,
        bn_kwargs: dict,
        num_channels: int,
        lookahead_conv_op: nn.Module,
        disable_quanti_input: bool = False,
        disable_dequanti_output: bool = False,
        bias: bool = True,
    ):
        super(GeneralTemporal, self).__init__()
        self.bn_kwargs = bn_kwargs
        self.num_classes = num_classes
        self.bias = bias
        self.disable_quanti_input = disable_quanti_input
        self.disable_dequanti_output = disable_dequanti_output
        self.lookahead_conv_block = lookahead_conv_op
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.output = ConvModule2d(
            num_channels, self.num_classes, 1, 1, 0, bias=self.bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x if self.disable_quanti_input else self.quant(x)
        x = self.lookahead_conv_block(x)
        xout = self.output(x)
        xout = xout if self.disable_dequanti_output else self.dequant(xout)
        return xout

    def fuse_model(self):
        modules = [self.lookahead_conv_block, self.output]
        if self.output is not None:
            modules.append(self.output)
        for mod in modules:
            if isinstance(mod, nn.Sequential):
                for m in mod:
                    if hasattr(m, "fuse_model"):
                        m.fuse_model()
            elif hasattr(mod, "fuse_model"):
                mod.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        if self.output is not None:
            if hasattr(self.output, "set_qconfig"):
                self.output[
                    -1
                ].qconfig = qconfig_manager.get_default_qat_out_qconfig()


@OBJECT_REGISTRY.register
class CocktailTopLayer(nn.Module):
    def __init__(
        self,
        num_channels: int,
        bias: bool = True,
    ):
        super(CocktailTopLayer, self).__init__()
        self.bias = bias
        self.output = ConvModule2d(
            num_channels, num_channels, 3, 1, 0, bias=self.bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xout = self.output(x)
        return xout


@OBJECT_REGISTRY.register
class TriHeadFrequencyOnChannelAudioVideoSpeech(nn.Module):
    """TriHeadFrequencyOnChannelAudioVideoSpeech.

    Args:
        video_inter_frame_encoder: Visual feature encoder.
        audio_extractor: Audio feature encoder.
        fusion_block: Audio and visual feature fusion module.
        av_classifier: Audio and visual fusion classifier.
        audio_lookahead_net: Audio timing feature extraction network.
        video_lookahead_net: Video timing feature extraction network.
        audio_classifier: Audio classifier.
        video_classifier: Video classifier.

    """

    def __init__(
        self,
        video_inter_frame_encoder: nn.Module,
        audio_extractor: nn.Module,
        fusion_block: nn.Module,
        av_classifier: nn.Module,
        audio_lookahead_net: nn.Module = None,
        video_lookahead_net: nn.Module = None,
        audio_classifier: Optional[nn.Module] = None,
        video_classifier: Optional[nn.Module] = None,
    ):
        super(TriHeadFrequencyOnChannelAudioVideoSpeech, self).__init__()
        self.video_inter_frame_encoder = video_inter_frame_encoder
        self.audio_feature_extractor = audio_extractor
        self.fusion_block = fusion_block
        self.audio_lookahead_net = audio_lookahead_net
        self.video_lookahead_net = video_lookahead_net
        self.av_classifier = av_classifier
        self.audio_classifier = audio_classifier
        self.video_classifier = video_classifier

    def forward(self, ax: torch.Tensor, vx: torch.Tensor):
        ax = self.audio_feature_extractor(ax)
        vx = self.video_inter_frame_encoder(vx)
        av_x = self.fusion_block(ax, vx)

        av_out = self.av_classifier(av_x)
        a_out = None
        v_out = None
        if self.audio_classifier is not None:
            ax = self.audio_lookahead_net(ax)
            a_out = self.audio_classifier(ax)

        if self.video_classifier is not None:
            vx = self.video_lookahead_net(vx)
            v_out = self.video_classifier(vx)

        return av_out, a_out, v_out

    def fuse_model(self):
        modules = [
            self.video_inter_frame_encoder,
            self.audio_feature_extractor,
            self.fusion_block,
            self.av_classifier,
            self.audio_classifier,
            self.video_classifier,
        ]
        for mod in modules:
            if isinstance(mod, nn.Sequential):
                for m in mod:
                    if hasattr(m, "fuse_model"):
                        m.fuse_model()
            elif hasattr(mod, "fuse_model"):
                mod.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        if self.av_classifier is not None:
            self.av_classifier.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
        if self.audio_classifier is not None:
            self.audio_classifier.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
        if self.video_classifier is not None:
            self.video_classifier.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
