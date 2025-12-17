# Copyright (c) Horizon Robotics. All rights reserved.
import torch

from hat.models.backbones.vargnetv2 import CocktailVargNetV2
from hat.models.losses.focal_loss import SoftmaxFocalLoss
from hat.models.structures.mmvad_structure import TriMMVADStructure
from hat.models.task_modules.avspeech.avspeech_feature import (
    CocktailFCLayer,
    FrequencyOnChannelAudioFeatureExtractor,
    VideoInterFrameEncoder,
)
from hat.models.task_modules.avspeech.avspeech_fusion import (
    AddFusion,
    VideoAudioDefaultFusion,
)
from hat.models.task_modules.avspeech.lookaheadconv import (
    AdaptiveLookAheadConv,
)
from hat.models.task_modules.avspeech.online_audio_video import (
    CocktailTopLayer,
    TriHeadFrequencyOnChannelAudioVideoSpeech,
)
from hat.registry import build_from_registry


def test_TriMMVADStructure():
    batchsize = 8
    seqlen = 90
    fbank_size = 40
    v2a_factor = 3
    bn_kwargs = {}
    num_cached_frames = 7

    model = TriMMVADStructure(
        vision_net=CocktailVargNetV2(
            bn_kwargs=bn_kwargs,
            flat_output=False,
            top_layer=CocktailTopLayer(
                num_channels=256,
            ),
        ),
        trihead_net=TriHeadFrequencyOnChannelAudioVideoSpeech(
            video_inter_frame_encoder=VideoInterFrameEncoder(
                in_channels=256,
                out_channels=256,
                last_layer_with_relu=True,
                bn_kwargs=bn_kwargs,
            ),
            audio_extractor=FrequencyOnChannelAudioFeatureExtractor(
                in_channels=fbank_size * v2a_factor * 3,
                out_channels=256,
                encoder_kernel_3=False,
                last_layer_with_relu=True,
                dropout_rate=0,
                bn_kwargs=bn_kwargs,
            ),
            fusion_block=VideoAudioDefaultFusion(
                fusion_block=AddFusion(
                    bn_kwargs=bn_kwargs,
                    v_in_channels=256,
                    a_in_channels=256,
                    a_v_out_channels=1024,
                    fusion_out_channels=512,
                ),
                lookahead_conv_net=AdaptiveLookAheadConv(
                    num_channels=512,
                    num_cached_frames=num_cached_frames,
                    bn_kwargs=bn_kwargs,
                    num_cached_padding=0,
                    channels_factor=1.0,
                    use_stride_2_conv=False,
                ),
            ),
            av_classifier=CocktailFCLayer(
                in_channels=512,
                num_classes=2,
                bias=False,
            ),
            audio_lookahead_net=AdaptiveLookAheadConv(
                num_channels=256,
                num_cached_frames=num_cached_frames,
                bn_kwargs=bn_kwargs,
            ),
            audio_classifier=CocktailFCLayer(
                in_channels=256,
                num_classes=2,
                bias=True,
            ),
            video_lookahead_net=AdaptiveLookAheadConv(
                num_channels=256,
                num_cached_frames=num_cached_frames,
                bn_kwargs=bn_kwargs,
            ),
            video_classifier=CocktailFCLayer(
                in_channels=256,
                num_classes=2,
                bias=True,
            ),
        ),
        loss=SoftmaxFocalLoss(
            loss_name="focal_loss",
            num_classes=2,
            reduction="mean",
            weight=1.0,
        ),
        loss_weight=[1.0, 1.0, 1.0],
        num_cached_frames=num_cached_frames,
        replicate_type="zero",
    )

    module = build_from_registry(model)
    imgs = torch.rand(batchsize, seqlen, 3, 96, 96)
    audio = torch.rand(batchsize, seqlen, 3, 120)
    x = {
        "images": imgs,
        "audio": audio,
        "labels": torch.rand(batchsize, seqlen),
        "masks": torch.rand(batchsize, seqlen),
    }
    losses, av_loss, a_loss, v_loss, label, av_out, a_out, v_out = module(x)
    assert label.shape == (batchsize * seqlen, 1)
    assert av_out.shape == (batchsize * seqlen, 2)
    assert a_out.shape == (batchsize * seqlen, 2)
    assert v_out.shape == (batchsize * seqlen, 2)
