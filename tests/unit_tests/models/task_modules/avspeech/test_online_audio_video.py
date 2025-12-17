import torch

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
    TriHeadFrequencyOnChannelAudioVideoSpeech,
)
from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test


def test_generaltemporal():
    bn_kwargs = {}
    config = dict(
        type="GeneralTemporal",
        bn_kwargs=bn_kwargs,
        num_channels=512,
        num_classes=2,
        lookahead_conv_op=dict(
            type="AdaptiveLookAheadConv",
            bn_kwargs=bn_kwargs,
            num_channels=512,
            use_stride_2_conv=True,
        ),
    )
    module = build_from_registry(config)
    x = torch.rand(80, 512, 1, 15)
    y = module(x)
    assert y.shape == (80, 2, 1, 1)
    qat_test(module, x, with_quantized=False)


def test_TriHeadFrequencyOnChannelAudioVideoSpeech():
    bn_kwargs = {}
    num_cached_frames = 7
    fbank_size = 40
    v2a_factor = 3
    trihead_net = TriHeadFrequencyOnChannelAudioVideoSpeech(
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
    )
    module = build_from_registry(trihead_net)
    v = torch.randn((360, 256, 1, 7))
    a = torch.randn((360, 360, 1, 7))
    av_out, a_out, v_out = module(a, v)
    assert av_out.shape == (360, 2, 1, 1)
    assert a_out.shape == (360, 2, 1, 1)
    assert v_out.shape == (360, 2, 1, 1)
