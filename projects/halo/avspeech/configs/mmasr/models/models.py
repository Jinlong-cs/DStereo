# flake8: noqa

from torch import nn

from hat.models.backbones.resnet import ResNet18
from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.losses.smoothing_loss import LabelSmoothingLoss
from hat.models.structures.avspeech.mmasr import (
    AcousticsSpeechRecognition,
    AVSpeechRecognition,
    VisualSpeechRecognition,
)
from hat.models.task_modules.avspeech.avspeech_fusion import CatFusion
from hat.models.task_modules.avspeech.conformer.decoder import (
    BiTransformerDecoder,
)
from hat.models.task_modules.avspeech.conformer.encoder import ConformerEncoder
from hat.models.task_modules.avspeech.ctc import CTC
from hat.models.task_modules.avspeech.decode.ctc_greedy import CTCGreedyDecoder
from hat.models.task_modules.avspeech.fusionformer.decoder import (
    BiTransformerCnnDecoder,
)
from hat.models.task_modules.avspeech.fusionformer.encoder import (
    FusionEncoder,
    FusionformerEncoder,
)
from hat.models.task_modules.avspeech.subsampling import (
    CausalSubsampling2,
    Conv2dSubsampling8,
)
from hat.models.task_modules.avspeech.utils import IGNORE_ID
from projects.halo.avspeech.utils.global_config import get_config


def get_avsr_fusionformer(config):
    """

    Args:
        config (edict):
            bn_kwargs (dict): batchnorm的参数
            mode (str): 可选项["AV", "AO", "VO"]
            global_cmvn (nn.Module): 用于归一化音频特征
            encoder_output_size (int): encoder output channnel size

    Returns:
        dict: model的定义
    """
    size = config.encoder_output_size
    bn_kwargs = dict(config.bn_kwargs)
    dropout_rate = 0.1

    model = dict(
        type=AVSpeechRecognition,
        vocab_size=get_config("vocab_size"),
        vfea_extractor=dict(
            type=ResNet18,
            num_classes=256,
            bn_kwargs=bn_kwargs,
            include_top=True,
            flat_output=True,
            top_layer=dict(
                type=ConvModule2d,
                in_channels=512,
                out_channels=size,
                kernel_size=3,
                bias=True,
                norm_layer=nn.BatchNorm2d(size, **bn_kwargs),
            ),
            quant_input=False,
            dequant_output=False,
        ),
        audio_cmvn=config.global_cmvn,
        encoder=dict(
            type=FusionEncoder,
            audio_encoder=dict(
                type=FusionformerEncoder,
                embed=dict(
                    type=Conv2dSubsampling8,
                    idim=80,
                    odim=size,
                    dropout_rate=dropout_rate,
                ),
                output_size=size,
                attention_heads=4,
                linear_units=2048,
                cnn_inner_channel=512,
                num_blocks=6,
                dropout_rate=dropout_rate,
                attention_dropout_rate=dropout_rate,
                activation=nn.ReLU(),
                cnn_module_kernel=7,
                causal=True,
                final_norm="layer_norm",
            ),
            video_encoder=dict(
                type=FusionformerEncoder,
                embed=dict(
                    type=CausalSubsampling2,
                    idim=256,
                    odim=256,
                    dropout_rate=0.1,
                    bn_kwargs=bn_kwargs,
                ),
                output_size=size,
                attention_heads=4,
                linear_units=2048,
                cnn_inner_channel=512,
                num_blocks=6,
                dropout_rate=dropout_rate,
                attention_dropout_rate=dropout_rate,
                activation=nn.ReLU(),
                cnn_module_kernel=7,
                causal=True,
                final_norm="layer_norm",
            ),
            fusion_block=dict(type=CatFusion, idim=256, odim=256),
            av_encoder=dict(
                type=FusionformerEncoder,
                embed=nn.Identity(),
                output_size=size,
                attention_heads=4,
                linear_units=2048,
                cnn_inner_channel=512,
                num_blocks=6,
                dropout_rate=dropout_rate,
                attention_dropout_rate=dropout_rate,
                activation=nn.ReLU(),
                cnn_module_kernel=7,
                causal=True,
                final_norm="layer_norm",
            ),
        ),
        decoder=dict(
            type=BiTransformerCnnDecoder,
            vocab_size=get_config("vocab_size"),
            encoder_output_size=size,
            attention_heads=4,
            dropout_rate=0.1,
            linear_units=2048,
            num_blocks=3,
            positional_dropout_rate=0.1,
            r_num_blocks=3,
            self_attention_dropout_rate=0.1,
            src_attention_dropout_rate=0.1,
        ),
        ctc=dict(
            type=CTC,
            odim=get_config("vocab_size"),
            encoder_output_size=size,
        ),
        ctc_decoder=dict(
            type=CTCGreedyDecoder,
            vocab_path=get_config("symbol_table_path"),
            blank_id=0,
            keep_blank=True,
        ),
        att_loss=dict(
            type=LabelSmoothingLoss,
            size=get_config("vocab_size"),
            padding_idx=IGNORE_ID,
            smoothing=0.1,
            normalize_length=False,
        ),
        ctc_weight=0.3,
        ignore_id=IGNORE_ID,
        reverse_weight=0.3,
    )

    return model


def get_vsr_conformer(config):
    """
    Args:
        config (edict):
            bn_kwargs (dict): batchnorm的参数
            mode (str): 可选项["AV", "AO", "VO"]
            global_cmvn (nn.Module): 用于归一化音频特征
            encoder_output_size (int): encoder output channnel size

    Returns:
        dict: model的定义
    """
    size = config.encoder_output_size
    dropout_rate = 0.1
    bn_kwargs = dict(config.bn_kwargs)

    model = dict(
        type=VisualSpeechRecognition,
        vocab_size=get_config("vocab_size"),
        vfea_extractor=dict(
            type=ResNet18,
            num_classes=256,
            bn_kwargs=bn_kwargs,
            include_top=True,
            flat_output=True,
            top_layer=dict(
                type=ConvModule2d,
                in_channels=512,
                out_channels=size,
                kernel_size=3,
                bias=True,
                norm_layer=nn.BatchNorm2d(size, **bn_kwargs),
            ),
            quant_input=False,
            dequant_output=False,
        ),
        encoder=dict(
            type=ConformerEncoder,
            embed=CausalSubsampling2(
                idim=256,
                odim=256,
                dropout_rate=0.1,
                bn_kwargs=bn_kwargs,
            ),
            output_size=size,
            attention_heads=4,
            linear_units=2048,
            cnn_inner_channel=512,
            num_blocks=6,
            dropout_rate=dropout_rate,
            attention_dropout_rate=dropout_rate,
            activation=nn.ReLU(),
            cnn_module_kernel=7,
            causal=True,
            final_norm="layer_norm",
        ),
        decoder=dict(
            type=BiTransformerDecoder,
            vocab_size=get_config("vocab_size"),
            encoder_output_size=size,
            attention_heads=4,
            linear_units=2048,
            num_blocks=3,
            r_num_blocks=3,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            self_attention_dropout_rate=0.1,
            src_attention_dropout_rate=0.1,
        ),
        ctc=dict(
            type=CTC,
            odim=get_config("vocab_size"),
            encoder_output_size=size,
        ),
        ctc_decoder=dict(
            type=CTCGreedyDecoder,
            vocab_path=get_config("symbol_table_path"),
            blank_id=0,
            keep_blank=True,
        ),
        ctc_weight=0.3,
        ignore_id=IGNORE_ID,
        reverse_weight=0.3,
        att_loss=dict(
            type=LabelSmoothingLoss,
            size=get_config("vocab_size"),
            padding_idx=IGNORE_ID,
            smoothing=0.1,
            normalize_length=False,
        ),
    )

    return model


def get_avsr_conformer(config):
    """
    Args:
        config (edict):
            bn_kwargs (dict): batchnorm的参数
            mode (str): 可选项["AV", "AO", "VO"]
            global_cmvn (nn.Module): 用于归一化音频特征
            encoder_output_size (int): encoder output channnel size

    Returns:
        dict: model的定义
    """
    size = config.encoder_output_size
    dropout_rate = 0.1
    bn_kwargs = dict(config.bn_kwargs)
    ignore_id = IGNORE_ID

    model = dict(
        type=AVSpeechRecognition,
        vocab_size=get_config("vocab_size"),
        vfea_extractor=dict(
            type=ResNet18,
            num_classes=size,
            bn_kwargs=bn_kwargs,
            include_top=True,
            flat_output=True,
            top_layer=dict(
                type=ConvModule2d,
                in_channels=512,
                out_channels=size,
                kernel_size=3,
                bias=True,
                norm_layer=nn.BatchNorm2d(size, **bn_kwargs),
            ),
            # quant_input=False,
            # dequant_output=False,
        ),
        audio_cmvn=config.global_cmvn,
        encoder=dict(
            type=FusionEncoder,
            audio_encoder=dict(
                type=ConformerEncoder,
                embed=dict(
                    type=Conv2dSubsampling8,
                    idim=80,
                    odim=size,
                    dropout_rate=dropout_rate,
                ),
                output_size=size,
                attention_heads=4,
                linear_units=2048,
                cnn_inner_channel=512,
                num_blocks=6,
                dropout_rate=dropout_rate,
                attention_dropout_rate=dropout_rate,
                activation=nn.ReLU(),
                cnn_module_kernel=7,
                final_norm="layer_norm",
            ),
            video_encoder=dict(
                type=ConformerEncoder,
                embed=dict(
                    type=CausalSubsampling2,
                    idim=size,
                    odim=size,
                    dropout_rate=0.1,
                    bn_kwargs=bn_kwargs,
                ),
                output_size=size,
                attention_heads=4,
                linear_units=2048,
                cnn_inner_channel=512,
                num_blocks=6,
                dropout_rate=dropout_rate,
                attention_dropout_rate=dropout_rate,
                activation=nn.ReLU(),
                cnn_module_kernel=7,
                final_norm="layer_norm",
            ),
            fusion_block=dict(type=CatFusion, idim=size, odim=size),
            av_encoder=dict(
                type=ConformerEncoder,
                embed=nn.Identity(),
                output_size=size,
                attention_heads=4,
                linear_units=2048,
                cnn_inner_channel=512,
                num_blocks=6,
                dropout_rate=dropout_rate,
                attention_dropout_rate=dropout_rate,
                activation=nn.ReLU(),
                cnn_module_kernel=7,
                final_norm="layer_norm",
            ),
            norm_before_fusion=True,
        ),
        decoder=dict(
            type=BiTransformerDecoder,
            vocab_size=get_config("vocab_size"),
            encoder_output_size=size,
            attention_heads=4,
            linear_units=2048,
            num_blocks=3,
            r_num_blocks=3,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            self_attention_dropout_rate=0.1,
            src_attention_dropout_rate=0.1,
        ),
        ctc=dict(
            type=CTC,
            odim=get_config("vocab_size"),
            encoder_output_size=size,
        ),
        ctc_decoder=dict(
            type=CTCGreedyDecoder,
            vocab_path=get_config("symbol_table_path"),
            blank_id=0,
            keep_blank=True,
        ),
        ctc_weight=0.3,
        ignore_id=ignore_id,
        reverse_weight=0.3,
        att_loss=dict(
            type=LabelSmoothingLoss,
            size=get_config("vocab_size"),
            padding_idx=ignore_id,
            smoothing=0.1,
            normalize_length=False,
        ),
    )

    return model
