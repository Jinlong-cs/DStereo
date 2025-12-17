# Copyright (c) Horizon Robotics, All rights reserved.

import logging
from typing import Tuple

import torch
from horizon_plugin_pytorch.quantization import QuantStub
from torch import nn
from torch.quantization import DeQuantStub

from hat.models.task_modules.avspeech.ctc import CTC
from hat.models.task_modules.avspeech.utils import (
    IGNORE_ID,
    add_sos_eos,
    reverse_pad_list,
    th_accuracy,
)
from hat.utils import qconfig_manager

logger = logging.getLogger(__name__)


class BaseSpeechRecognition(torch.nn.Module):
    """Base class for speech recognition models.

    Args:
        vocab_size: vocabulary size.
        encoder: encoder module.
        decoder: decoder module.
        ctc: ctc module.
        ctc_decoder: ctc decoder module.
        att_loss: attention loss module.
        ctc_weight: weight of ctc loss.
        reverse_weight: weight of reverse loss.
        ignore_id: ignore id.
    """

    def __init__(
        self,
        vocab_size: int,
        encoder: nn.Module,
        decoder: nn.Module,
        ctc: nn.Module,
        ctc_decoder: nn.Module,
        att_loss: nn.Module,
        ctc_weight: float = 0.5,
        reverse_weight: float = 0.0,
        ignore_id: int = IGNORE_ID,
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        super().__init__()

        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.reverse_weight = reverse_weight

        self.encoder = encoder
        self.ctc = ctc
        self.ctc_decoder = ctc_decoder
        self.decoder = decoder
        self.att_loss = att_loss

        self.enc_out_dequant = DeQuantStub()
        self.enc_ctc_dequant = DeQuantStub()
        self.decode_quant = QuantStub(scale=None)
        self.rdecode_quant = QuantStub(scale=None)
        self.memory_quant = QuantStub(scale=None)
        self.decode_dequant = DeQuantStub()
        self.rdecode_dequant = DeQuantStub()
        # cnn cache quant and dequant
        self.cnn_cache_quants = nn.ModuleList(
            QuantStub(scale=None) for _ in range(self.encoder.num_layers)
        )
        self.att_cache_quants = nn.ModuleList(
            QuantStub(scale=None) for _ in range(self.encoder.num_layers)
        )
        self.cnn_cache_dequants = nn.ModuleList(
            DeQuantStub() for _ in range(self.encoder.num_layers)
        )
        self.att_cache_dequants = nn.ModuleList(
            DeQuantStub() for _ in range(self.encoder.num_layers)
        )

    def forward_decoder(
        self, encoder_out, encoder_out_lens, encoder_mask, label, label_lens
    ):
        # 2a. ctc branch
        if self.ctc_weight != 0.0:
            ctc_out = self.ctc(encoder_out)
            ctc_out = self.enc_ctc_dequant(ctc_out)

            loss_ctc = self.ctc.cal_ctc_loss(
                ctc_out, encoder_out_lens, label, label_lens
            )
            with torch.no_grad():
                ctc_out = ctc_out.squeeze(2).permute(0, 2, 1)
                ctc_result = self.ctc_decoder(
                    ctc_out.log_softmax(2), encoder_out_lens
                )
        else:
            loss_ctc = None

        # 2b. Attention-decoder branch
        if self.ctc_weight != 1.0:
            encoder_out = self.enc_out_dequant(encoder_out)
            loss_att, acc_att = self._calc_att_loss(
                encoder_out, encoder_mask, label, label_lens
            )
        else:
            loss_att = None
            acc_att = None

        if loss_ctc is None:
            loss = loss_att
        elif loss_att is None:
            loss = loss_ctc
        else:
            loss = (
                self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att
            )

        return loss, loss_att, loss_ctc, acc_att, ctc_result

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        ys_in_pad, ys_out_pad = add_sos_eos(
            ys_pad, self.sos, self.eos, self.ignore_id
        )
        ys_in_lens = ys_pad_lens + 1

        # reverse the seq, used for right to left decoder
        r_ys_pad = reverse_pad_list(ys_pad, ys_pad_lens, float(self.ignore_id))
        r_ys_in_pad, r_ys_out_pad = add_sos_eos(
            r_ys_pad, self.sos, self.eos, self.ignore_id
        )
        # 1. Forward decoder

        x, r_x = self.decoder.forward_embedding(ys_in_pad, r_ys_in_pad)

        encoder_out = self.memory_quant(encoder_out)
        x = self.decode_quant(x)
        r_x = self.rdecode_quant(r_x)
        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out,
            encoder_mask,
            x,
            ys_in_pad,
            ys_in_lens,
            r_x,
            r_ys_in_pad,
            self.reverse_weight,
        )
        decoder_out = self.decode_dequant(decoder_out)
        r_decoder_out = self.rdecode_dequant(r_decoder_out)
        # 2. Compute attention loss
        loss_att = self.att_loss(decoder_out, ys_out_pad)
        r_loss_att = torch.tensor(0.0)
        if self.reverse_weight > 0.0:
            r_loss_att = self.att_loss(r_decoder_out, r_ys_out_pad)
        loss_att = (
            loss_att * (1 - self.reverse_weight)
            + r_loss_att * self.reverse_weight
        )
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        return loss_att, acc_att

    def quant_dequant_caches(self, cnn_caches, att_caches):
        # dequant
        cnn_caches = [
            cache_dequant(cnn_cache)
            for cnn_cache, cache_dequant in zip(
                cnn_caches, self.cnn_cache_dequants
            )
        ]
        att_caches = [
            cache_dequant(att_cache)
            for att_cache, cache_dequant in zip(
                att_caches, self.att_cache_dequants
            )
        ]
        # quant
        cnn_caches = [
            cache_quant(cnn_cache)
            for cnn_cache, cache_quant in zip(
                cnn_caches, self.cnn_cache_quants
            )
        ]
        att_caches = [
            cache_quant(att_cache)
            for att_cache, cache_quant in zip(
                att_caches, self.att_cache_quants
            )
        ]


class AVSpeechRecognition(BaseSpeechRecognition):
    """CTC-attention hybrid E2E Encoder-Decoder model."""

    def __init__(
        self,
        vocab_size: int,
        vfea_extractor: nn.Module,
        audio_cmvn: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        ctc: nn.Module,
        ctc_decoder: nn.Module,
        att_loss: nn.Module,
        ctc_weight: float = 0.5,
        reverse_weight: float = 0.0,
        ignore_id: int = IGNORE_ID,
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        super().__init__(
            vocab_size=vocab_size,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            ctc_decoder=ctc_decoder,
            att_loss=att_loss,
            ctc_weight=ctc_weight,
            reverse_weight=reverse_weight,
            ignore_id=ignore_id,
        )
        # note that eos is the same as sos (equivalent ID)

        self.vfea_extractor = vfea_extractor
        self.audio_cmvn = audio_cmvn

        # quant and dequant blocks
        self.images_quant = QuantStub(scale=1.0 / 128)  # YUV部署时不用 scale.
        self.images_dequant = DeQuantStub()
        self.afea_quant = QuantStub(scale=None)
        self.vfea_quant = QuantStub(scale=None)

    def forward(self, data):
        valid = data.get("valid", ["images", "audio"])
        # VFEA EXTRACTOR
        vfea = data["images"]
        b, t, *_ = vfea.size()
        if "images" in valid:
            b, t, c, h, w = vfea.size()
            input_layout = getattr(self.vfea_extractor, "input_layout", "BCHW")
            if input_layout == "BCTHW":
                vfea = vfea.transpose(2, 1)
            elif input_layout == "BCHW":
                vfea = vfea.view(-1, c, h, w)
            else:
                raise NotImplementedError

            vfea = self.images_quant(vfea)
            vfea = self.vfea_extractor(vfea)
            vfea = self.images_dequant(vfea)
            # B T C 1 -> B C 1 T

        vfea = vfea.view(b, t, -1, 1).contiguous().permute(0, 2, 3, 1)

        afea = data["audio"]
        # afea: B T D -> B 1 T D
        afea = afea.unsqueeze(1)
        # if "audio" in valid:
        afea = self.audio_cmvn(afea)

        vfea = self.vfea_quant(vfea)
        afea = self.afea_quant(afea)

        pad_mask = data["pad_mask"]
        chunk_mask = data["chunk_mask"]

        # 1.  encoder
        encoder_out, encoder_mask, att_caches, cnn_caches = self.encoder(
            afea, vfea, chunk_mask, pad_mask
        )
        self.quant_dequant_caches(cnn_caches, att_caches)
        encoder_mask = encoder_mask.squeeze(1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        label = data["label"]
        label_lens = data["label_length"]
        assert label_lens.dim() == 1, label_lens.shape

        # 2 decoder and loss
        loss, loss_att, loss_ctc, acc_att, ctc_result = self.forward_decoder(
            encoder_out, encoder_out_lens, encoder_mask, label, label_lens
        )

        if "loss_weight" in data:
            loss = loss * data["loss_weight"]
        return loss, loss_att, loss_ctc, acc_att, ctc_result

    def fuse_model(self):
        self.vfea_extractor.fuse_model()
        self.encoder.fuse_model()
        self.ctc.fuse_model()

    def set_calibration_qconfig(self):
        self.qconfig = qconfig_manager.get_default_calibration_qconfig()
        self.audio_cmvn.qconfig = None
        self.decoder.left_decoder.embed.qconfig = None
        self.decoder.right_decoder.embed.qconfig = None

    def set_qconfig(self):
        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        self.audio_cmvn.qconfig = None
        self.decoder.left_decoder.embed.qconfig = None
        self.decoder.right_decoder.embed.qconfig = None

    def trace_vfea_extractor(self, images):
        images = self.images_quant(images)
        vfea = self.vfea_frontend(images)
        vfea = self.images_dequant(vfea)
        return vfea

    def trace_encoder_and_ctc(self, cnn_caches, afea, vfea):
        afea = self.afea_quant(afea)
        vfea = self.vfea_quant(vfea)
        cnn_caches = [
            cache_quant(cnn_cache)
            for cnn_cache, cache_quant in zip(
                cnn_caches, self.cnn_cache_quants
            )
        ]
        encoder_out, _, new_att_caches, new_cnn_caches = self.encoder.forward(
            afea=afea,
            afea_lens=None,
            vfea=vfea,
            vfea_lens=None,
            att_caches=None,
            cnn_caches=cnn_caches,
        )

        ctc_out = self.ctc(encoder_out)
        # 按顺序输出 cache 靠前
        new_cnn_caches = [
            cache_dequant(cnn_cache)
            for cnn_cache, cache_dequant in zip(
                new_cnn_caches, self.cnn_cache_dequants
            )
        ]
        # enc out 倒数第二
        encoder_out = self.enc_out_dequant(encoder_out)
        # ctc_out 倒数第一
        ctc_out = self.enc_ctc_dequant(ctc_out)
        # return encoder_out, ctc_out, *ret_cnn_caches
        return new_cnn_caches, encoder_out, ctc_out


class VisualSpeechRecognition(BaseSpeechRecognition):
    """CTC-attention hybrid E2E Encoder-Decoder model."""

    def __init__(
        self,
        vocab_size: int,
        vfea_extractor: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        ctc: CTC,
        ctc_decoder: nn.Module,
        att_loss: nn.Module,
        ctc_weight: float = 0.5,
        reverse_weight: float = 0.0,
        ignore_id: int = IGNORE_ID,
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        super().__init__(
            vocab_size=vocab_size,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            ctc_decoder=ctc_decoder,
            att_loss=att_loss,
            ctc_weight=ctc_weight,
            reverse_weight=reverse_weight,
            ignore_id=ignore_id,
        )

        self.vfea_extractor = vfea_extractor

        # quant and dequant blocks
        self.images_quant = QuantStub(scale=1.0 / 128)  # YUV部署时不用 scale.
        self.images_dequant = DeQuantStub()
        self.vfea_quant = QuantStub(scale=None)

    def forward(self, data):
        vfea = data["images"]
        pad_mask = data["pad_mask"]
        chunk_mask = data["chunk_mask"]

        label = data["label"]
        label_lens = data["label_length"]
        assert label_lens.dim() == 1, label_lens.shape

        # VFEA EXTRACTOR
        b, t, c, h, w = vfea.size()
        input_layout = getattr(self.vfea_extractor, "input_layout", "BCHW")
        if input_layout == "BCTHW":
            vfea = vfea.transpose(2, 1)
        elif input_layout == "BCHW":
            vfea = vfea.view(-1, c, h, w)
        else:
            raise NotImplementedError

        vfea = self.images_quant(vfea)
        vfea = self.vfea_extractor(vfea)
        vfea = self.images_dequant(vfea)
        # B T C 1 -> B C 1 T
        vfea = vfea.view(b, t, -1, 1).contiguous().permute(0, 2, 3, 1)

        vfea = self.vfea_quant(vfea)

        encoder_out, encoder_mask, att_caches, cnn_caches = self.encoder(
            vfea, chunk_mask, pad_mask
        )
        self.quant_dequant_caches(cnn_caches, att_caches)

        # 2. decoder and loss
        encoder_mask = encoder_mask.squeeze(1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        loss, loss_att, loss_ctc, acc_att, ctc_result = self.forward_decoder(
            encoder_out, encoder_out_lens, encoder_mask, label, label_lens
        )

        if "loss_weight" in data:
            loss = loss * data["loss_weight"]
        return loss, loss_att, loss_ctc, acc_att, ctc_result

    def fuse_model(self):
        self.encoder.fuse_model()

    def set_calibration_qconfig(self):
        self.qconfig = qconfig_manager.get_default_calibration_qconfig()
        self.decoder.left_decoder.embed.qconfig = None
        self.decoder.right_decoder.embed.qconfig = None

    def set_qconfig(self):
        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        self.decoder.left_decoder.embed.qconfig = None
        self.decoder.right_decoder.embed.qconfig = None

    def trace_vfea_extractor(self, images):
        images = self.images_quant(images)
        vfea = self.vfea_extractor(images)
        vfea = self.images_dequant(vfea)
        return vfea

    def trace_encoder_and_ctc(self, cnn_caches, afea, vfea):
        afea = self.afea_quant(afea)
        vfea = self.vfea_quant(vfea)
        cnn_caches = [
            cache_quant(cnn_cache)
            for cnn_cache, cache_quant in zip(
                cnn_caches, self.cnn_cache_quants
            )
        ]
        encoder_out, _, new_att_caches, new_cnn_caches = self.encoder.forward(
            afea=afea,
            afea_lens=None,
            vfea=vfea,
            vfea_lens=None,
            att_caches=None,
            cnn_caches=cnn_caches,
        )

        ctc_out = self.ctc(encoder_out)
        # 按顺序输出 cache 靠前
        new_cnn_caches = [
            cache_dequant(cnn_cache)
            for cnn_cache, cache_dequant in zip(
                new_cnn_caches, self.cnn_cache_dequants
            )
        ]
        # enc out 倒数第二
        encoder_out = self.enc_out_dequant(encoder_out)
        # ctc_out 倒数第一
        ctc_out = self.enc_ctc_dequant(ctc_out)
        # return encoder_out, ctc_out, *ret_cnn_caches
        return new_cnn_caches, encoder_out, ctc_out


class AcousticsSpeechRecognition(BaseSpeechRecognition):
    """CTC-attention hybrid E2E Encoder-Decoder model."""

    def __init__(
        self,
        vocab_size: int,
        audio_cmvn: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        ctc: CTC,
        ctc_decoder: nn.Module,
        att_loss: nn.Module,
        ctc_weight: float = 0.5,
        reverse_weight: float = 0.0,
        ignore_id: int = IGNORE_ID,
    ):
        super().__init__(
            vocab_size=vocab_size,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            ctc_decoder=ctc_decoder,
            att_loss=att_loss,
            ctc_weight=ctc_weight,
            reverse_weight=reverse_weight,
            ignore_id=ignore_id,
        )

        self.audio_cmvn = audio_cmvn
        # quant and dequant blocks
        self.afea_quant = QuantStub(scale=None)

    def forward(self, data):
        pad_mask = data["pad_mask"]
        chunk_mask = data["chunk_mask"]

        label = data["label"]
        label_lens = data["label_length"]
        assert label_lens.dim() == 1, label_lens.shape

        afea = data["audio"]
        # afea: B T D -> B 1 T D
        # afea = afea.unsqueeze(1)
        # if "audio" in valid:
        afea = self.audio_cmvn(afea)

        afea = self.afea_quant(afea)
        chunk_mask = chunk_mask.squeeze(1)
        pad_mask = pad_mask.squeeze(1)
        encoder_out, encoder_mask, att_caches, cnn_caches = self.encoder(
            afea, chunk_mask, pad_mask
        )
        self.quant_dequant_caches(cnn_caches, att_caches)

        # 2. decoder and loss
        # encoder_mask = encoder_mask.squeeze(1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        loss, loss_att, loss_ctc, acc_att, ctc_result = self.forward_decoder(
            encoder_out, encoder_out_lens, encoder_mask, label, label_lens
        )

        if "loss_weight" in data:
            loss = loss * data["loss_weight"]
        return loss, loss_att, loss_ctc, acc_att, ctc_result
