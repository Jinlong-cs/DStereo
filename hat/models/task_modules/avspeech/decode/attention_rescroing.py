# -*- coding:utf-8 -*-
# Copyright (c) Horizon Robotics, All rights reserved.

import logging

import torch
from torch import nn

from hat.models.task_modules.avspeech.utils import (
    add_sos_eos,
    reverse_pad_list,
)

logger = logging.getLogger(__file__)


class AttentionRescoring(nn.Module):
    """Attention Rescoring.

    Args:
        beam_size (int): beam size for beam search.
        vocab_path (str): path to vocab file.
    """

    def __init__(self, beam_size: int, vocab_path: str):
        super().__init__()
        self.beam_size = beam_size
        self.vocab_path = vocab_path
        self.read_symbol_table()

    def read_symbol_table(self):
        symbol_table = {}
        with open(self.vocab_path, "r", encoding="utf8") as fin:
            for line in fin:
                arr = line.strip().split()
                assert len(arr) == 2
                symbol_table[arr[0]] = int(arr[1])
        return symbol_table

    def forward(
        self,
        hyps,
        encoder_out,
        sos,
        eos,
        ignore_id,
        att_decoder_embed,
        att_decoder_mask,
        att_decoder,
    ):
        """forward.

        对beam search 解码结果对进行rescore.
        返回解码后的文本列表.

        Args:
            encoder_out: (B C 1 T) C:dim(encoder_out) eg:1, 256, 1, 102.
        Returns:
            [att_rescroing_out]:attention rescoring解码之后的文本.
        """
        # 修改label的标签
        # ys_in       2, 256, 1, 10  hyps_pad    2(beam) 256  1 17
        # ys_mask     2, 1, 10, 10   ys_mask     2 1 7 7
        # encoder_out 2, 256, 1, 47  encoder out 2 44 256
        # mask_pad    2, 1, 1, 47    encoder mask2 1 44
        # att_out     2, 10, 4233
        # hyps = self._ctc_prefix_beam_search(ctc_out,self.beam_size)
        # #返回beam个预测结果 id score
        device = encoder_out.device
        assert len(hyps) == self.beam_size
        from torch.nn.utils.rnn import pad_sequence

        hyps_pad = pad_sequence(
            [
                torch.tensor(hyp[0], device=device, dtype=torch.long)
                for hyp in hyps
            ],
            True,
            ignore_id,
        )  # (beam_size, max_hyps_len)
        ori_hyps_pad = hyps_pad
        hyps_lens = torch.tensor(
            [len(hyp[0]) for hyp in hyps], device=device, dtype=torch.long
        )  # (beam_size,)

        # 修改label的标签
        hyps_pad, _ = add_sos_eos(hyps_pad, sos, eos, ignore_id)  # 2, 7(t+1)
        hyps_pad, _ = att_decoder_embed(hyps_pad)  # 2, 7, 256
        hyps_pad = hyps_pad.unsqueeze(0)
        hyps_pad = hyps_pad.permute((1, 3, 0, 2)).contiguous()
        hyps_lens = hyps_lens + 1  # Add <sos> at begining
        encoder_out = encoder_out.repeat(self.beam_size, 1, 1, 1)
        encoder_mask = torch.ones(
            self.beam_size,
            1,
            1,
            encoder_out.size(3),
            dtype=torch.bool,
            device=device,
        )

        ys_mask, _ = att_decoder_mask(hyps_lens, hyps_pad.size(-1))
        # ys_mask, _ = self.att_decoder_mask(ys_lens + 1, ys_in.size(-1))
        att_out = att_decoder(
            hyps_pad, ys_mask, encoder_out, encoder_mask
        )  # 2, 4233, 1, 7
        # beam_size, max_hyps_len, vocab_size
        att_out = (
            att_out.squeeze(2).permute((0, 2, 1)).contiguous()
        )  # 2, 7, 4233

        decoder_out = att_out
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        decoder_out = decoder_out.cpu().numpy()
        # conventional transformer decoder. used for right to left decoder
        r_hyps_pad = reverse_pad_list(ori_hyps_pad, hyps_lens, ignore_id)
        r_hyps_pad, _ = add_sos_eos(r_hyps_pad, sos, eos, ignore_id)
        r_hyps_pad, _ = att_decoder_embed(r_hyps_pad)  # 2, 7, 256
        r_hyps_pad = r_hyps_pad.unsqueeze(0)
        r_hyps_pad = r_hyps_pad.permute((1, 3, 0, 2)).contiguous()
        # 2, 256, 1, 10
        # 1 10, 15, 256  1 3 0 2
        r_ys_mask, _ = att_decoder_mask(hyps_lens, r_hyps_pad.size(-1))
        r_att_out = att_decoder(
            r_hyps_pad, r_ys_mask, encoder_out, encoder_mask
        )  # 2, 4233, 1, 7
        r_att_out = (
            r_att_out.squeeze(2).permute((0, 2, 1)).contiguous()
        )  # 2, 7, 4233
        r_decoder_out = r_att_out
        r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
        r_decoder_out = r_decoder_out.cpu().numpy()

        best_score = -float("inf")
        best_index = 0
        for i, hyp in enumerate(hyps):
            score = 0.0
            for j, w in enumerate(hyp[0]):
                score += decoder_out[i][j][w]
            score += decoder_out[i][len(hyp[0])][eos]
            # add right to left decoder score
            # if self.reverse_weight > 0:
            #     r_score = 0.0
            #     for j, w in enumerate(hyp[0]):
            #         r_score += r_decoder_out[i][len(hyp[0]) - j - 1][w]
            #     r_score += r_decoder_out[i][len(hyp[0])][self.eos]
            #     score = score * (1 - self.reverse_weight) + r_score
            #               * self.reverse_weight
            # # add ctc score
            # score += hyp[1] * self.ctc_weight
            if score > best_score:
                best_score = score
                best_index = i
        aa = hyps[best_index][0]
        symbol_table = self.read_symbol_table()
        # Load dict
        char_dict = {v: k for k, v in symbol_table.items()}
        eos = len(char_dict) - 1
        content = ""
        for w in aa:
            if w == eos:
                break
            content += char_dict[w]
            content += " "
        att_rescroing_out = content[0:-1]
        return [att_rescroing_out]
