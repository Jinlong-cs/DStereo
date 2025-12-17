# -*- coding:utf-8 -*-
# Copyright (c) Horizon Robotics, All rights reserved.

import logging
from typing import List, Tuple

import torch
from torch import nn

logger = logging.getLogger(__file__)


class CTCPrefixBeamSearch(nn.Module):
    def __init__(self, beam_size: int, vocab_path: str):
        super().__init__()
        self.beam_size = beam_size
        self.vocab_path = vocab_path
        self.read_symbol_table()

    def log_add(self, args: List[int]) -> float:
        """Stable log add."""
        if all(a == -float("inf") for a in args):
            return -float("inf")
        a_max = max(args)
        import math

        lsp = math.log(sum(math.exp(a - a_max) for a in args))
        return a_max + lsp

    def read_symbol_table(self):
        symbol_table = {}
        with open(self.vocab_path, "r", encoding="utf8") as fin:
            for line in fin:
                arr = line.strip().split()
                assert len(arr) == 2
                symbol_table[arr[0]] = int(arr[1])
        return symbol_table

    def forward(
        self, encoder_out: torch.Tensor
    ) -> Tuple[List[List[int]], torch.Tensor]:
        """forward.

        利用CTC prefix beam search 解码器对preds进行解码.
        返回解码后的文本列表.
        Args:
            encoder_out: 预测概率, (T, B, C) C:dim(ctc_out) eg 92 1 4233
        Returns:
            [ctc_prefix_beam_out] :CTC prefix beam search 解码之后的文本
            hyps:beam个解码结果[[id],score]
        """
        maxlen = encoder_out.size(0)
        encoder_out = encoder_out.permute((1, 0, 2)).contiguous()
        ctc_probs = encoder_out  # B T C[1, 92, 4233])
        ctc_probs = ctc_probs.squeeze(0)
        cur_hyps = [((), (0.0, -float("inf")))]
        # 2. CTC beam search step by step
        from collections import defaultdict

        for t in range(0, maxlen):
            logp = ctc_probs[t]  # (vocab_size,)
            # key: prefix, value (pb, pnb), default value(-inf, -inf)
            next_hyps = defaultdict(lambda: (-float("inf"), -float("inf")))
            # 2.1 First beam prune: select topk best
            top_k_logp, top_k_index = logp.topk(self.beam_size)  # (beam_size,)
            for s in top_k_index:
                s = s.item()
                ps = logp[s].item()
                for prefix, (pb, pnb) in cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if s == 0:  # blank
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pb = self.log_add([n_pb, pb + ps, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                    elif s == last:
                        #  Update *ss -> *s;
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pnb = self.log_add([n_pnb, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                        # Update *s-s -> *ss, - is for blank
                        n_prefix = prefix + (s,)
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = self.log_add([n_pnb, pb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)
                    else:
                        n_prefix = prefix + (s,)
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = self.log_add([n_pnb, pb + ps, pnb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)
            # 2.2 Second beam prune
            next_hyps = sorted(
                next_hyps.items(),
                key=lambda x: self.log_add(list(x[1])),
                reverse=True,
            )
            cur_hyps = next_hyps[: self.beam_size]
        hyps = [(y[0], self.log_add([y[1][0], y[1][1]])) for y in cur_hyps]
        aa = hyps[0][0]
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
        ctc_prefix_beam_out = content[0:-1]
        return [ctc_prefix_beam_out], hyps
