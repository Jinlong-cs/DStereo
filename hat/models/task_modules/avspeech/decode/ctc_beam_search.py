# Copyright (c) Horizon Robotics, All rights reserved.


from itertools import groupby

import torch
from torch import nn

NEG_INF = -float("inf")


class CTCBeamSearch(nn.Module):
    """CTC Beam Search.

    Args:
        beam_size (int): beam size for beam search.
    """

    def __init__(self, beam_size: int):
        super().__init__()
        self.beam_size = beam_size

    def forward(
        self,
        encoder_out: torch.Tensor,
        beam_size: int,
    ):
        # maxlen = encoder_out.size(0)
        encoder_out = encoder_out.permute((1, 0, 2)).contiguous()
        ctc_probs = encoder_out  # [1, 92, 4233])
        ctc_probs = ctc_probs.squeeze(0)

        log_probs = ctc_probs.cpu().numpy()
        blank = 0
        T, V = log_probs.shape
        # log_probs = np.log(probs)

        beam = [([], 0)]
        # top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)
        for t in range(T):
            new_beam = []
            for prefix, score in beam:
                for i in range(V):
                    new_prefix = prefix + [i]
                    new_score = score + log_probs[t, i]

                    new_beam.append((new_prefix, new_score))

            # top beam_size
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:beam_size]

        # 去除相邻重复和blank
        res = []
        for label, score in beam:
            # 去除相邻重复
            index_list = [key for key, group in groupby(label)]

            # 去除blank
            index_list = [*filter(lambda x: x != blank, index_list)]

            res.append((index_list, score))

        return res
