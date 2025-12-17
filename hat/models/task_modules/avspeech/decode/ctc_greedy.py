# Copyright (c) Horizon Robotics, All rights reserved.

import logging

import torch
from torch import nn

from hat.utils.filesystem import get_filesystem

logger = logging.getLogger(__file__)


class CTCGreedyDecoder(nn.Module):

    BLANK_TOKEN_STR = "<blank>"

    def __init__(self, vocab_path: str, blank_id=0, keep_blank=True):
        super().__init__()
        self.vocab_path = vocab_path
        self.blank_id = blank_id
        self.keep_blank = keep_blank
        self._load_vocab_path()

    def _load_vocab_path(self):
        fs = get_filesystem(self.vocab_path)
        cls2token_vocab = {}
        with fs.open(self.vocab_path, mode="r", encodings="utf-8") as fr:
            for line in fr:
                token, cls_id = line.strip().split()
                if token == "<blank>":
                    self.blank_id = int(cls_id)
                    msg = f"词表中包含 <blank>, 将blank_id设置为 {self.blank_id}"
                    logger.info(msg)
                cls2token_vocab[int(cls_id)] = token
        self.cls2token_vocab = cls2token_vocab

    def forward(self, preds: torch.Tensor, pred_lens: torch.Tensor):
        """forward.

        利用CTC贪心解码器对preds进行解码.
        返回解码后的文本列表.
        即使 ``preds`` 是

        Args:
            preds: 预测概率, (B, T, C) or (T, C)
            pred_lens: 输入的有效长度, (B)

        Returns:
            CTC贪心解码之后的结果.
        """
        indices = torch.argmax(preds, dim=-1)
        if len(indices.size()) == 1:
            indices = indices.unsqueeze(0)
        indices, inverse = torch.unique_consecutive(
            indices, return_inverse=True
        )

        def callback(_indices: torch.Tensor):
            res = [
                self.cls2token_vocab[int(idx)]
                for idx in _indices
                if idx != self.blank_id
            ]
            if len(res) == 0:
                res = ["<blank>"]
            return res

        inter_str = " " if self.keep_blank else ""
        ret = []
        for i, v_len in enumerate(pred_lens):
            beg = inverse[i, 0]
            end = inverse[i, v_len - 1]
            ret.append(inter_str.join(callback(indices[beg : end + 1])))
        return ret


class CTCGreedySearcher(CTCGreedyDecoder):
    def forward(self, preds, pred_lens):
        # preds: (B, C, 1, T) -> B C T -> B T C
        preds = preds.squeeze(2).permute(0, 2, 1)
        return super().forward(preds, pred_lens)
