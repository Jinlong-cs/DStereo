# mypy: allow-untyped-defs
"""AVSPEECH(多模语音)模态增强模块.

目前该模块下共有一个类:

RandomZero  : 对指定关键字数据随机置零.
"""

import random

import torch


class RandomZero(object):
    """RandomZero.

    以一定概率对data中的指定关键字内容置零.

    Args:
        keys: 关键字列表.
        p: 应用变换的概率值.
    """

    def __init__(
        self,
        keys: list,
        p: float = 0.0,
    ):
        self.keys = keys
        self.p = p

    def __call__(self, data):
        if random.random() > self.p:
            return data
        for key in self.keys:
            assert key in data, f"{__class__.__name__} use ``{key}`` in data"
            x = data[key]
            if x is None:
                continue
            assert isinstance(x, torch.Tensor)
            y = x.clone().detach()
            y = torch.zeros_like(y)
            data[key] = y
        return data
