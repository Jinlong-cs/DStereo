# Copyright (c) Horizon Robotics, All rights reserved.
"""AVSPEECH(多模语音)处理文本数据转换.

目前该模块下共有四个类、一个函数:

TextInterface        : 文本数据处理基类, 检查字段是否存在.
Tokenize             : 将文本拆解为字符或 BPE.
LabelToTensor        : 将 label 转成 torch.Tensor.
MonophoneTokenize    : 将文本转成单音素.

tokenize_by_bpe_model: 将文本拆解为 BPE.
"""

import pathlib
import re
from typing import Mapping, Optional, Union

import torch

from hat.utils.filesystem import get_filesystem

try:
    import sentencepiece as spm
except ImportError:
    spm = None

from hat.registry import OBJECT_REGISTRY


def tokenize_by_bpe_model(sp, txt):
    """将本文拆解为BPE.

    Args:
        sp: BPE 模型.
        txt: 文本.
    """

    tokens = []
    pattern = re.compile(r"([\u4e00-\u9fff])")
    chars = pattern.split(txt.upper())
    mix_chars = [w for w in chars if len(w.strip()) > 0]
    for ch_or_w in mix_chars:
        if pattern.fullmatch(ch_or_w) is not None:
            tokens.append(ch_or_w)
        else:
            for p in sp.encode_as_pieces(ch_or_w):
                tokens.append(p)

    return tokens


class TextInterface(object):
    """文本处理基类.

    要求调用时传入的data包含 "label" 键,
    要求子类实现transform接口.
    """

    def __call__(self, data):
        assert "label" in data, f"{__class__.__name__} use ``label`` in data"
        data = self.transform(data)
        return data

    def transform(self, data):
        """所有子类都应该实现该接口."""
        raise NotImplementedError


@OBJECT_REGISTRY.register
class Tokenize(TextInterface):
    """将文本拆解为字符或BPE.

    Args:
        symbol_table: 字典.
        bpe_model_path: BPE 模型路径.
        non_lang_syms: 特殊字符.
        split_with_space: 如果设置为True, 以空格分割.
                          如果设置为False, 不以空格分割.
                          默认为False.
    """

    def __init__(
        self,
        symbol_table: Mapping[str, int],
        bpe_model_path: Optional[Union[pathlib.Path, str]] = None,
        non_lang_syms: Mapping[str, int] = None,
        split_with_space: bool = False,
    ):
        self.symbol_table = symbol_table
        self.bpe_model_path = (
            bpe_model_path if bpe_model_path is None else str(bpe_model_path)
        )
        if non_lang_syms is not None:
            self.non_lang_syms_pattern = re.compile(
                r"(\[[^\[\]]+\]|<[^<>]+>|{[^{}]+})"
            )
            self.non_lang_syms = non_lang_syms
        else:
            self.non_lang_syms = {}
            self.non_lang_syms_pattern = None

        self.split_with_space = split_with_space
        if self.bpe_model_path is not None:
            sp = spm.SentencePieceProcessor()
            sp.load(self.bpe_model_path)
            self.bpe_model = sp
        else:
            self.bpe_model = None

    def transform(
        self, data: Mapping[str, Optional[torch.Tensor]]
    ) -> Mapping[str, Optional[torch.Tensor]]:
        text = data["label"].strip()
        if self.non_lang_syms_pattern is not None:
            parts = self.non_lang_syms_pattern.split(text.upper())
            parts = [w for w in parts if len(w.strip()) > 0]
        else:
            parts = [text]

        label = []
        tokens = []
        for part in parts:
            if part in self.non_lang_syms:
                tokens.append(part)
            else:
                if self.bpe_model is not None:
                    tokens.extend(tokenize_by_bpe_model(self.bpe_model, part))
                else:
                    if self.split_with_space:
                        part = part.split(" ")
                    for ch in part:
                        if ch == " ":
                            ch = "▁"
                        tokens.append(ch)
        for ch in tokens:
            if ch in self.symbol_table:
                label.append(self.symbol_table[ch])
            elif "<unk>" in self.symbol_table:
                label.append(self.symbol_table["<unk>"])
        data["text"] = text
        data["tokens"] = tokens
        data["label"] = label
        data["label_length"] = len(label)
        return data


@OBJECT_REGISTRY.register
class LabelToTensor(TextInterface):
    """将label转成torch.Tensor."""

    def transform(self, data):
        data["label"] = torch.tensor(data["label"], dtype=torch.int64)
        return data


@OBJECT_REGISTRY.register
class MonophoneTokenize(object):
    """将音素转成ID.

    Args:
        phone_file: 单音素文件路径.
    """

    def __init__(self, phone_file: str):

        self._phone_index_file = phone_file
        fs = get_filesystem(phone_file)
        self.phone2index = {}
        with fs.open(phone_file, "r", encoding="utf-8") as fr:
            for index, line in enumerate(fr.readlines()):
                lsp = line.split()
                assert len(lsp) == 1, f"Illegal line: {line}"
                phone = lsp[0]
                self.phone2index[phone] = index

    def __call__(self, data: dict) -> dict:
        assert (
            "tokens" in data
        ), f"{__class__.__name__} use ``tokens`` in data."

        phones = data["tokens"]
        indexes = [self.phone2index[p] for p in phones]
        data["label"] = indexes
        data["tokens"] = phones
        data["label_length"] = len(indexes)

        return data
