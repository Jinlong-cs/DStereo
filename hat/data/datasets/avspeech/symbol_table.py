# Copyright (c) Horizon Robotics, All rights reserved.
"""AVSPEECH(多模语音)标注符号(字典)读取模块.

目前该模块下共有一个类:

SymbolTable : 标注符号(字典)读取类.
"""

import pathlib
from typing import Any, Tuple, Union

from hat.registry import OBJECT_REGISTRY
from hat.utils.filesystem import get_filesystem


@OBJECT_REGISTRY.register
class SymbolTable(object):
    """SymbolTable.

    用于标注符号(字典)读取, 本质是一个dict.
    可以基于此类扩充symbol处理相关的功能.

    Args:
        path: 标注表的文件路径或pathlib.Path对象.
    """

    def __init__(self, path: Union[pathlib.Path, str]):
        self._path = path
        # 获取symbol列表 (这个dict的长度比较可控, 可以不开fork)
        fs = get_filesystem(self._path)
        with fs.open(self._path, "r", encoding="utf-8") as fr:

            def callback(line: str) -> Tuple[str, int]:
                lsp = line.strip().split()
                assert len(lsp) == 2, f"错误的symbol table文件: {line}"
                sym, cls = lsp
                return sym, int(cls)

            _iter = (callback(line) for line in fr)
            self._symbol_table = {sym: cls for sym, cls in _iter}

    def __getitem__(self, key: Any) -> int:
        return self._symbol_table[key]

    def __setitem__(self, key: Any, value: int):
        assert isinstance(value, int), "class id should be int."
        for key_, value_ in self._symbol_table.items():
            assert (
                key == key_ or value != value_
            ), f"{value} 已经被指定给 {key_}, 不能在指定给 {key}"
        self._symbol_table[key] = value

    def __len__(self) -> int:
        return len(self._symbol_table)

    def __contains__(self, item) -> bool:
        return item in self._symbol_table

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__ + ":"
        repr_str += f"path=.../{pathlib.Path(self._path).name}"
        keys = list(self._symbol_table.keys())
        if len(keys) > 10:
            table_head_str = ", ".join(
                f"{key}:{self._symbol_table[key]}" for key in keys[:5]
            )
            table_tail_str = ", ".join(
                f"{key}:{self._symbol_table[key]}" for key in keys[-5:]
            )
            repr_str += f", {{{table_head_str}, ...{table_tail_str}}}"
        else:
            table_str = ", ".join(
                f"{key}:{self._symbol_table[key]}" for key in keys
            )
            repr_str += f", {{{table_str}}}"
        repr_str += f", lens={len(self)}"
        return repr_str
