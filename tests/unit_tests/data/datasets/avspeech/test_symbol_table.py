# Copyright (c) Horizon Robotics, All rights reserved.

import pathlib
import tempfile

from hat.data.datasets.avspeech.symbol_table import SymbolTable

FAKE_TABLE = """<sos/eos> 0
一 1
二 2
三 3
"""


def check_data(path):
    # 测试默认情况
    symbol_table = SymbolTable(path)
    assert len(symbol_table) == 4
    assert symbol_table["<sos/eos>"] == 0
    assert symbol_table["一"] == 1
    assert symbol_table["二"] == 2
    assert symbol_table["三"] == 3


def test_symbol_table():
    with tempfile.TemporaryDirectory() as dtmp:
        path = pathlib.Path(dtmp).joinpath("symbol.table")
        with open(path, "w", encoding="utf-8") as fw:
            fw.write(FAKE_TABLE)
        check_data(path)
