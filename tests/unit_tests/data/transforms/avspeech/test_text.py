import pathlib
import tempfile

import torch

from hat.data.datasets.avspeech.symbol_table import SymbolTable
from hat.data.transforms.avspeech.text import (
    LabelToTensor,
    MonophoneTokenize,
    Tokenize,
)

FAKE_TABLE = """<sos/eos> 0
我 1
爱 2
中 3
国 4
"""

FAKE_PHONE = """sil
b_t0
ai_t2
t_t0
ian_t1
"""


def check_data(symbol_path, phone_path):
    # 测试Tokenize
    data = {"label": "我爱中国"}
    symbol_table = SymbolTable(symbol_path)
    data = Tokenize(
        symbol_table=symbol_table,
        bpe_model_path=None,
        non_lang_syms=None,
        split_with_space=False,
    )(data)
    assert "text" in data
    assert "tokens" in data
    assert "label" in data
    assert "label_length" in data and data["label_length"] == 4

    # 测试LabelToTensor
    data = LabelToTensor()(data)
    assert isinstance(data["label"], torch.Tensor)

    # 测试MonophoneTokenize
    data = {"tokens": ["b_t0", "ai_t2", "t_t0", "ian_t1"]}
    data = MonophoneTokenize(phone_path)(data)
    assert "label" in data
    assert "tokens" in data
    assert "label_length" in data and data["label_length"] == 4


def test_text():
    with tempfile.TemporaryDirectory() as dtmp:
        symbol_path = pathlib.Path(dtmp).joinpath("symbol.table")
        with open(symbol_path, "w", encoding="utf-8") as fw:
            fw.write(FAKE_TABLE)

        phone_path = pathlib.Path(dtmp).joinpath("phone.txt")
        with open(phone_path, "w", encoding="utf-8") as fw:
            fw.write(FAKE_PHONE)

        check_data(symbol_path, phone_path)
