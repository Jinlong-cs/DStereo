# Copyright (c) Horizon Robotics, All rights reserved.

import pathlib
import tempfile

from hat.data.datasets.avspeech.info import MMASRInfoList, MMASRJsonInfoList
from tests import HAT_BUCKET_PATH

FAKE_MMASR_INFO = """utt1 你好 10.0 11.0
utt2 H A T 12.0 14.0
utt3 你好 14.0 14.05
utt4 H A T 15.01 15.02
utt5 你好 16.0 22.1
utt6 H A T 23.0 30.0
"""


def check_data(info_path):
    # 测试默认情况
    info_list = MMASRInfoList(info_path)
    assert len(info_list) == 2
    assert info_list[0]["seg_utt"] == "utt1"
    assert info_list[0]["seg_beg"] == 10.0
    assert info_list[0]["seg_end"] == 11.0
    assert info_list[0]["text"] == "你好"
    assert info_list[1]["seg_utt"] == "utt2"
    assert info_list[1]["seg_beg"] == 12.0
    assert info_list[1]["seg_end"] == 14.0
    assert info_list[1]["text"] == "HAT"
    # 测试左右限制的情况
    info_list = MMASRInfoList(info_path, left_limit=0.05, right_limit=6.5)
    assert len(info_list) == 4


def test_mmasr_info_list():
    # 测试MMASRInfoList
    with tempfile.TemporaryDirectory() as dtmp:
        info_path = pathlib.Path(dtmp).joinpath("info.txt")
        with open(info_path, "w", encoding="utf-8") as fw:
            fw.write(FAKE_MMASR_INFO)
        check_data(info_path)


def test_mmasr_json_info_list():
    json_path = f"{HAT_BUCKET_PATH}/unit_test_data/avspeech/data/datasets/mmasr_json_info.json"  # noqa: E501
    # 测试默认情况
    info_list = MMASRJsonInfoList(json_path)
    assert len(info_list) == 10

    # 测试左右限制情况
    info_list = MMASRJsonInfoList(json_path, left_limit=1.5, right_limit=6.5)
    assert len(info_list) == 6
