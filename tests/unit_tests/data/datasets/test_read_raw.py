# Copyright (c) Horizon Robotics. All rights reserved.

import os

import pytest

from hat.data.datasets.read_raw import read_raw
from tests import BasicAlgorithm_BUCKET_EXISTS, BasicAlgorithm_BUCKET_PATH

input_list = [
    [
        "frm/debug_dump_out_tpg_0_1.frm",
        {
            "height": 2160,
            "width": 3840,
            "bit_nums_upper": 20,
            "mipi": 0,
            "channels": 1,
            "pre_offset": 64,
        },
    ],
    [
        "frm/debug_dump_out_ccm_1.frm",
        {
            "height": 2160,
            "width": 3840,
            "bit_nums_upper": 12,
            "mipi": 0,
            "channels": 3,
            "pre_offset": 64,
        },
    ],
    [
        "mipi_raw/ADAS_20220130-092315_512_3_00014356_mipi.raw",
        {
            "height": 2160,
            "width": 3840,
            "bit_nums_upper": 12,
            "mipi": 1,
            "channels": 1,
            "pre_offset": 0,
        },
    ],
]


@pytest.mark.skipif(
    not BasicAlgorithm_BUCKET_EXISTS, reason="path does not exist"
)
def test_read_raw():
    for rel_path, image_info in input_list:
        data_path = os.path.join(
            BasicAlgorithm_BUCKET_PATH, "Neural_ISP/unit_test_data", rel_path
        )
        out_raw = read_raw(data_path, image_info)
        assert out_raw.shape == (
            image_info["height"],
            image_info["width"],
            image_info["channels"],
        )
        print(data_path)
