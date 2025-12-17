# Copyright (c) Horizon Robotics, All rights reserved.
"""音频是否保存在HDF5文件中的存在检查工具.

根据Info文件目录, 检查音频文件是否已经保存在 HDF5 中.
UTT 经过解析后, 会访问 Tendis 获取 BasicInfo 信息,
继而检查所有BasicInfo中提到的音频是否存在.
Info文件要求第一列是需要检查的UTT序列:

.. code::

    UTT 其他内容1 其他内容2 ...
    UTT 其他内容1 其他内容2 ...
    UTT 其他内容1 其他内容2 ...
"""


import argparse
import json

import h5py
from tqdm import tqdm

from hat.data.datasets.avspeech.basic_data_info import TendisBasicInfoReader
from hat.data.datasets.avspeech.info import UttParser
from hat.utils.filesystem import get_filesystem

TENDIS_KWRAGS = dict(  # noqa: C408
    host="aidi-kv-cluster-01.hogpu.cc",
    port=7617,
    db=0,
    socket_connect_timeout=5,
    password="pB9cA2eN1eD1lJ0",
    socket_keepalive=False,
)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    pa = parser.add_argument
    pa("--h5", type=str, help="保存音频的HDF5文件")
    pa("--info", type=str, help="待检查的INFO文件")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    utt_parser = UttParser()
    basic_info_reader = TendisBasicInfoReader(tendis_kwargs=TENDIS_KWRAGS)
    h5_fs = get_filesystem(args.h5)
    info_fs = get_filesystem(args.info)
    with h5_fs.open(args.h5) as fr:
        with info_fs.open(args.info, mode="r", encoding="utf-8") as info_fr:
            with h5py.File(fr, "r") as h5_fileno:
                for line in tqdm(info_fr, desc="Processing..."):
                    utt, *_ = line.strip().split()
                    matched = utt_parser.match(utt)
                    snt_utt = f"{matched['dataset']}_{matched['batch']}-{matched['snt']}"  # noqa: E501
                    basic_info = basic_info_reader.read_basic_data_info(
                        snt_utt
                    )
                    if basic_info is None:
                        print(f"{snt_utt} has none basic info")
                        continue
                    mics = basic_info.get_available_mic_ids()
                    if len(mics) == 0:
                        out = json.dumps(
                            basic_info.data_info, ensure_ascii=False
                        )
                        print(f"{snt_utt} has no mic\n{out}")
                        continue
                    for mic in mics:
                        try:
                            audio_utt = f"{snt_utt}_{mic}"
                            audio_data = h5_fileno["audio"][audio_utt]
                            attrs = audio_data.attrs
                        except Exception:
                            out = json.dumps(
                                basic_info.data_info, ensure_ascii=False
                            )
                            print(f"{audio_utt} not exist.\n{out}")
