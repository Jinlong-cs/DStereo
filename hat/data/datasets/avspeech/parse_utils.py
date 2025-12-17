# Copyright (c) Horizon Robotics, All rights reserved.
"""AVSPEECH(多模语音)多模命令词依赖文件读取模块.

目前该模块下共有三个函数:

parse_video_info    : info文件读取接口.
parse_monophone     : 样本utt和单音素序列对应文件读取接口.
parse_id_map        : 样本utt和特征rec时间戳对应文件读取接口.
"""

import logging

from tqdm import tqdm

from hat.utils.filesystem import get_filesystem


def parse_video_info(path: str):
    """parse_video_info.

    Info文件读取接口, 返回结果字典.

    Args:
        path: info文件路径.
    """

    fs = get_filesystem(path)
    ret = {}
    with fs.open(path, "r", encoding="utf-8") as fr:
        for line in tqdm(fr.readlines(), ncols=120, desc="Viedo Info Load..."):
            utt, *text, beg, end = line.strip().split()
            info = {
                "seg_utt": utt,
                "seg_beg": float(beg),
                "seg_end": float(end),
                "text": "".join(text),
            }
            if info["seg_end"] - info["seg_beg"] < 0.1:  # type: ignore
                logging.warning(f"ErrorInfo: {line}")
                continue
            ret[utt] = info
    return ret


def parse_monophone(path: str):
    """parse_monophone.

    样本utt和单音素序列对应文件读取接口, 返回结果字典.

    Args:
        path: 样本utt和单音素序列对应文件路径.
    """

    fs = get_filesystem(path)
    ret = {}
    with fs.open(path, "r", encoding="utf-8") as fr:
        for line in tqdm(fr.readlines(), ncols=120, desc="Monophone Load..."):
            utt, *monophones = line.strip().split()
            info = {
                "seg_utt": utt,
                "monophone": monophones,
            }
            ret[utt] = info
    return ret


def parse_id_map(path: str):
    """parse_id_map.

    样本utt和特征rec时间戳对应文件读取接口, 返回结果字典.

    Args:
        path: 样本utt和特征rec时间戳对应文件路径.
    """

    fs = get_filesystem(path)
    ret = {}
    with fs.open(path, "r", encoding="utf-8") as fr:
        for line in tqdm(fr.readlines(), ncols=120, desc="ID Map Load..."):
            line = line.strip()
            if not line:
                continue
            rec_idx, utt = line.split()
            if utt not in ret:
                ret[utt] = []
            ret[utt].append(rec_idx)
    return ret
