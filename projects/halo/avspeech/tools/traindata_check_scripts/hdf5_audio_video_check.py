# Copyright (c) Horizon Robotics, All rights reserved.
"""HDF5保存音频方式的音视频数据检查工具.

使用方式1: python3 hdf5_audio_video_check.py --snt_utt SNT_UTT
            --cam_id CAM_ID --mic_id MIC_ID
            --seg_beg SEG_BEG --seg_end SEG_END
            --out_root OUT_ROOT
            [--h5 H5] [--sil_mp4] [--only_wav]

根据snt_utt对应的cam_id和mic_id对应的从seg_beg到seg_end之间的视频.
用于某一小段数据是否存在问题.在不指定 --sil_mp4 的情况下。
需要指定``--h5``的 HDF5路径.


使用方式2: python3 hdf5_audio_video_check.py --info INFO --out_root OUT_ROOT
            [--with_index] [--h5 H5] [--sil_mp4] [--only_wav]

根据Info文件列表, 批量生成音视频段对应的视频
UTT 经过解析后, 会访问 Tendis 获取 BasicInfo 信息,
继而检查所有BasicInfo中提到的音频是否存在.
Info文件要求第一列是需要检查的UTT序列,
最后两列是INFO段的开始和结束:

.. code::

    UTT 其他内容1 其他内容2... SEG_BEG SEG_END
    UTT 其他内容1 其他内容2... SEG_BEG SEG_END
    UTT 其他内容1 其他内容2... SEG_BEG SEG_END

Info文件也可以只有一列, 此时会将整段音视频导出:

.. code::
    UTT
    UTT
    UTT

使用示例:

python3 hdf5_audio_video_check.py --snt_utt FYZ_mmasr_batch01-F0011 \\
    --cam_id cam13 --mic_id cam33 --seg_beg 0.0 --seg_end 10.0 \\
    --h5 "/horizon-bucket/J2MM/speech/hdf5/audio.hdf5"
    --out_root "./"

python3 hdf5_audio_video_check.py --snt_utt FYZ_mmasr_batch01-F0011 \\
    --cam_id cam13 --seg_beg 0.0 --seg_end 10.0 \\
    --out_root "./" --sil_mp4

python3 hdf5_audio_video_check.py --snt_utt FYZ_mmasr_batch01-F0011 \\
    --mic_id cam33 --seg_beg 0.0 --seg_end 10.0 \\
    --h5 "/horizon-bucket/J2MM/speech/hdf5/audio.hdf5" \\
    --out_root "./" --only_wav

mkdir -p mp4/;
echo 'FYZ_mmasr_batch01-F0011' > info.txt;
python3 hdf5_audio_video_check.py --info info.txt \\
    --h5 "/horizon-bucket/J2MM/speech/hdf5/audio.hdf5" \\
    --out_root "mp4/"

mkdir -p mp4;
echo 'FYZ_mmasr_batch01-F0011_0 播放两只老虎 6.649 8.215' > info.txt;
echo 'FYZ_mmasr_batch01-F0011_1 我要起床 8.834 10.352' >>info.txt;
echo 'FYZ_mmasr_batch01-F0011_2 哈尔滨银行浑南支行 11.081 13.121' >>info.txt
echo 'FYZ_mmasr_batch01-F0011_3 播放音乐 13.741 15.082' >> info.txt;
python3 hdf5_audio_video_check.py --info info.txt \\
    --h5 "/horizon-bucket/J2MM/speech/hdf5/audio.hdf5" \\
    --out_root "mp4/"
"""
import argparse
import itertools
import sys
from pathlib import Path

from tqdm import tqdm

from hat.data.datasets.avspeech.audio import HDF5WaveformReader
from hat.data.datasets.avspeech.basic_data_info import TendisBasicInfoReader
from hat.data.datasets.avspeech.image import TendisVideoImageReader
from hat.data.datasets.avspeech.info import UttParser
from hat.utils.filesystem import get_filesystem
from projects.halo.avspeech.utils.data_check import (
    FPS,
    make_segment_check_data,
)

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
    pa("--out_root", type=str, required=True, help="输出视频的根目录")
    pa("--snt_utt", type=str, default=None, help="想要检查的片段的snt_utt")
    pa("--cam_id", type=str, default=None, help="想要检查的片段的cam_id")
    pa("--mic_id", type=str, default=None, help="想要检查的片段的mic_id")
    pa("--seg_beg", type=float, default=None, help="想要检查的片段的起始时间")
    pa("--seg_end", type=float, default=None, help="想要检查的片段的结束时间")
    pa(
        "--info",
        type=str,
        default=None,
        help="待检查的INFO文件列表,当指定info文件时,"
        "snt_utt, cam_id, mic_id,seg_beg,seg_end都将无效",
    )
    pa(
        "--with_index",
        action="store_true",
        help="当指定--info时起作用, 通过指定该项, 给保存的文件开头指定编号",
    )
    pa(
        "--h5",
        type=str,
        default=None,
        help="保存音频的HDF5文件, 指定--sil_mp4时不需要指定",
    )

    pa(
        "--sil_mp4",
        action="store_true",
        help="只输出不包含音频的视频",
    )
    pa("--only_wav", action="store_true", help="只输出音频wav")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    assert not (
        args.sil_mp4 and args.only_wav
    ), "sil_mp4 和 only_wav 不能同时设置为True"
    # 准备一些辅助类
    utt_parser = UttParser()
    basic_info_reader = TendisBasicInfoReader(tendis_kwargs=TENDIS_KWRAGS)
    if args.only_wav:
        image_reader = None
    else:
        image_reader = TendisVideoImageReader(
            tendis_kwargs=TENDIS_KWRAGS,
            fps=FPS,
            origin_fps=30,
            missing_image_complete_mode="zero",
        )
    if args.sil_mp4:
        audio_reader = None
    else:
        audio_reader = HDF5WaveformReader(
            args.h5, ret_dtype="float32", channel_first=False
        )
    # 根据 info 文件批量导出
    if args.info is not None:
        info_fs = get_filesystem(args.info)
        with info_fs.open(args.info, mode="r", encoding="utf-8") as info_fr:
            for idx, line in enumerate(tqdm(info_fr, desc="Processing...")):
                # 解析 utt 和 seg_beg, seg_end
                lsp = line.strip().split()
                if len(lsp) == 1:
                    utt = lsp[0]
                    seg_beg, seg_end = None, None
                else:
                    utt, *_, seg_beg, seg_end = lsp
                    seg_beg, seg_end = float(seg_beg), float(seg_end)
                # 根据 matched basic info 信息
                matched = utt_parser.match(utt)
                snt_utt = f"{matched['dataset']}_{matched['batch']}-{matched['snt']}"  # noqa: E501
                basic_info = basic_info_reader.read_basic_data_info(snt_utt)
                if basic_info is None:
                    print(f"{snt_utt} has none basic info", file=sys.stderr)
                    continue
                if args.sil_mp4:
                    mics = [
                        None,
                    ]
                else:
                    mics = basic_info.get_available_mic_ids()
                if args.only_wav:
                    cams = [
                        None,
                    ]
                else:
                    cams = basic_info.get_available_cam_ids()
                if seg_beg is None and seg_end is None:
                    seg_beg = 0.0
                    seg_end = basic_info.get_audio_length()
                # 根据是否编号构造前缀
                prefix = f"{idx}" if args.with_index else ""
                for mic, cam in itertools.product(mics, cams):
                    make_segment_check_data(
                        snt_utt=snt_utt,
                        cam_id=cam,
                        mic_id=mic,
                        seg_beg=seg_beg,
                        seg_end=seg_end,
                        out_root=Path(args.out_root),
                        image_reader=image_reader,
                        audio_reader=audio_reader,
                        prefix=prefix,
                    )
    # 指定 snt_utt 等具体信息检查某一条数据
    else:
        make_segment_check_data(
            snt_utt=args.snt_utt,
            cam_id=args.cam_id,
            mic_id=args.mic_id,
            seg_beg=args.seg_beg,
            seg_end=args.seg_end,
            out_root=Path(args.out_root),
            image_reader=image_reader,
            audio_reader=audio_reader,
        )
