# Copyright (c) Your Company, All rights reserved.


import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import soundfile

from hat.data.datasets.avspeech.audio import BaseWaveformReader
from hat.data.datasets.avspeech.image import BaseImageReader

FPS = 30  # 目前是固定的, 后续可以改成返回ImageReader返回 FPS 的版本


def make_segment_check_data(
    snt_utt: str,
    cam_id: Optional[str],
    mic_id: Optional[str],
    seg_beg: float,
    seg_end: float,
    out_root: Path,
    image_reader: Optional[BaseImageReader],
    audio_reader: Optional[BaseWaveformReader],
    prefix: str = "",
    suffix: str = "",
    quiet: bool = False,
):
    """制作一段用于检查的数据.

    导出保存在训练数据库中的一段数据。
    可能是纯音频、纯视频和带音频的视频三种情况。
    最终生成的文件路径是:
    f"{out_root}/{prefix}{filename}{suffix}.{wav|mp4}"
    filename的格式是:
    f"{snt_utt}(_{cam_id})(_{mic_id})_{seg_beg}_{seg_end}"

    Args:
        snt_utt: 在数据库中保存的音频和视频共用的UTT字符串
        cam_id: 要导出的视频的cam_id, 当cam_id是None时,只导出音频.
        mic_id: 要导出的音频的mic_id, 当mic_id是None时,只导出视频.
        seg_beg: 片段的开始时间, 以s为单位的浮点数.
        seg_end: 片段的结束时间, 以s为单位的浮点数.
        out_root: 保存输出文件的地址.
        image_reader: 训练数据视频读取器.
        audio_reader: 训练数据音频读取器.
        prefix: 文件名的前缀.
        suffix: 文件名的后缀.
        quiet: 是否静默运行ffmpeg命令.

    Raises:
        Exception: 当cv2.VideoWriter写视频出错时会输出异常
    """
    # 临时文件下的数据将被删除
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        # 获取视频图片序列
        if cam_id is not None:
            video_utt = f"{snt_utt}_{cam_id}"
            images = image_reader.read_seg_images(video_utt, seg_beg, seg_end)
            # 生成静音视频
            sil_mp4_path = tmpdir.joinpath("sil.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            try:
                vw = cv2.VideoWriter(str(sil_mp4_path), fourcc, FPS, (96, 96))
                for image in images:
                    vw.write(image)
            except Exception as e:
                raise e
            finally:
                vw.release()
            # 如果 mic_id 是None的话, 就不需要生成带音频的视频
            if mic_id is None:
                out_path = out_root.joinpath(
                    f"{prefix}{video_utt}_{seg_beg: .3f}_{seg_end :.3f}{suffix}.mp4"  # noqa: E501
                )
                shutil.copy2(sil_mp4_path, out_path)
        # 获取音频wave
        if mic_id is not None:
            audio_utt = f"{snt_utt}_{mic_id}"
            waveform, framerate = audio_reader.read_seg_audio(
                audio_utt, seg_beg, seg_end
            )
            # 保存音频文件
            wav_path = tmpdir.joinpath("tmp.wav")
            soundfile.write(wav_path, waveform, framerate)
            if cam_id is None:
                out_path = out_root.joinpath(
                    f"{prefix}{audio_utt}_{seg_beg :.3f}_{seg_end :.3f}{suffix}.wav"  # noqa: E501
                )
                shutil.copy2(wav_path, out_path)
        # 音视频混合
        if cam_id is not None and mic_id is not None:
            out_video_path = out_root.joinpath(
                f"{prefix}{video_utt}_{cam_id}_{mic_id}"
                f"_{seg_beg :.3f}_{seg_end :.3f}{suffix}.mp4"
            )
            if quiet:
                cmd = (
                    f"ffmpeg -hide_banner -loglevel error -y -i {sil_mp4_path} -i {wav_path} "  # noqa: E501
                    "-c:v copy -c:a libmp3lame -map 0:v:0 -map 1:a:0 "
                    f"{out_video_path} </dev/null"
                )
            else:
                cmd = (
                    f"ffmpeg -y -i {sil_mp4_path} -i {wav_path} "
                    "-c:v copy -c:a libmp3lame -map 0:v:0 -map 1:a:0 "
                    f"{out_video_path} </dev/null"
                )
                print(cmd, file=sys.stderr, flush=True)
            subprocess.check_call(cmd, shell=True)
