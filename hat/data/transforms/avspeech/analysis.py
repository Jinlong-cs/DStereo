# mypy: allow-untyped-defs
"""AVSPEECH(多模语音)分析模块.

目前该模块下共有一个类:

SyntheticVideo  : 将图片序列和音频合成视频.
"""

import os
import shutil
import tempfile

import cv2
import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None
import torch

from hat.utils.package_helper import require_packages
from .utils import call_subproc_cmd


class SyntheticVideo(object):
    """SyntheticVideo.

    将图片序列和对应的音频合成为视频.

    Args:
        save_dir: 保存视频的目录.
        fps: 合成视频的帧率.
    """

    @require_packages("soundfile")
    def __init__(self, save_dir="./tmp", fps=25):
        self.dir = save_dir
        self.fps = fps
        if not os.path.exists(self.dir):
            os.makedirs(self.dir, exist_ok=True)

    def __call__(self, data):

        images = data["images"]
        if isinstance(images, torch.Tensor):
            if images.shape[1] == 3:
                images = images.permute(0, 2, 3, 1)  # n c h w -> n h w c
            images = [
                images[i].numpy().astype(np.uint8)
                for i in range(images.shape[0])
            ]
        uttid = data["key"]
        video_path = os.path.join(self.dir, uttid + ".mp4")
        self.write_video(images, video_path)
        waveform, samplerate = data["waveform"]
        audio_path = os.path.join(self.dir, uttid + ".wav")
        waveform = waveform.numpy()
        if data["channel_dim"] == 0:
            waveform = waveform.transpose()
        sf.write(audio_path, waveform, samplerate)
        self.merge_audio_to_video(video_path, audio_path)
        del data["waveform"]
        return data

    def write_video(self, images, video_path):
        for idx, image in enumerate(images):
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if idx == 0:
                height, width = image.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                vwriter = cv2.VideoWriter(
                    video_path, fourcc, self.fps, (width, height)
                )
            vwriter.write(image)
        vwriter.release()

    def merge_audio_to_video(self, video_path, audio_path):
        with tempfile.TemporaryDirectory() as dtmp:
            # merge to a tmp file
            tmp_vpath = os.path.join(dtmp, "temp.mp4")
            cmd = (
                f"ffmpeg -y -i {video_path} -i {audio_path} "
                "-c:v copy -c:a libmp3lame -map 0:v:0 -map 1:a:0 "
                f"{tmp_vpath} </dev/null"
            )
            call_subproc_cmd(cmd)
            # move back
            shutil.move(tmp_vpath, video_path)
