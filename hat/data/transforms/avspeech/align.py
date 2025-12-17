# Copyright (c) Horizon Robotics, All rights reserved.
"""AVSPEECH(多模语音)音视频对齐.

目前该模块下共有三个类:

AlignInterface  : 音视频对齐基类, 检查字段是否存在.
ForceAlign      : 音视频对齐.
AVWindowAlign   : 窗内音视频对齐.

其中, ForceAlign 用于多模通用识别音视频对齐,
AVWindowAlign 用于多模命令词音视频对齐.
"""

import numpy as np
import torch

from hat.registry import OBJECT_REGISTRY


class AlignInterface(object):
    """音视频对齐基类.

    要求调用时传入的data包含 "audio" 和 "images" 键,
    要求子类实现transform接口.
    """

    def __call__(self, data):
        assert "audio" in data, f"{__class__.__name__} use ``audio`` in data"
        assert "images" in data, f"{__class__.__name__} use ``images`` in data"
        if data["audio"] is None or data["images"] is None:
            return data
        data = self.transform(data)
        return data

    def transform(self, data):
        """所有子类都应该实现该接口."""
        raise NotImplementedError


@OBJECT_REGISTRY.register
class ForceAlign(AlignInterface):
    """音视频对齐.

    将训练样本中的音频特征和视频特征对齐.

    Args:
        v2a_factor: 一帧视频对应多少帧音频. 例如: 在多模通用识别任务中, 视频25fps,
                    每帧40ms, 音频每帧10ms, 训练的时候取v2a_factor=4.
        audio_left_extend: 音频特征左边扩展的长度, 用真实数据代替因果卷积向左padding,
                           默认为0, 不做扩展.
    """

    def __init__(
        self,
        v2a_factor: int = 4,
        audio_left_extend: int = 0,
    ):
        self.v2a_factor = v2a_factor
        self.audio_left_extend = audio_left_extend

    def transform(self, data):
        audio = data["audio"]
        images = data["images"]
        v_len = len(images)
        a_len = audio.shape[0] - self.audio_left_extend
        legal_len = min(a_len // self.v2a_factor, v_len)
        a_legal_len = legal_len * self.v2a_factor + self.audio_left_extend
        images = images[-legal_len:]
        audio = audio[-a_legal_len:, :]
        data["audio"] = audio
        data["audio_lens"] = a_legal_len
        data["images"] = images
        data["images_lens"] = legal_len
        return data


@OBJECT_REGISTRY.register
class AVWindowAlign(AlignInterface):
    """窗内音视频对齐.

    将audio特征整理成(video_seq_len, video_window_size * v2afactor, 1, 3, 40)的尺寸;
    确保每个window内视频和音频的对齐.

    Args:
        video_window_size: 视频的窗长.
        v2a_factor: 一帧视频对应多少帧音频. 例如: 在多模命令词任务中, 视频30fps,
                    每帧33.33333ms, 音频每帧11ms, 训练的时候取v2a_factor=3.
        audio_pre_padding_mode: 数据段刚开始的时间音频往前padding的模式;
                                如果设置为zero: 补零.
                                如果设置为edge: 补第一帧.
    """

    def __init__(
        self,
        video_window_size: int = 10,
        v2a_factor: int = 3,
        audio_pre_padding_mode: str = "zero",
    ):

        self.v2a_factor = v2a_factor
        self.video_window_size = video_window_size
        self.audio_window_size = video_window_size * v2a_factor
        self.audio_pre_padding_mode = audio_pre_padding_mode

    def transform(self, data):
        audio = data["audio"]
        images = data["images"]
        audio_len = audio.shape[0]
        images_len = len(images)

        # chunk audio feature
        audio_idx_array = np.linspace(
            0, audio_len, num=images_len, endpoint=False
        )
        audio_idx_array = np.flipud(np.ceil(audio_len - 1 - audio_idx_array))
        audio_idx_array = np.reshape(audio_idx_array, (-1, 1))
        cached_array = np.flipud(np.arange(0, self.audio_window_size))
        audio_idx_array = audio_idx_array - cached_array

        audio_idx_array = audio_idx_array + 1
        if self.audio_pre_padding_mode == "zero":
            audio_idx_array = np.clip(audio_idx_array, 0, 1e9)
        elif self.audio_pre_padding_mode == "edge":
            audio_idx_array = np.clip(audio_idx_array, 1, 1e9)
        else:
            raise NotImplementedError

        audio_pad_value = torch.zeros(
            (1,) + audio.shape[1:], device=audio.device
        )
        audio = torch.cat((audio_pad_value, audio), dim=0)
        audio_idx_array = torch.tensor(audio_idx_array.astype(int))
        audio = audio[audio_idx_array]

        # reshape [seqlen, audio_window_size, 1, 3, 40] to
        # [seqlen, video_window_size, 120 * 3]
        audio_shape = audio.shape
        assert len(audio_shape) == 5
        assert audio_shape[2] == 1
        audio = audio.reshape(
            audio_shape[0],
            audio_shape[1] // 3,
            audio_shape[3] * audio_shape[4] * 3,
        )

        images_len_should_be = audio.shape[0]

        if images_len_should_be > images_len:
            images_pad_value = [
                torch.zeros_like(images[0], device=images[0].device)
                for _ in range(images_len_should_be - images_len)
            ]
            images = images.extend(images_pad_value)
        else:
            images = images[:images_len_should_be]

        data["audio"] = audio
        data["images"] = images
        data["audio_lens"] = audio.shape[0]
        data["images_lens"] = len(images)

        return data
