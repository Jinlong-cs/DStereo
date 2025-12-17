# Copyright (c) Horizon Robotics, All rights reserved.
# type: ignore
"""AVSPEECH(多模语音)处理音频数据转换.

目前该模块下共有十个类:

WaveformInterface   : 音频数据处理基类, 检查字段是否存在以及数据是否为空.
AudioInterface      : 音频特征处理基类, 检查字段是否存在以及数据是否为空.
WaveformToTensor    : 将音频数据转为torch.Tensor格式.
ReSample            : 将音频数据转为目标采样率.
ChannelSelect       : 对音频数据进行通道选择, 选择一个通道.
RandomSpeedPerturb  : 对音频数据应用随机速度扰动增强.
FBank               : 提取音频的fbank特征.
MFCC                : 提取音频的mfcc特征.
SpecAug             : 对音频特征在时频维度做Mask和Warp增强.
SpecSub             : 对音频特征在时频维度做替换增强.

其中 WaveformToTensor 将音频数据的array转成torch.Tensor
ReSample, ChannelSelect, RandomSpeedPerturb 都是对音频数据做操作.
FBank, MFCC 调用torchaudio的接口提音频特征.
SpecAug, SpecSub 都是对音频特征做操作, 在 FBank 和 MFCC 之后调用.
"""

import logging
import random
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image

from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages

try:
    import py_audio_feature as paf
except ImportError:
    paf = None

try:
    import torchaudio
    from torchaudio.compliance import kaldi
except ImportError:
    torchaudio = None


@OBJECT_REGISTRY.register
class WaveformInterface(object):
    """音频数据处理基类.

    要求调用时传入的data包含 "waveform" 键,
    要求子类实现transform接口.
    """

    def __call__(self, data):
        assert (
            "waveform" in data
        ), f"{__class__.__name__} use ``waveform`` in data"
        waveform, _ = data["waveform"]
        if waveform is None:
            data["audio"] = None
            data["audio_lens"] = None
            return data
        data = self.transform(data)
        return data

    def transform(self, data):
        """所有子类都应该实现该接口."""
        raise NotImplementedError


@OBJECT_REGISTRY.register
class AudioInterface(object):
    """音频特征处理基类.

    要求调用时传入的data包含 "audio" 键,
    要求子类实现transform接口.
    """

    def __call__(self, data):
        assert "audio" in data, f"{__class__.__name__} use ``audio`` in data"
        x = data["audio"]
        if x is None:
            return data
        assert isinstance(x, torch.Tensor)
        data = self.transform(data)
        return data

    def transform(self, data):
        """所有子类都应该实现该接口."""
        raise NotImplementedError


@OBJECT_REGISTRY.register
class WaveformToTensor(WaveformInterface):
    """将音频数据转成torch.Tensor."""

    def transform(self, data):
        waveform, samplerate = data["waveform"]
        waveform = torch.from_numpy(waveform).float()
        data["waveform"] = (waveform, samplerate)
        return data


@OBJECT_REGISTRY.register
class ReSample(WaveformInterface):
    """重采样.

    将音频采样率转为目标采样率.

    Args:
        new_freq: 目标采样率.
    """

    @require_packages("torchaudio")
    def __init__(self, new_freq: float = 16000):
        self.new_freq = new_freq

    def transform(self, data):
        waveform, samplerate = data["waveform"]
        if samplerate == self.new_freq:
            return data
        assert len(waveform.shape) == 2
        if data["channel_dim"] != 0:
            waveform = waveform.transpose(1, 0)
        waveform = torchaudio.transforms.Resample(
            orig_freq=samplerate, new_freq=self.new_freq
        )(waveform)
        if data["channel_dim"] != 0:
            waveform = waveform.transpose(1, 0)
        data["waveform"] = (waveform, self.new_freq)
        return data

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__ + ":"
        repr_str += f"new_freq={self.new_freq}, "
        return repr_str


@OBJECT_REGISTRY.register
class ChannelSelect(WaveformInterface):
    """通道选择.

    对音频数据进行通道选择, 选出一个有效通道.

    Args:
        random: 如果设置为True, 随机选择通道.
                如果设置为False, 选择一个固定通道.
                默认为False.
        channel: 如果random设置为False, 选择channel通道.
    """

    def __init__(self, random: bool = False, channel: int = 0):
        self.channel = channel
        self.random = random

    def transform(self, data):
        waveform, sample_rate = data["waveform"]
        if self.random:
            channels = waveform.shape[data["channel_dim"]]
            channel = [random.randint(0, channels - 1)]
        else:
            channel = [self.channel]
        if data["channel_dim"] == 0:
            waveform = waveform[channel, :]
        else:
            waveform = waveform[:, channel]

        data["waveform"] = (waveform, sample_rate)
        for key in ["role", "source_audio"]:
            if key in data:
                del data[key]
        return data


@OBJECT_REGISTRY.register
class RandomSpeedPerturb(WaveformInterface):
    """随机速度扰动.

    作为一种音频数据增强的方式, 对音频数据应用随机速度扰动.

    Args:
        speeds: 可选的速度序列.
        channel_first: 如果设置为True, 音频数据的第一维是通道.
                       如果设置为False, 音频数据的第二维是通道.
                       默认为True.
    """

    @require_packages("torchaudio")
    def __init__(
        self,
        speeds: Optional[Sequence[float]] = None,
        channel_first: bool = True,
    ):
        self.speeds = speeds
        if self.speeds is None:
            self.speeds = [0.9, 1.0, 1.1]
        self.channel_first = channel_first

    def transform(self, data):
        speed = random.choice(self.speeds)
        if speed == 1.0:
            return data
        waveform, samplerate = data["waveform"]
        waveform, samplerate = torchaudio.sox_effects.apply_effects_tensor(
            waveform,
            samplerate,
            [["speed", str(speed)], ["rate", str(samplerate)]],
            channels_first=self.channel_first,
        )
        data["waveform"] = (waveform, samplerate)
        return data

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__ + ":"
        repr_str += f"speeds={self.speeds}, "
        repr_str += f"channel_first={self.channel_first}"
        return repr_str


@OBJECT_REGISTRY.register
class FBank(WaveformInterface):
    """提取fbank特征.

    提取音频数据的fbank特征.

    Args:
        num_mel_bins: 梅尔滤波器的个数.
        frame_length: 以毫秒为单位的帧长.
        frame_shift: 以毫秒为单位的帧移.
        energy_floor: 频谱图计算中的能量下限.
        dither: 抖动常数.
        window_type: 窗类型, 默认为"povey".
    """

    @require_packages("torchaudio")
    def __init__(
        self,
        num_mel_bins: int = 40,
        frame_length: float = 25.0,
        frame_shift: float = 10.0,
        energy_floor: float = 0.0,
        dither: float = 0.0,
        window_type: str = "povey",
    ):
        self.kaldi_kwargs = {
            "num_mel_bins": num_mel_bins,
            "frame_length": frame_length,
            "frame_shift": frame_shift,
            "energy_floor": energy_floor,
            "dither": dither,
            "window_type": window_type,
        }

    def transform(self, data):
        waveform, samplerate = data["waveform"]
        if data["channel_dim"] != 0:
            waveform = waveform.transpose(1, 0)
        waveform = waveform * (1 << 15)
        feat = kaldi.fbank(
            waveform,
            sample_frequency=samplerate,
            **self.kaldi_kwargs,
        )
        data["audio"] = feat
        data["audio_lens"] = feat.size(0)
        del data["waveform"]
        logging.debug(f"Remove 'waveform' during {__class__.__name__}")
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ":"
        kw_str = ", ".join(f"{k}={v}" for k, v in self.kaldi_kwargs.items())
        repr_str += kw_str
        return repr_str


@OBJECT_REGISTRY.register
class PAFFBank(WaveformInterface):
    """使用 py_audio_feature 包提取fbank特征.

    Args:
        conf_path: py_audio_feature 要求的配置文件的路径.
                   如果不设置, 默认的配置文件如下:

    .. code::plain

        #原始特征维度, 不包含delta, 默认80
        --num_bins=80
        #帧长, ms
        --frame_length=25
        #帧移, ms
        --frame_shift=10
        #差分特征的阶数, 默认0不做差分特征
        --delta=0
        #窗类型,目前仅支持povey和hamming
        --window_type=povey
        #fft类型, 默认使用春棋优化的版本, hesr_x86仅为了适配hesr x86版本
        --fft_type=""
        #是否对音频数据做减均值处理, 默认true
        --remove_dc_offset=true
        #计算mel特征系数时的最低频率. 对应 kaldi.fbank 的 low_freq.
        --coeff_low_frequency=20
    """

    @require_packages("py_audio_feature")
    def __init__(self, conf_path: Optional[str] = None):
        self.conf_path = conf_path
        self.fbank = paf.FBank(conf_path)

    def transform(self, data):
        waveform, samplerate = data["waveform"]
        if data["channel_dim"] != 0:
            waveform = waveform.transpose(1, 0)
        waveform = waveform * (1 << 15)
        waveform: torch.Tensor
        feat = self.fbank(waveform[-1].tolist())
        data["audio"] = torch.Tensor(feat)
        data["audio_lens"] = feat.size(0)
        del data["waveform"]
        logging.debug(f"Remove 'waveform' during {__class__.__name__}")
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__
        if self.conf_path is not None:
            repr_str += ":" + f"<{self.conf_path}>"
        return repr_str


@OBJECT_REGISTRY.register
class MFCC(WaveformInterface):
    """提取mfcc特征.

    提取音频数据的mfcc特征.

    Args:
        num_mel_bins: 梅尔滤波器的个数.
        frame_length: 以毫秒为单位的帧长.
        frame_shift: 以毫秒为单位的帧移.
        energy_floor: 频谱图计算中的能量下限.
        dither: 抖动常数.
        high_freq: 梅尔滤波器的高截止频率.
        low_freq: 梅尔滤波器的低截止频率.
        num_ceps: mfcc计算中的倒谱数.
        window_type: 窗类型, 默认为"povey".
    """

    @require_packages("torchaudio")
    def __init__(
        self,
        num_mel_bins: int = 40,
        frame_length: float = 25.0,
        frame_shift: float = 10.0,
        energy_floor: float = 1.0,
        dither: float = 0.0,
        high_freq: float = 0.0,
        low_freq: float = 20.0,
        num_ceps: int = 40,
        window_type: str = "povey",
    ):
        self.kaldi_kwargs = {
            "num_mel_bins": num_mel_bins,
            "frame_length": frame_length,
            "frame_shift": frame_shift,
            "energy_floor": energy_floor,
            "dither": dither,
            "high_freq": high_freq,
            "low_freq": low_freq,
            "num_ceps": num_ceps,
            "window_type": window_type,
        }

    def transform(self, data):
        waveform, samplerate = data["waveform"]
        if data["channel_dim"] != 0:
            waveform = waveform.transpose(1, 0)
        waveform = waveform * (1 << 15)
        feat = kaldi.mfcc(
            waveform, sample_frequency=samplerate, **self.kaldi_kwargs
        )
        data["audio"] = feat
        data["audio_lens"] = feat.size(0)
        del data["waveform"]
        logging.debug(f"Remove 'waveform' during {__class__.__name__}")
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ":"
        kw_str = ", ".join(f"{k}={v}" for k, v in self.kaldi_kwargs.items())
        repr_str += kw_str
        return repr_str


@OBJECT_REGISTRY.register
class SpecAug(AudioInterface):
    """时频谱增强.

    以一定概率对音频的时频谱做数据增强.

    Args:
        num_t_mask: 时间维度的mask次数.
        num_f_mask: 频域维度的mask次数.
        max_t: 时间维度mask的最大宽度.
        max_f: 频率维度mask的最大宽度.
        max_w: 时间维度warp的最大宽度.
        warp_for_time: 如果设置为True, 在时间维度做warp.
                       如果设置为False, 在时间维度不做warp.
                       默认为False.
        p: 应用变换的概率值.
    """

    def __init__(
        self,
        num_t_mask: int = 2,
        num_f_mask: int = 2,
        max_t: int = 50,
        max_f: int = 10,
        max_w: int = 80,
        warp_for_time: bool = False,
        p: float = 0.5,
    ):
        self.num_t_mask = num_t_mask
        self.num_f_mask = num_f_mask
        self.max_t = max_t
        self.max_f = max_f
        self.max_w = max_w
        self.warp_for_time = warp_for_time
        self.p = p

    def transform(self, data):
        if random.random() > self.p:
            return data
        x = data["audio"]
        y = x.clone().detach()
        max_frames = y.size(0)
        max_freq = y.size(1)
        # time warp
        if self.warp_for_time and max_frames > self.max_w * 2:
            center = random.randrange(self.max_w, max_frames - self.max_w)
            warped = (
                random.randrange(center - self.max_w, center + self.max_w) + 1
            )

            left = Image.fromarray(y[:center].numpy()).resize(
                (max_freq, warped), Image.BICUBIC
            )
            right = Image.fromarray(y[center:].numpy()).resize(
                (max_freq, max_frames - warped), Image.BICUBIC
            )
            y = torch.from_numpy(np.concatenate((left, right), 0)).to(x.device)
        # time mask
        for _ in range(self.num_t_mask):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, self.max_t)
            end = min(max_frames, start + length)
            y[start:end, :] = 0
        # freq mask
        for _ in range(self.num_f_mask):
            start = random.randint(0, max_freq - 1)
            length = random.randint(1, self.max_f)
            end = min(max_freq, start + length)
            y[:, start:end] = 0
        data["audio"] = y
        return data

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__ + ":"
        repr_str += f"num_t_mask={self.num_t_mask}, "
        repr_str += f"num_f_mask={self.num_f_mask}, "
        repr_str += f"max_t={self.max_t}, "
        repr_str += f"max_f={self.max_f}, "
        repr_str += f"max_w={self.max_w}, "
        repr_str += f"warp_for_time={self.warp_for_time}"
        return repr_str


@OBJECT_REGISTRY.register
class SpecSub(AudioInterface):
    """时频谱替换.

    以一定概率对音频的时频谱做时频替换增强.

    Args:
        max_t: 时频替换的最大宽度.
        num_t_sub: 时频替换的最大次数.
    """

    def __init__(self, max_t: int = 20, num_t_sub: int = 3):
        self.max_t = max_t
        self.num_t_sub = num_t_sub

    def transform(self, data):
        x = data["audio"]
        y = x.clone().detach()
        max_frames = y.size(0)
        for _ in range(self.num_t_sub):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, self.max_t)
            end = min(max_frames, start + length)
            # only substitute the earlier time chosen randomly for current time
            pos = random.randint(0, start)
            y[start:end, :] = x[(start - pos) : (end - pos), :]
        data["audio"] = y
        return data

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__ + ":"
        repr_str += f"max_t={self.max_t}, "
        repr_str += f"num_t_sub={self.num_t_sub}"
        return repr_str
