# Copyright (c) Horizon Robotics, All rights reserved.
"""AVSPEECH(多模语音)音频数据读取模块.

目前该模块下共有四个类、一个函数:

BaseWaveformReader       : 音频数据读取基类.
HDF5WaveformReader       : 从hdf5文件读取音频数据.
TendisWaveformReader     : 从tendis数据库读取音频数据.
DirectWaveformReader     : 从音频文件读取音频数据.

convert_waveform_to_float: 将音频数据转为float类型.

其中, HDF5WaveformReader、TendisWaveformReader 和 DirectWaveformReader
都继承了 BaseWaveformReader 类, 用于读取音频数据. convert_waveform_to_float
函数用于读取到音频数据之后对其数据类型做转换, 可选.
"""

import logging
import math
import pickle
from typing import Any, Mapping, Optional, Tuple

import h5py
import numpy as np

try:
    import soundfile
except ImportError:
    soundfile = None

from hat.data.datasets.avspeech.redis_client import (
    TendisClient,
    multimodal_default_rename_handle,
)
from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages

logger = logging.getLogger(__name__)

SUBTYPE_2_DTYPE = {
    "PCM_16": "int16",
    "PCM_S8": "int8",
    "PCM_32": "float32",
    "DOUBLE": "float64",
}


def convert_waveform_to_float(
    data: np.ndarray, dtype: str = "float32"
) -> np.ndarray:
    """convert_waveform_to_float.

    将waveform转换成float类型. 由于waveform读取出来可能是int16或者int8类型的数据,
    该接口将其转换为float类型的数组.

    Args:
        data: 需要处理的waveform数组, 当数组是浮点数组时不处理;
              当数组是int型数组时, 所属int型的取值空间转换到[-1, 1]之间的浮点型.
        dtype: 目标float数组, 支持np.floating的所有float类型. 默认是"float32".
    """

    assert np.issubdtype(dtype, np.floating), "dtype is not an ``np.floating``"
    if np.issubdtype(data.dtype, np.floating):
        data = data.astype(dtype)
    elif np.issubdtype(data.dtype, np.signedinteger):
        iinfo = np.iinfo(data.dtype)
        data = data.astype(dtype) / (-iinfo.min)
    else:
        raise NotImplementedError(f"{data.dtype} is not support now!")
    return data


class BaseWaveformReader(object):
    """音频数据读取基类."""

    def read_seg_audio(  # type: ignore[no-untyped-def]
        self, *args, **kwargs
    ) -> Tuple[np.ndarray, int]:
        """所有子类都应该实现该接口."""
        raise NotImplementedError


@OBJECT_REGISTRY.register
class HDF5WaveformReader(BaseWaveformReader):
    """HDF5WaveformReader.

    该类实现了从hdf5文件读取音频数据的接口.

    Args:
        hdf5_file: hdf5文件.
        ret_dtype: 目标数据类型, 默认为"float32".
        channel_first: 如果设置为True, 将通道放在数组第一维, 默认为True.
    """

    group = "audio"
    _SUPPORT_DTYPE = ["int16", "int32", "float32", "float64"]

    def __init__(
        self,
        hdf5_file: str,
        ret_dtype: str = "float32",
        channel_first: bool = True,
    ):
        self.hdf5_file = hdf5_file
        assert (
            ret_dtype in HDF5WaveformReader._SUPPORT_DTYPE
        ), f"Only support {HDF5WaveformReader._SUPPORT_DTYPE}, but get ``{ret_dtype}``"  # noqa: E501
        self.ret_dtype = ret_dtype
        self.channel_first = channel_first
        self._fork()

    def _fork(self):  # type: ignore
        if getattr(self, "hdf5_fileno", None) is None:
            self.hdf5_fileno = h5py.File(str(self.hdf5_file), "r")
        else:
            logger.error(f"UNKNOWN HDF5 FILENO : {self.hdf5_fileno.name}")

    def read_seg_audio(  # type: ignore[override]
        self, utt: str, seg_beg: float, seg_end: float
    ) -> Tuple[np.ndarray, int]:
        """read_seg_audio.

        读取片段对应的音频数据.

        Args:
            utt: 片段对应音频的utt_id, 格式是'${batch_name}-${audio_name}'.
                 示例：'J2MM_develop_speech_batch02-00146_02_04_mic1'.
            seg_beg: 片段的开始时间点, 单位为秒.
            seg_end: 片段的结束时间点, 单位为秒.

        Returns:
            Tuple[np.ndarray, int].
            返回片段对应的音频数据和采样率.
        """

        logger.debug(f"seg_info: {seg_beg} {seg_end}")
        # 获取数据
        seg_beg_sec = math.floor(seg_beg)
        seg_end_sec = math.ceil(seg_end)
        logger.debug(f"sec index: {seg_beg_sec} {seg_end_sec}")
        waveform = self.hdf5_fileno[self.group][utt][seg_beg_sec:seg_end_sec]
        # 读取音频信息并变形
        attrs = self.hdf5_fileno[self.group][utt].attrs
        channels = attrs["channels"]
        waveform = [wf.reshape((-1, channels)) for wf in waveform]
        waveform = np.concatenate(waveform, axis=0)
        # 截取数据
        duration = attrs["duration"]
        samplerate = attrs["samplerate"]
        seg_vad_dur = int(samplerate * (seg_end - seg_beg))
        seg_beg_idx = int(samplerate * (seg_beg - seg_beg_sec))
        seg_end_idx = min(waveform.shape[0], seg_beg_idx + seg_vad_dur)
        if seg_end_idx - seg_beg_idx < seg_vad_dur:
            msg = f"{utt}:{seg_beg}:{seg_end} with duration {duration} maybe error label"  # noqa: E501
            logging.warning(msg)
        logging.debug(f"seg index: {seg_beg_idx} {seg_end_idx}")
        waveform = waveform[seg_beg_idx:seg_end_idx]
        # 转成目标dtype
        if self.ret_dtype in ["float32", "float64"]:
            waveform = convert_waveform_to_float(waveform, self.ret_dtype)
        # 转换 channel 的维度位置
        if self.channel_first:
            waveform = waveform.transpose(1, 0)
        return waveform, samplerate

    def read_attrs(self, utt: str):
        attrs = self.hdf5_fileno[self.group][utt].attrs
        return attrs

    def __getstate__(self):  # type: ignore
        state = self.__dict__.copy()
        del state["hdf5_fileno"]
        return state

    def __setstate__(self, state):  # type: ignore
        self.__dict__ = state
        self._fork()


@OBJECT_REGISTRY.register
class TendisWaveformReader(BaseWaveformReader):
    """TendisWaveformReader.

    该类实现了从tendis数据库读取音频数据的接口.

    Args:
        zfill_num: 整数规范化的位数, 默认为6.
        ret_dtype: 目标数据类型, 默认为"float32".
        channel_first: 如果设置为True, 将通道放在数组第一维, 默认为True.
        tendis_kwargs: tendis数据库相关参数.
    """

    _SUPPORT_DTYPE = ["int16", "int32", "float32", "float64"]
    KEY_HEAD = "Key2Waveform"

    def __init__(
        self,
        zfill_num: int = 6,
        ret_dtype: str = "float32",
        channel_first: bool = True,
        tendis_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        assert (
            ret_dtype in TendisWaveformReader._SUPPORT_DTYPE
        ), f"Only support {TendisWaveformReader._SUPPORT_DTYPE}, but get ``{ret_dtype}``"  # noqa: E501
        self.zfill_num = zfill_num
        self.ret_dtype = ret_dtype
        self.channel_first = channel_first
        self.tendis_kwargs = {} if tendis_kwargs is None else tendis_kwargs
        self._fork()

    def _fork(self):  # type: ignore
        if getattr(self, "db_client", None) is None:
            self.db_client = TendisClient(
                rename_handle=multimodal_default_rename_handle,
                kwargs=self.tendis_kwargs,
            )

    def read_seg_audio(  # type: ignore[override]
        self, utt: str, seg_beg: float, seg_end: float
    ) -> Tuple[np.ndarray, int]:
        """read_seg_audio.

        读取片段对应的音频数据.

        Args:
            utt: 片段对应音频的utt_id, 格式是'${batch_name}-${audio_name}'.
                 示例：'J2MM_develop_speech_batch02-00146_02_04_mic1'.
            seg_beg: 片段的开始时间点, 单位为秒.
            seg_end: 片段的结束时间点, 单位为秒.

        Returns:
            Tuple[np.ndarray, int].
            返回片段对应的音频数据和采样率.
        """

        logging.debug(f"seg_info: {seg_beg} {seg_end}")
        # 读取音频信息和数据
        seg_beg_sec = math.floor(seg_beg)
        seg_end_sec = math.ceil(seg_end)
        logging.debug(f"sec index: {seg_beg_sec} {seg_end_sec}")
        attrs_key = f"{self.KEY_HEAD}_{utt}_attrs"
        wav_keys = [
            f"{self.KEY_HEAD}_{utt}_{idx :>0{self.zfill_num}d}"
            for idx in range(seg_beg_sec, seg_end_sec)
        ]
        keys = [attrs_key, *wav_keys]
        attrs, *waveform = self.db_client.mget(keys)
        try:
            attrs = pickle.loads(attrs)
        except TypeError:
            msg = f"{utt}:{seg_beg}:{seg_end} has no attrs."
            return None, None
        if any(wf is None for wf in waveform):
            err = ",".join(
                f"{key}" for key, wf in zip(wav_keys, waveform) if wf is None
            )
            msg = f"{utt}:{seg_beg}:{seg_end} has none data. which is [{err}]"
            logging.warning(msg)
        channels = attrs["channels"]
        dtype = attrs["dtype"]
        samplerate = attrs["samplerate"]
        waveform = [
            np.frombuffer(wf, dtype=dtype).reshape((-1, channels))
            if wf is not None
            else np.zeros((samplerate, channels))
            for wf in waveform
        ]
        waveform = np.concatenate(waveform, axis=0)
        # 截取数据
        duration = attrs["duration"]
        seg_vad_dur = int(samplerate * (seg_end - seg_beg))
        seg_beg_idx = int(samplerate * (seg_beg - seg_beg_sec))
        seg_end_idx = min(waveform.shape[0], seg_beg_idx + seg_vad_dur)
        if seg_end_idx - seg_beg_idx < seg_vad_dur:
            msg = f"{utt}:{seg_beg}:{seg_end} with duration {duration} maybe error label"  # noqa: E501
            logging.warning(msg)
        logging.debug(f"seg index: {seg_beg_idx} {seg_end_idx}")
        waveform = waveform[seg_beg_idx:seg_end_idx]
        # 转成目标dtype
        if self.ret_dtype in ["float32", "float64"]:
            waveform = convert_waveform_to_float(waveform, self.ret_dtype)
        # 转换 channel 的维度位置
        if self.channel_first:
            waveform = waveform.transpose(1, 0)
        return waveform, samplerate

    def __getstate__(self):  # type: ignore
        state = self.__dict__.copy()
        del state["db_client"]
        return state

    def __setstate__(self, state):  # type: ignore
        self.__dict__ = state
        self._fork()


@OBJECT_REGISTRY.register
class DirectWaveformReader(BaseWaveformReader):
    """DirectWaveformReader.

    该类实现了从音频文件直接读取音频数据的接口.

    Args:
        ret_dtype: 目标数据类型, 默认为"float32".
        channel_first: 如果设置为True, 将通道放在数组第一维, 默认为True.
    """

    group = "audio"
    _SUPPORT_DTYPE = ["int16", "int32", "float32", "float64"]

    @require_packages("soundfile")
    def __init__(self, ret_dtype: str = "float32", channel_first: bool = True):
        assert (
            ret_dtype in DirectWaveformReader._SUPPORT_DTYPE
        ), f"Only support {DirectWaveformReader._SUPPORT_DTYPE}, but get ``{ret_dtype}``"  # noqa: E501
        self.ret_dtype = ret_dtype
        self.channel_first = channel_first

    def read_seg_audio(self, wav_path: str) -> Tuple[np.ndarray, int]:
        """read_seg_audio.

        从音频文件中读取音频数据.

        Args:
            wav_path: 音频文件.

        Returns:
            Tuple[np.ndarray, int].
            返回音频数据和采样率.
        """

        waveform, samplerate = soundfile.read(
            wav_path, dtype=self.ret_dtype, always_2d=True
        )
        # 转换 channel 的维度位置
        if self.channel_first:
            waveform = waveform.transpose(1, 0)
        return waveform, samplerate
