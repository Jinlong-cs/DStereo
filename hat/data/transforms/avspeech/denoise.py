"""AVSPEECH(多模语音)音频数据降噪模块.

目前该模块下共有八个类:

LocalHisfDenoiser       : 在线HISF降噪基类.
LocalHisfDenoiserV103   : 在线HISF车载1.0.3版本降噪.
LocalHisfDenoiserV200   : 在线HISF车载2.0.0版本降噪.
PyHisfDenoiser          : 在线HISF WHL版本降噪.
ChannelSelector         : 通道选择.
OnLineChannelSelector   : 在线通道选择.
STOI                    : 短时客观清晰度.
SNRFilter               : 统计信噪比和筛选vad段.

其中 LocalHisfDenoiserV103, LocalHisfDenoiserV200,
PyHisfDenoiser 作为外部调用的接口, 对传入的音频进行在线降噪.
ChannelSelector 和 OnLineChannelSelector 用于降噪之后做通道选择,
STOI 和 SNRFilter 在通道选择时作为工具类使用.
"""

import ctypes
import logging
import math
import random
from os.path import join

import numpy as np

try:
    import pyhisf
except ImportError:
    pyhisf = None

try:
    import soundfile as sf
except ImportError:
    sf = None

import torch
from numpy.fft import rfft

from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages


class LocalHisfDenoiser(object):
    """LocalHisfDenoiser.

    以一定概率对音频进行在线HISF降噪基类.

    Args:
        hisf_root: 在线hisf依赖文件的路径.
        channel_select: 过hisf之后是否做通道选择, 默认为False.
        pre_pad: 是否在过hisf之前对音频做复制拼接, 默认为True.
        p: 应用变换的概率值, 默认为1.0.
        delay: 该版本hisf的延迟.
    """

    def __init__(
        self, hisf_root, channel_select=False, pre_pad=True, p=1.0, delay=None
    ):
        assert delay is not None
        self.hisf_root = hisf_root
        self.channel_select = channel_select
        self.pre_pad = pre_pad
        self.p = p
        self.delay = delay

    def __call__(self, data) -> dict:

        waveform, samplerate = data["waveform"]

        if "source_audio" not in data:
            data["source_audio"] = waveform

        clean_audio = data["source_audio"]
        if random.random() > self.p:
            if data["channel_dim"] == 0:
                waveform = waveform[[0], :]
            else:
                waveform = waveform[:, [0]]
        else:
            if data["channel_dim"] == 0:
                waveform = waveform.transpose(1, 0)
                clean_audio = clean_audio.transpose(1, 0)
            role = data["role"]
            duration = waveform.shape[0] / samplerate
            vad_segment_list = [(0, duration)]
            waveform = waveform.numpy()
            waveform = self.infer(
                waveform, samplerate, vad_segment_list, role, clean_audio
            )
            waveform = torch.tensor(waveform)
            if data["channel_dim"] == 0:
                waveform = waveform.transpose(1, 0)

        data["waveform"] = (waveform, samplerate)
        del data["source_audio"]
        del data["role"]
        return data

    def infer(
        self,
        audio_signal,
        samplerate,
        cur_vad_segment_list,
        role,
        clean_audio_signal,
    ):
        assert audio_signal.shape[1] == 2 and len(audio_signal.shape) == 2
        assert len(cur_vad_segment_list) == 1
        assert (
            samplerate == 16000
        ), f"only support sample rate = 16000, but get {samplerate}"

        if self.pre_pad:
            origin_lens = audio_signal.shape[0]
            duration = origin_lens / samplerate
            copy_times = max(1, math.floor(3.0 / duration))
            copied_audio = [audio_signal for _ in range(copy_times + 1)]
            audio_signal = np.concatenate(copied_audio, axis=0)
            pad_lens = origin_lens * copy_times

        audio_signal = audio_signal * 32768.0
        audio_signal = audio_signal.astype(np.int16)

        # 补充2通道参考信号
        audio_signal_pad = np.zeros_like(audio_signal)
        audio_signal = np.concatenate([audio_signal, audio_signal_pad], axis=1)
        # 添加hisf的延时段音频，保证音频完整
        audio_signal_deley = np.zeros([self.delay, 4], dtype=np.int16)
        audio_signal = np.concatenate(
            [audio_signal, audio_signal_deley], axis=0
        )

        audio_len = audio_signal.shape[0]
        out_len = (audio_len) * 2
        out_buff = (ctypes.c_int16 * out_len)()
        new_audio_signal = audio_signal.flatten()
        self.libhisf.pyHisfProc(
            new_audio_signal.tobytes(),
            audio_len,
            out_buff,
        )
        hisf_out_audio = np.asarray(out_buff).reshape(-1, 2)[
            self.delay : audio_len
        ]

        hisf_out_audio = hisf_out_audio.astype(np.float32) / 32768.0

        if self.pre_pad:
            assert hisf_out_audio.shape[0] == pad_lens + origin_lens, (
                f"pad_lens: {pad_lens}, origin_lens:{origin_lens}"
                f"hisf_out_lens:{hisf_out_audio.shape[0]}"
            )
            # 截取目标段
            hisf_out_audio = hisf_out_audio[pad_lens:]

        if self.channel_select:
            result_wav = self.select_channel(
                clean_audio_signal, hisf_out_audio, cur_vad_segment_list
            )
        else:
            assert role == "pilot" or role == "copilot"
            result_wav = (
                hisf_out_audio[:, [0]]
                if role == "pilot"
                else hisf_out_audio[:, [1]]
            )

        return result_wav

    def select_channel(self, clean_audio, hisf_audio, cur_vad_segment_list):
        channel_selector = OnLineChannelSelector(
            clean_audio,
            hisf_audio,
            cur_vad_segment_list,
        )
        proc_wav = channel_selector.channel_select()
        return proc_wav

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["libhisf"]
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._fork()


@OBJECT_REGISTRY.register
class LocalHisfDenoiserV103(LocalHisfDenoiser):
    """LocalHisfDenoiserV103.

    以一定概率对音频利用在线HISF车载1.0.3版本降噪.

    Args:
        hisf_root: 在线hisf依赖文件的路径.
        channel_select: 过hisf之后是否做通道选择, 默认为False.
        pre_pad: 是否在过hisf之前对音频做复制拼接, 默认为True.
        p: 应用变换的概率值, 默认为1.0.
        delay: 该版本hisf的延迟.
    """

    def __init__(
        self, hisf_root, channel_select=False, pre_pad=True, p=1.0, delay=1920
    ):
        super(LocalHisfDenoiserV103, self).__init__(
            hisf_root=hisf_root,
            channel_select=channel_select,
            pre_pad=pre_pad,
            p=p,
            delay=delay,
        )
        self._fork()

    def _fork(self):
        if getattr(self, "libhisf", None) is None:
            self.libhisf = ctypes.CDLL(
                join(self.hisf_root, "libhisf_loader.so")
            )
            self.libhisf.pyHisfInit(
                ctypes.create_string_buffer(
                    join(self.hisf_root, "hisf_config.ini").encode()
                ),
                ctypes.create_string_buffer(
                    join(self.hisf_root, "hisf_config_outer.ini").encode()
                ),
            )


@OBJECT_REGISTRY.register
class LocalHisfDenoiserV200(LocalHisfDenoiser):
    """LocalHisfDenoiserV200.

    以一定概率对音频利用在线HISF车载2.0.0版本降噪.

    Args:
        hisf_root: 在线hisf依赖文件的路径.
        channel_select: 过hisf之后是否做通道选择, 默认为False.
        pre_pad: 是否在过hisf之前对音频做复制拼接, 默认为True.
        p: 应用变换的概率值, 默认为1.0.
        delay: 该版本hisf的延迟.
    """

    def __init__(
        self, hisf_root, channel_select=False, pre_pad=True, p=1.0, delay=896
    ):
        super(LocalHisfDenoiserV200, self).__init__(
            hisf_root=hisf_root,
            channel_select=channel_select,
            pre_pad=pre_pad,
            p=p,
            delay=delay,
        )
        self._fork()

    def _fork(self):
        if getattr(self, "libhisf", None) is None:
            self.libhisf = ctypes.CDLL(
                join(self.hisf_root, "libhisf_loader.so")
            )
            self.libhisf.pyHisfInit(
                ctypes.create_string_buffer(
                    join(self.hisf_root, "hisf_config_omni.ini").encode()
                ),
                ctypes.create_string_buffer(
                    join(self.hisf_root, "hisf_config_outer.ini").encode()
                ),
            )


class PyHisfDenoiser(object):
    """PyHisfDenoiser.

    以一定概率对音频利用WHL包版本HISF进行降噪.
    Args:
        channel_select: 过hisf之后是否做通道选择, 默认为False.
        pre_pad: 是否在过hisf之前对音频做复制拼接, 默认为True.
        p: 应用变换的概率值, 默认为1.0.
    """

    @require_packages("pyhisf")
    def __init__(self, channel_select=False, pre_pad=True, p=1.0):
        self.channel_select = channel_select
        self.pre_pad = pre_pad
        self.p = p
        self._fork()

    def _fork(self):
        if getattr(self, "libhisf", None) is None:
            self.libhisf = pyhisf.Hisf()

    def __call__(self, data) -> dict:

        waveform, samplerate = data["waveform"]

        if "source_audio" not in data:
            data["source_audio"] = waveform

        clean_audio = data["source_audio"]
        if random.random() > self.p:
            if data["channel_dim"] == 0:
                waveform = waveform[[0], :]
            else:
                waveform = waveform[:, [0]]
        else:
            if data["channel_dim"] == 0:
                waveform = waveform.transpose(1, 0)
                clean_audio = clean_audio.transpose(1, 0)
            role = data["role"]
            duration = waveform.shape[0] / samplerate
            vad_segment_list = [(0, duration)]
            waveform = waveform.numpy()
            waveform = self.infer(
                waveform, samplerate, vad_segment_list, role, clean_audio
            )
            waveform = torch.tensor(waveform)
            if data["channel_dim"] == 0:
                waveform = waveform.transpose(1, 0)

        data["waveform"] = (waveform, samplerate)
        del data["source_audio"]
        del data["role"]
        return data

    def infer(
        self,
        audio_signal,
        samplerate,
        cur_vad_segment_list,
        role,
        clean_audio_signal,
    ):

        assert audio_signal.shape[1] == 2 and len(audio_signal.shape) == 2
        assert len(cur_vad_segment_list) == 1
        assert (
            samplerate == 16000
        ), f"only support sample rate = 16000, but get {samplerate}"

        if self.pre_pad:
            origin_lens = audio_signal.shape[0]
            duration = origin_lens / samplerate
            copy_times = max(1, math.floor(3.0 / duration))
            copied_audio = [audio_signal for _ in range(copy_times + 1)]
            audio_signal = np.concatenate(copied_audio, axis=0)
            pad_lens = origin_lens * copy_times

        audio_signal = audio_signal * 32768.0
        audio_signal = audio_signal.astype(np.int16)

        hisf_out_audio = self.libhisf.proc(audio_signal)

        hisf_out_audio = hisf_out_audio.astype(np.float32) / 32768.0

        if self.pre_pad:
            assert hisf_out_audio.shape[0] == pad_lens + origin_lens, (
                f"pad_lens: {pad_lens}, origin_lens:{origin_lens}"
                f"hisf_out_lens:{hisf_out_audio.shape[0]}"
            )
            # 截取目标段
            hisf_out_audio = hisf_out_audio[pad_lens:]

        if self.channel_select:
            result_wav = self.select_channel(
                clean_audio_signal, hisf_out_audio, cur_vad_segment_list
            )
        else:
            assert role == "pilot" or role == "copilot"
            result_wav = (
                hisf_out_audio[:, [0]]
                if role == "pilot"
                else hisf_out_audio[:, [1]]
            )

        return result_wav

    def select_channel(self, clean_audio, hisf_audio, cur_vad_segment_list):
        channel_selector = OnLineChannelSelector(
            clean_audio,
            hisf_audio,
            cur_vad_segment_list,
        )
        proc_wav = channel_selector.channel_select()

        return proc_wav

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["libhisf"]
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._fork()


class ChannelSelector(object):
    """ChannelSelector.

    从本地文件读取音频, 根据能量策略对音频数据进行通道选择.

    Args:
        clean_wav: 原始音频文件路径.
        hisf_wav: 过hisf后的音频文件路径.
        vad_segment_list: 音频的vad段列表.
    """

    @require_packages("soundfile")
    def __init__(self, clean_wav, hisf_wav, vad_segment_list):
        self.clean_audio, self.fs = sf.read(clean_wav, always_2d=True)
        self.hisf_audio, _ = sf.read(hisf_wav, always_2d=True)
        self.vad_segment_list = vad_segment_list
        self.stoi = STOI(self.fs, 512)

    @staticmethod
    def target_audio_syn(audio, fs, vad, channel):
        assert len(channel) <= len(vad)
        s_out = np.zeros((audio.shape[0], 1), audio.dtype)
        L = audio.shape[0]
        vad_pair = []
        for i in range(len(vad)):
            vad_expand = (vad[i][1] - vad[i][0]) / 2
            vad_start = max(0, int((vad[i][0] - vad_expand) * fs))
            vad_end = min(int((vad[i][1] + vad_expand) * fs), L)
            if i > 0:
                if vad_start > vad_pair[i - 1][1]:
                    ll = math.ceil((vad_start - vad_pair[i - 1][1]) / 2)
                    vad_start -= ll
                    vad_pair[i - 1][1] += ll
            else:
                vad_start = 0
            if i == len(vad) - 1:
                vad_end = L
            vad_pair.append([int(vad_start), int(vad_end)])
        for i in range(len(vad_pair)):
            s_out[vad_pair[i][0] : vad_pair[i][1], 0] = audio[
                vad_pair[i][0] : vad_pair[i][1], channel[i]
            ]

        return s_out

    def audio_quality_eval(self, max_channels=2):
        if self.hisf_audio is None:
            raise TypeError("Empty hisf wav")
        logging.debug(
            f"fs: {self.fs}, vad_segment_list: {self.vad_segment_list}"
        )
        seg_pos_ref, _ = SNRFilter.cut_audio_by_vad(
            self.clean_audio, self.fs, self.vad_segment_list
        )

        seg_pos_pro, _ = SNRFilter.cut_audio_by_vad(
            self.hisf_audio, self.fs, self.vad_segment_list
        )
        channels = min(max_channels, self.hisf_audio.shape[1])
        logging.debug(
            f"seg_pos_ref: {seg_pos_ref[0].shape}, \
                seg_pos_pro: {seg_pos_pro[0].shape}"
        )
        stoi_score_lst = []
        for c in range(channels):
            stoi_score_lst.append(
                [
                    self.stoi.score(proc[:, c], ref[:, 0])
                    for proc, ref in zip(seg_pos_pro, seg_pos_ref)
                ]
            )

        stoi_score = list(zip(*stoi_score_lst))

        return stoi_score

    def channel_select(self, out_file=None):
        stoi_score = self.audio_quality_eval()
        channels = np.argmax(np.array(stoi_score), axis=1)
        audio_out = self.target_audio_syn(
            self.hisf_audio, self.fs, self.vad_segment_list, channels
        )
        if out_file is None:
            return audio_out
        else:
            sf.write(out_file, audio_out, self.fs, "PCM_16")


class OnLineChannelSelector(ChannelSelector):
    """OnLineChannelSelector.

    根据能量策略对通过参数传入的音频数据进行通道选择.

    Args:
        clean_audio: 原始音频数据.
        hisf_audio: 过hisf后的音频数据.
        vad_segment_list: 音频的vad段列表.
        fs: 音频数据对应的采样率.
    """

    def __init__(self, clean_audio, hisf_audio, vad_segment_list, fs=16000):
        self.clean_audio, self.fs = clean_audio, fs
        self.hisf_audio = hisf_audio
        self.vad_segment_list = vad_segment_list
        self.stoi = STOI(self.fs, 512)


class STOI(object):
    """STOI.

    短时客观清晰度.

    Args:
        fs: 采样率.
        fralen: 窗长.
        fmin: 下限频率.
        fmax: 上限频率.
        cf0: 第一个频率子带的中心频率.
        lctx: 上文长度.
        clip_dB: 截断能量.
        eps: 保护小值.
    """

    def __init__(
        self,
        fs,
        fralen,
        fmin=50,
        fmax=7500,
        cf0=150,
        lctx=24,
        clip_dB=-15,
        eps=1e-10,
    ):
        self._nb = int(3 * np.log2(fs / (2 * cf0))) + 1
        self._lctx = lctx
        self._fralen = fralen
        self.weight = self.get_thirdoct_weight(
            fs, fralen, self._nb, cf0, fmax, fmin
        )
        self._eps = eps
        self._clip = 1 + 10 ** (-clip_dB / 20)
        self._win = self.get_win_hanning(fralen)

    def score(self, deg, ref):
        logging.debug(f"deg: {deg.shape}, ref: {ref.shape}")
        deg_stft = self.stft(
            deg, self._fralen, self._fralen // 2, self._fralen, self._win
        )
        ref_stft = self.stft(
            ref, self._fralen, self._fralen // 2, self._fralen, self._win
        )

        deg_psd = self.psd(deg_stft)
        ref_psd = self.psd(ref_stft)

        deg_oct = self.freq2oct(deg_psd)
        ref_oct = self.freq2oct(ref_psd)

        deg_ctx = self.expand_ctx(deg_oct, self._lctx)
        ref_ctx = self.expand_ctx(ref_oct, self._lctx)

        stoi_ctx = self.pearson_cor(deg_ctx, ref_ctx, 1)
        stoi = np.mean(stoi_ctx)

        return stoi

    def freq2oct(self, x):
        """freq2oct.

        x[T,F]
        weight[Foct,F]
        x_oct[T, Foct]
        """
        x_oct = np.einsum(x, [0, 1], self.weight, [2, 1], [0, 2])
        return x_oct

    def pearson_cor(self, x, y, axis):
        x = self.rmdc(x, axis)
        y = self.rmdc(y, axis)

        xy_dot = self.dot(x, y, axis, keepdims=True)
        xx_dot = self.dot(x, x, axis, keepdims=True)
        yy_dot = self.dot(y, y, axis, keepdims=True)

        cor = xy_dot / np.sqrt(xx_dot * yy_dot + self._eps)

        return cor

    def expand_ctx(self, x, ctx):
        T = x.shape[0] - ctx
        if T <= 0:
            x_lst = [x]
        else:
            x_lst = [x[i : i + ctx, :] for i in range(T)]
        x_out = np.stack(x_lst, axis=1)
        return x_out

    @staticmethod
    def rmdc(x, axis):
        return x - np.mean(x, axis=axis, keepdims=True)

    @staticmethod
    def dot(x, y, axis, keepdims):
        return np.sum(x * y, axis=axis, keepdims=keepdims)

    @staticmethod
    def get_thirdoct_weight(fs, fralen, nb, cf0, fmax, fmin):
        df = fs / fralen
        cf = np.power(2, np.arange(0, nb) / 3) * cf0
        fl = cf * 2 ** (-1 / 6)
        fh = cf * 2 ** (1 / 6)
        assert fmin < fl[1] and fmax > fh[-2]
        fl[0] = fmin
        fh[-1] = fmax
        fl_idx = np.around(fl / df).astype(np.int16)
        fh_idx = np.minimum(
            np.around(fh / df).astype(np.int16) - 1, int(fralen / 2)
        )
        w = np.zeros((nb, fralen // 2 + 1), dtype=np.float32)
        for i in range(nb):
            w[i, fl_idx[i] : fh_idx[i] + 1] = 1
        return w[:, :]

    @staticmethod
    def get_win_hanning(size):
        a = 2 * np.pi / size
        f = np.arange(size)
        window = 0.54 - 0.46 * np.cos(a * f)
        return window

    @staticmethod
    def stft(s, fralen, hop, nfft, win):
        win = win.reshape(1, -1)
        logging.debug(
            f"s:{s.shape}, fralen:{fralen}, hop:{hop}, nfft:{nfft}, win:{win}"
        )
        s_frames_list = [
            s[i : i + fralen] for i in range(0, s.shape[0] - fralen + 1, hop)
        ]
        s_frames = np.stack(s_frames_list, axis=0)  # [num_frame, fralen]
        s_frames = s_frames * win

        S = rfft(s_frames, n=nfft, axis=1)
        return S

    @staticmethod
    def psd(x):
        return x.real * x.real + x.imag * x.imag


class SNRFilter(object):
    """SNRFilter.

    获取音频的信噪比, 或对音频的vad段进行筛选.

    Args:
        audio_wav: 音频文件路径.
    """

    @require_packages("soundfile")
    def __init__(self, audio_wav):
        self.audio_wav_path = audio_wav
        self.audio, self.fs = sf.read(audio_wav, always_2d=True)

    @staticmethod
    def extract_vad_from_json(data):
        vad = [
            (i["vad_beg"], i["vad_end"], i["text"]) for i in data["sentence"]
        ]
        return vad

    @staticmethod
    def cut_audio_by_vad(audio, fs, vad):
        s = audio
        L = s.shape[0]
        logging.debug(f"Length: {L}")
        # 去掉无效的vad段
        tmp_vad = vad
        vad = []
        audio_end = float(L) / fs
        for vad_ in tmp_vad:
            vad_ = list(vad_)
            if vad_[0] > audio_end or vad_[1] - audio_end > 0.01:
                continue
            if vad_[1] - audio_end > 0 and vad_[1] - audio_end < 0.01:
                vad_[1] = audio_end
            vad_ = tuple(vad_)
            vad.append(vad_)
        vad_active_seg = [
            s[max(0, int(i[0] * fs)) : min(int(i[1] * fs), L)] for i in vad
        ]

        vad_disable = [[vad[i][1], vad[i + 1][0]] for i in range(len(vad) - 1)]
        vad_disable_seg = [
            s[max(0, int(i[0] * fs)) : min(int(i[1] * fs), L)]
            for i in vad_disable
        ]

        return vad_active_seg, vad_disable_seg

    @staticmethod
    def dot(x, y, axis, keepdims):
        return np.sum(x * y, axis=axis, keepdims=keepdims)

    def sisnr(self, process, ref):
        process = process - np.mean(process, axis=0, keepdims=True)
        ref = ref - np.mean(ref, axis=0, keepdims=True)
        pr_dot = self.dot(process, ref, axis=0, keepdims=True)
        rr_dot = self.dot(ref, ref, axis=0, keepdims=True)
        p_proj = pr_dot * ref / (rr_dot + 1e-10)
        e_proj = process - p_proj

        pp_proj_dot = self.dot(p_proj, p_proj, axis=0, keepdims=False)
        ee_proj_dot = self.dot(e_proj, e_proj, axis=0, keepdims=False)

        sisnr = ee_proj_dot / (pp_proj_dot + 1e-10)

        return 10 * np.log10(sisnr + 1e-10)

    @staticmethod
    def audio_rms(audio):
        return np.sqrt(np.mean(audio * audio))

    def get_snr(self, channel, vad_segment_list, filter_outlier=False):
        seg_pos, seg_neg = self.cut_audio_by_vad(
            self.audio, self.fs, vad_segment_list
        )
        if filter_outlier:
            seg_pos = self.filter_outlier_segment(seg_pos)
            seg_neg = self.filter_outlier_segment(seg_neg)

        pos_wav = np.concatenate(seg_pos, axis=0)[:, channel]
        pos_rms = self.audio_rms(pos_wav)
        neg_wav = np.concatenate(seg_neg, axis=0)[:, channel]
        neg_rms = self.audio_rms(neg_wav)

        snr = 20 * np.log10(pos_rms / max(neg_rms, 1e-10))

        return snr

    def filter_outlier_segment(self, audio_segments):
        rms_list = []
        audio_idx_list = []
        for idx, segment in enumerate(audio_segments):
            if segment.shape[0] == 0:
                continue
            rms_list.append(self.audio_rms(segment))
            audio_idx_list.append(idx)
        rms_array = np.array(rms_list)
        rms_mean = np.mean(rms_array)
        rms_std = np.std(rms_array)
        remain = np.abs(rms_array - rms_mean) < 3 * rms_std
        remain_segments = []
        for idx, remain in zip(audio_idx_list, remain):
            if remain:
                remain_segments.append(audio_segments[idx])
        assert len(remain_segments) > 0
        return remain_segments
