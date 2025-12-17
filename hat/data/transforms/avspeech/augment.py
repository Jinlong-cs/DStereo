"""AVSPEECH(多模语音)音频数据增强模块.

目前该模块下共有六个类:

OnLineAugmentPipeline       : 对多通道音频数据做通道选择和增强.
OnLineJsonAugmentPipeline   : 对短视频数据中的音频数据做增强.
SignalAugmentor             : 对信号做线性加噪、仿真等增强.
NoiseSampler                : 噪声采样器.
AudioNoiseRetriever         : 噪声读取器.
RIRSimulator                : RIR信号仿真器.

其中, OnLineAugmentPipeline 和 OnLineJsonAugmentPipeline 作为外部调用的接口,
对传入的音频数据做随机增强. 加噪时, NoiseSampler 作为噪声采样器决定每次获取什么
类型的噪声, 以及确定噪声ID、 加噪信噪比; AudioNoiseRetriever 从噪声文件中读取
噪声数据. RIRSimulator 只有在仿真时才会调用, 用于仿真RIR信号.
"""

import io
import logging
import os
import random
from os.path import join
from typing import List, Tuple

import h5py
import numpy as np
import scipy.signal as ss
import torch
import yaml
from easydict import EasyDict as edict
from scipy.optimize import minimize

from hat.data.datasets.avspeech.audio import HDF5WaveformReader
from hat.data.datasets.avspeech.info import UttParser
from hat.data.transforms.avspeech import signal_utils
from hat.utils.package_helper import require_packages

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import pyroomacoustics as pra
except ImportError:
    pra = None

try:
    from hat.data.datasets.pack_type.mxrecord import MXRecord
except ImportError:
    MXRecord = None

try:
    from gpuRIR_bind import gpuRIR_bind
except ImportError:
    gpuRIR_bind = None

try:
    import mxnet as mx
except ImportError:
    mx = None


from hat.registry import OBJECT_REGISTRY

EPS = np.finfo(np.float32).eps


@OBJECT_REGISTRY.register
class OnLineAugmentPipeline(object):
    """OnLineAugmentPipeline.

    要求调用时传入的data包含 "waveform" 和 "key" 键,
    对多通道音频数据做通道选择, 并以一定概率对音频数据做增强.

    Args:
        config: 加噪配置文件.
        noise_hdf5_file: 噪声hdf5文件路径.
        ret_dtype: 读取噪声的数据类型.
        local_rir: 是否使用本地缓存的rir,
                   如果设置为True, 则使用,
                   如果设置为False, 则不使用.
        gpu_rir_path: 本地缓存的rir信号路径.
        noise_info: 噪声信息文件路径.
        p: 应用变换的概率值.
    """

    def __init__(
        self,
        config: dict,
        noise_hdf5_file: str,
        ret_dtype: str = "float64",
        local_rir: bool = True,
        gpu_rir_path: str = None,
        noise_info: str = None,
        p: float = 1.0,
    ):
        config = edict(config)
        self._config = config.augment_and_denoise_kwargs
        self.add_noise_kwargs = self._config.add_noise_kwargs
        bgnk = self._config.background_noise_kwargs
        self.noise_retriever = AudioNoiseRetriever(
            hdf5_file=noise_hdf5_file,
            ret_dtype=ret_dtype,
            rec_content_type=bgnk.rec_content_type,
            rec_info=bgnk.noise_rec_kwargs,
            noise_lst_groups=bgnk.noise_lst_groups,
            noise_lst_prefix=bgnk.noise_lst_prefix,
            noise_info=noise_info,
            noise_distribution=bgnk.noise_probs,
            signal_to_noise_snr_distribution=bgnk.snr_distribution,
            noise_mixture_snr_kwargs=bgnk.noise_mixture_snr_kwargs,
            mode=self._config.mode,
            noise_simulate=bgnk.noise_simulate,
            gpu_rir_path=gpu_rir_path,
            local_rir=local_rir,
        )
        self.signal_augmentor = SignalAugmentor(
            normalize_signal=self._config.signal_augmentor_normalize_signal,
            gpu_rir_path=gpu_rir_path,
            local_rir=local_rir,
        )
        self.utt_parser = UttParser()
        self.ret_type = ret_dtype
        self.p = p

    def __call__(self, data):

        assert (
            "waveform" in data
        ), f"{__class__.__name__} use ``waveform`` in data"

        assert "key" in data, f"{__class__.__name__} use ``key`` in data"
        if data["waveform"] is None:
            return data

        seg_utt = data["key"]
        matched = self.utt_parser.match(seg_utt)
        dataset = matched["dataset"]
        role = data["role"]
        waveform, samplerate = data["waveform"]
        if data["channel_dim"] == 0:
            waveform = waveform.transpose(1, 0)
        waveform = waveform.numpy()
        mixed_audio, source_audio, role = self.infer(dataset, role, waveform)
        mixed_audio = torch.tensor(mixed_audio, dtype=torch.float32)
        source_audio = torch.tensor(source_audio, dtype=torch.float32)
        if data["channel_dim"] == 0:
            mixed_audio = mixed_audio.transpose(1, 0)
            source_audio = source_audio.transpose(1, 0)
        data["waveform"] = (mixed_audio, samplerate)
        data["source_audio"] = source_audio
        data["role"] = role

        return data

    def infer(self, dataset, role, wav_data):

        assert len(wav_data.shape) == 2

        add_noise_type = self._config.add_noise_type[dataset]
        wav_channels = self._config.wav_channels[dataset]
        source_audio = wav_data[:, wav_channels]

        if random.random() > self.p:
            if add_noise_type == "cabin":
                noise_signal, snr = (None, None)
            else:
                add_noise_type = None
                mixed_audio = source_audio
        else:
            source_signal_length = source_audio.shape[0]
            noise_signal, snr = self.noise_retriever.get_noise(
                source_signal_length
            )

        if add_noise_type == "linearly":
            (
                mixed_audio,
                normed_src_audio,
            ) = self.signal_augmentor.add_noise_linearly(
                source_audio,
                noise_signal,
                snr,
            )
        elif add_noise_type == "cabin":
            role = random.choice(["pilot", "copilot"])
            (
                mixed_audio,
                normed_src_audio,
                steer_vector,
            ) = self.signal_augmentor.add_noise_in_cabin(
                source_audio,
                noise_signal=noise_signal,
                noise_snr=snr,
                **self.add_noise_kwargs,
                target_pos=role,
            )

        return (
            mixed_audio,
            source_audio,
            role,
        )


@OBJECT_REGISTRY.register
class OnLineJsonAugmentPipeline(OnLineAugmentPipeline):
    """OnLineJsonAugmentPipeline.

    要求调用时传入的data包含 "waveform" 键,
    对音频数据做通道合并, 并以一定概率对短视频数据中的音频数据做增强.

    Args:
        config: 加噪配置文件.
        noise_hdf5_file: 噪声hdf5文件路径.
        ret_dtype: 读取噪声的数据类型.
        local_rir: 是否使用本地缓存的rir,
                   如果设置为True, 则使用,
                   如果设置为False, 则不使用.
        gpu_rir_path: 本地缓存的rir信号路径.
        noise_info: 噪声信息文件路径.
        p: 应用变换的概率值.
    """

    @require_packages("pyroomacoustics")
    def __init__(
        self,
        config: dict,
        noise_hdf5_file: str,
        ret_dtype: str = "float64",
        local_rir: bool = True,
        gpu_rir_path: str = None,
        noise_info: str = None,
        p: float = 1.0,
    ):
        super(OnLineJsonAugmentPipeline, self).__init__(
            config=config,
            noise_hdf5_file=noise_hdf5_file,
            ret_dtype=ret_dtype,
            local_rir=local_rir,
            gpu_rir_path=gpu_rir_path,
            noise_info=noise_info,
            p=p,
        )

    def __call__(self, data):

        assert (
            "waveform" in data
        ), f"{__class__.__name__} use ``waveform`` in data"

        if data["waveform"] is None:
            return data

        waveform, samplerate = data["waveform"]
        if data["channel_dim"] == 0:
            waveform = waveform.transpose(1, 0)
        waveform = waveform.numpy()
        mixed_audio, source_audio = self.infer(waveform)
        mixed_audio = torch.tensor(mixed_audio, dtype=torch.float32)
        source_audio = torch.tensor(source_audio, dtype=torch.float32)
        if data["channel_dim"] == 0:
            mixed_audio = mixed_audio.transpose(1, 0)
            source_audio = source_audio.transpose(1, 0)
        data["waveform"] = (mixed_audio, samplerate)
        data["source_audio"] = source_audio

        return data

    def infer(self, wav_data):

        assert len(wav_data.shape) == 2
        assert wav_data.shape[1] == 2

        source_audio = (wav_data[:, [0]] + wav_data[:, [1]]) / 2
        source_audio = np.concatenate([source_audio, source_audio], axis=1)

        if random.random() > self.p:
            mixed_audio = source_audio
        else:
            source_signal_length = source_audio.shape[0]
            noise_signal, snr = self.noise_retriever.get_noise(
                source_signal_length
            )
            (
                mixed_audio,
                normed_src_audio,
            ) = self.signal_augmentor.add_noise_linearly(
                source_audio,
                noise_signal,
                snr,
            )

        return (
            mixed_audio,
            source_audio,
        )


class SignalAugmentor(object):
    """SignalAugmentor.

    对传入的信号做增强, 支持线性加噪、CPU仿真、GPU仿真.

    Args:
        normalize_signal: 是否对增强后的信号做归一化,
                          如果设置为True, 做归一化,
                          如果设置为False, 不做归一化.
        gpu_rir_path: 本地缓存的rir信号路径.
        local_rir: 是否使用本地缓存的rir,
                   如果设置为True, 则使用,
                   如果设置为False, 则不使用.
    """

    @require_packages("pyroomacoustics!=0.4.2")
    def __init__(
        self,
        normalize_signal: bool,
        gpu_rir_path: str,
        local_rir: bool,
    ):
        self._local_rir = local_rir
        self._gpu_rir_path = gpu_rir_path
        self._normalize_signal = normalize_signal
        self._current_device = torch.cuda.current_device()
        self._fork()

    def _fork(self):
        if self._local_rir:
            if getattr(self, "gpu_rir_reader", None) is None:
                self.gpu_rir_reader = h5py.File(self._gpu_rir_path, "r")
        torch.cuda.set_device(self._current_device)

    def add_noise_linearly(
        self,
        src_signal,
        noise_signal,
        snr,
        src_vad_list=None,
        noise_vad_list=None,
    ):
        """add_noise_linearly.

        对一条原始数据线性叠加一条噪声.

        Args:
            src_signal: 原始数据.
            noise_signal: 噪声数据.
            snr: 信噪比.
            src_vad_list: 原始数据的vad段列表.
            noise_vad_list: 噪声数据的vad段列表.
        """

        if src_vad_list is None or len(src_vad_list) == 0:
            src_power = signal_utils.get_signal_average_power(src_signal)
        else:
            src_power = signal_utils.get_signal_segments_average_power(
                src_signal, src_vad_list
            )

        if noise_vad_list is None:
            noise_power = signal_utils.get_signal_average_power(noise_signal)
        else:
            noise_power = signal_utils.get_signal_segments_average_power(
                noise_signal, noise_vad_list
            )

        if noise_power > 0:
            scale_ratio = np.sqrt(src_power / (10 ** (snr / 10) * noise_power))
        else:
            scale_ratio = 0
        assert src_signal.shape[0] == noise_signal.shape[0]
        if len(src_signal.shape) != len(noise_signal.shape):
            noise_signal = noise_signal[:, np.newaxis]
            logging.warning(
                f"source signal has {src_signal.shape[1]} channel and "
                f"noise signal has {noise_signal.shape[1]} channel"
            )
            logging.warning("Copy noise signal from 1 channel to 2 channels.")
        mixed_signal = src_signal + noise_signal * scale_ratio

        if self._normalize_signal:
            mean_value = np.mean(mixed_signal)
            mixed_signal = mixed_signal - mean_value
            max_value = np.max(np.abs(mixed_signal) + 1e-20)
            mixed_signal = mixed_signal / max_value
            src_signal = src_signal / max_value
            return mixed_signal, src_signal
        elif mixed_signal.max() > 1:
            max_value = mixed_signal.max()
            mixed_signal = mixed_signal / max_value
            src_signal = src_signal / max_value

        return mixed_signal, src_signal

    def add_noise_list_linearly(
        self,
        src_signal,
        noise_signal_list,
        snr_list,
        src_vad_list=None,
        noise_vad_list_list=None,
    ):
        """add_noise_list_linearly.

        对一条原始数据线性叠加多条噪声.

        Args:
            src_signal: 原始数据.
            noise_signal_list: 噪声数据列表.
            snr_list: 信噪比列表.
            src_vad_list: 原始数据的vad段列表.
            noise_vad_list_list: 噪声数据vad段列表的列表.
        """

        if src_vad_list is None or len(src_vad_list) == 0:
            src_power = signal_utils.get_signal_average_power(src_signal)
        else:
            src_power = signal_utils.get_signal_segments_average_power(
                src_signal, src_vad_list
            )

        mixed_signal = src_signal
        for indx, noise_signal in enumerate(noise_signal_list):
            if noise_vad_list_list is None:
                noise_vad_list = None
            else:
                noise_vad_list = noise_vad_list_list[indx]

            if noise_vad_list is None or len(noise_vad_list) == 0:
                noise_power = signal_utils.get_signal_average_power(
                    noise_signal
                )
            else:
                noise_power = signal_utils.get_signal_segments_average_power(
                    noise_signal, noise_vad_list
                )

            snr = snr_list[indx]
            if noise_power > 0:
                scale_ratio = np.sqrt(
                    src_power / (10 ** (snr / 10) * noise_power)
                )
            else:
                scale_ratio = 0

            assert src_signal.shape[0] == noise_signal.shape[0]
            if len(src_signal.shape) != len(noise_signal.shape):
                noise_signal = noise_signal[:, np.newaxis]
                logging.warning(
                    "Expand noise signal from 1 channel to 2 channels."
                )

            mixed_signal = mixed_signal + noise_signal * scale_ratio

        if self._normalize_signal:
            mean_value = np.mean(mixed_signal)
            mixed_signal = mixed_signal - mean_value
            max_value = np.max(np.abs(mixed_signal) + 1e-20)
            mixed_signal = mixed_signal / max_value
            src_signal = src_signal / max_value
            return mixed_signal, src_signal

        return mixed_signal, src_signal

    def _random_rt60(self):
        RT60 = random.gauss(0.1, 0.05)
        while RT60 < 0.05 or RT60 > 0.15:
            RT60 = random.gauss(0.1, 0.05)

        fix_RT60 = random.uniform(0.08, 0.12)

        return RT60, fix_RT60

    def _random_cabin(
        self, mic_topo, ret_itf_pos=True, target_pos=None, itf_pos=None
    ):
        """_random_cabin.

        Tips:
            进行数据仿真时, 没有在pyroomacoustics 和 gpuRIR 的文档中找到关于坐标系的介绍.
            经过分析认为不需要纠结左手系还是右手系，应该关注仿真时麦克风的含义、位置和
            声源跟麦克风的相对位置，能在假设的坐标系下匹配.

            举例来说，下面两个不同坐标系下的两组参数含义相同:
                1. 左手系下: mic_pos: [(0.545, 0.6, 1.099), (0.655, 0.6, 1.099)],
                    src_pos: (0.4, 1.2, 0.7).
                2. 右手系下: mic_pos: [(0.655, 0.6, 1.099), (0.545, 0.6, 1.099)],
                    src_pos: (0.8, 1.2, 0.7).
        """
        # for CD569
        #              width, length, height
        min_cab_size = [1.2, 2.6, 1.1]
        max_cab_size = [1.4, 3.5, 1.3]
        cab_size = [
            random.uniform(s1, s2)
            for s1, s2 in zip(min_cab_size, max_cab_size)
        ]
        mic_pos = [cab_size[0] / 2, 0.6, cab_size[2] - 0.001]
        accurate_mic_pos = np.array(mic_topo) + np.array(mic_pos)

        # 计算模拟乘客位置
        min_height = 0.5
        offset = 0.1
        # 主驾位置:
        # width: mic左侧到座舱边缘
        # length: mic位置到座舱深度1/2
        # height: 最小高度到mic高度
        driver_pos = [
            random.uniform(offset, mic_pos[0] - offset),
            random.uniform(mic_pos[1] + offset, cab_size[1] / 2 - offset),
            random.uniform(min_height, mic_pos[2] - offset),
        ]
        # 副驾位置
        # width: mic右侧到座舱边缘
        # length: mic位置到座舱深度1/2
        # height: 最小高度到mic高度
        copilot_pos = [
            random.uniform(mic_pos[0] + offset, cab_size[0] - offset),
            random.uniform(mic_pos[1] + offset, cab_size[1] / 2 - offset),
            random.uniform(min_height, mic_pos[2] - offset),
        ]
        # 后排左侧乘客
        # width: mic左侧到座舱边缘
        # length: 座舱深度1/2到座舱边缘
        # height: 最小高度到mic高度
        rear_left_passenger_pos = [
            random.uniform(offset, mic_pos[0] - offset),
            random.uniform(cab_size[1] / 2 + offset, cab_size[1] - offset),
            random.uniform(min_height, mic_pos[2] - offset),
        ]
        # 后排左侧乘客
        # width: mic右侧到座舱边缘
        # length: 座舱深度1/2到座舱边缘
        # height: 最小高度到mic高度
        rear_right_passenger_pos = [
            random.uniform(mic_pos[0] + offset, cab_size[0] - offset),
            random.uniform(cab_size[1] / 2 + offset, cab_size[1] - offset),
            random.uniform(min_height, mic_pos[2] - offset),
        ]
        seat_dict = {
            "pilot": driver_pos,
            "copilot": copilot_pos,
            "rear_left_passenger": rear_left_passenger_pos,
            "rear_right_passenger_pos": rear_right_passenger_pos,
        }
        # 先筛选干扰信号的位置
        if ret_itf_pos:
            if itf_pos is not None:
                itf_seat = seat_dict[itf_pos]
            else:
                itf_pos, itf_seat = random.sample(seat_dict.items(), 1)[0]
                del seat_dict[itf_pos]
        else:
            itf_seat = None
        # 获取目标信号的位置
        if target_pos is not None:
            target_seat = seat_dict[target_pos]
        else:
            target_pos, target_seat = random.sample(seat_dict.items(), 1)[0]

        fixed_driver_seat = [
            0.5 * (cab_size[0] - offset - mic_pos[0]) + mic_pos[0],
            0.5 * (cab_size[1] / 2 - mic_pos[1]) + mic_pos[1],
            0.5 * (mic_pos[2] - min_height) + min_height,
        ]
        fixed_front_passenger_seat = [
            -0.5 * (cab_size[0] - offset - mic_pos[0]) + mic_pos[0],
            0.5 * (cab_size[1] / 2 - mic_pos[1]) + mic_pos[1],
            0.5 * (mic_pos[2] - min_height) + min_height,
        ]

        noise_pos = [
            random.uniform(offset, cab_size[0] - offset),
            random.uniform(offset, cab_size[1] - offset),
            random.uniform(offset, cab_size[2] - offset),
        ]
        return (
            cab_size,
            mic_pos,
            accurate_mic_pos,
            (target_seat, itf_seat),
            (fixed_driver_seat, fixed_front_passenger_seat),
            noise_pos,
        )

    def _calc_steer_vector(
        self, mic_pos, mic_topo, target_seat, simulate_kwargs=None
    ):
        # calculate doa
        doa = signal_utils.calculate_doa(mic_pos, target_seat)
        if simulate_kwargs is not None and simulate_kwargs.doa_disturb.active:
            random_int = random.randint(0, 1)
            disturb_angle = random.uniform(
                simulate_kwargs.doa_disturb.disturb_angle["lowerbound"],
                simulate_kwargs.doa_disturb.disturb_angle["upperbound"],
            )
            if random_int == 0:
                doa += disturb_angle
            elif random_int == 1:
                doa -= disturb_angle
            else:
                raise NotImplementedError
        # TODO 根据mic的拓扑结构计算
        mic_topo = np.array(mic_topo)
        mic_topo = (mic_topo - mic_topo[0])[:, 0]
        steer_vector = signal_utils.calculate_steer_vector(mic_topo, doa)
        return steer_vector

    def add_noise_in_cabin(
        self,
        target_signal,
        interferer_signal=None,
        itf_snr=0.0,
        noise_signal=None,
        noise_snr=0.0,
        mode="train",
        simulate_kwargs=None,
        target_pos=None,
        itf_pos=None,
        mic_topo=[[-0.055, 0, 0], [0.055, 0, 0]],  # noqa: B006
    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        """add_noise_in_cabin.

        添加座舱中的噪声信号.

        Args:
            target_signal: np.ndarray, mxnet.nd.ndarray. 目标信号.
            interferer_signal: np.ndarray, mxnet.nd.ndarray. 干扰信号.
            itf_snr: float, 可选. 目标信号比干扰信号的信噪比, 默认值0.0.
            noise_signal: np.ndarray, mxnet.nd.ndarray, 可选.
                噪声信号, 默认值 None.
            noise_snr: float, 可选.
                混合信号比噪声的信噪比, 默认值0.0
            mode: str, 可选.
                数据处理模式, 默认值 "train".
            mic_topo: List[Tuple[int, int, int]].
                麦克风位置拓扑列表，列表的每一个元素是 (x, y, z) 相对麦克风位置的坐标.

        Returns:
            Tuple[ np.NDArray, List[np.NDArray], np.NDArray].
            混合后的信号.
            混合过程中产生的仿真信号.
            导向向量.
        """

        # v0.4.2 will cause all nan in generated mix signal
        assert pra.__version__ != "0.4.2"
        # 如果 mode != 'train' 使用固定的随机种子，每次执行的结果将相同
        if mode == "train":
            random.seed()
        else:
            random.seed(777)
        # 信号转换为 np.ndarray
        if isinstance(target_signal, torch.Tensor):
            target_signal = target_signal.numpy()
        assert (
            len(target_signal.shape) == 1 or target_signal.shape[1] == 1
        ), "simulate Noise in Cabin should be 1 channel."
        if len(target_signal.shape) > 1:
            target_signal = target_signal.reshape(-1)
        # 处理干扰信号的shape
        if interferer_signal is not None:
            if isinstance(interferer_signal, torch.Tensor):
                interferer_signal = interferer_signal.numpy()
            assert (
                len(interferer_signal.shape) == 1
                or interferer_signal.shape[1] == 1
            ), "simulate Noise in Cabin should be 1 channel."
            if len(interferer_signal.shape) > 1:
                interferer_signal = interferer_signal[:, 0].reshape(-1)
        # 处理噪声信号
        if noise_signal is not None:
            if isinstance(noise_signal, torch.Tensor):
                noise_signal = noise_signal.numpy()
            if len(noise_signal.shape) > 1:
                noise_signal = noise_signal[:, 0].reshape(-1)

        if simulate_kwargs is None or not simulate_kwargs.gpu_accelerate:
            return self._add_noise_in_cabin_cpu(
                target_signal,
                interferer_signal,
                itf_snr=itf_snr,
                noise_signal=noise_signal,
                noise_snr=noise_snr,
                mode=mode,
                target_pos=target_pos,
                itf_pos=itf_pos,
                mic_topo=mic_topo,
            )
        else:
            return self._add_noise_in_cabin_gpu(
                target_signal,
                interferer_signal,
                itf_snr,
                noise_signal,
                noise_snr,
                mode,
                simulate_kwargs,
                target_pos,
                itf_pos,
                mic_topo=mic_topo,
            )

    def _add_noise_in_cabin_cpu(
        self,
        target_signal,
        interferer_signal,
        itf_snr=0.0,
        noise_signal=None,
        noise_snr=20.0,
        mode="train",
        target_pos=None,
        itf_pos=None,
        mic_topo=[[-0.055, 0, 0], [0.055, 0, 0]],  # noqa: B006
    ):
        # V0.4.2 will cause all nan in generated mix signal
        assert pra.__version__ != "0.4.2"
        ret_itf_pos = interferer_signal is not None
        RT60, fix_RT60 = self._random_rt60()
        (
            cab_size,
            mic_pos,
            accurate_mic_pos,
            (target_seat, itf_seat),
            (fixed_target_seat, fixed_itf_seat),
            noise_seat,
        ) = self._random_cabin(mic_topo, ret_itf_pos, target_pos, itf_pos)
        try:
            e_absorption, max_order = pra.inverse_sabine(RT60, cab_size)
        except BaseException:
            e_absorption, max_order = pra.inverse_sabine(fix_RT60, cab_size)

        logging.debug(f"ShoeBox Size: {cab_size}")
        shoebox = pra.ShoeBox(
            cab_size,
            fs=16000,
            materials=pra.Material(e_absorption),
            max_order=max_order,
        )
        shoebox.add_microphone_array(np.c_[accurate_mic_pos.T])
        # 添加目标信号
        try:
            shoebox.add_source(target_seat, delay=0, signal=target_signal)
            logging.debug(f"Target Seat: {target_seat}")
        except BaseException:
            target_seat = fixed_target_seat
            shoebox.add_source(target_seat, delay=0, signal=target_signal)
            logging.debug(f"Target Seat: {target_seat}")
        # 添加干扰人声
        if interferer_signal is not None:
            try:
                shoebox.add_source(itf_seat, delay=0, signal=target_signal)
            except Exception:
                shoebox.add_source(
                    fixed_itf_seat, delay=0, signal=interferer_signal
                )
        # 添加非人声噪音
        if noise_signal is not None:
            try:
                shoebox.add_source(noise_seat, delay=0, signal=noise_signal)
                logging.debug(f"Noise Seat: {noise_seat}")
            except BaseException:
                pass
        # 配置 mixed_singals 函数所需的系数
        n_src = 1
        n_itf = 0
        if interferer_signal is not None:
            n_itf = 1
            n_src += 1
        if noise_signal is not None:
            n_src += 1
        mix_kwargs = {
            "snr": noise_snr,  # SNR target in decibels
            "sir": itf_snr,  # SIR target in decibels
            "n_src": n_src,
            "n_tgt": 1,
            "n_itf": n_itf,
            "ref_mic": 0,
        }
        logging.debug(f"Mix Kwargs: {mix_kwargs}")
        # 返回混合前的信号, 经过仿真的信号
        premix_signals = shoebox.simulate(
            callback_mix=signal_utils.mix_signals,
            callback_mix_kwargs=mix_kwargs,
            return_premix=True,
        )
        # 信号值归一化并抽取仿真的 target_singal, interferer_signal[, noise_signal]
        premix_signals_max_value = np.max(np.abs(premix_signals) + 1e-20)
        premix_signals = premix_signals / premix_signals_max_value
        premix_signals = [
            sig[0, : len(target_signal)] for sig in premix_signals
        ]
        # 获得混合后的信号，并将混合后的信号维度进行调整
        # (n_mics, n_samples) -> (n_samples, n_mics)
        mixed_signal = np.transpose(shoebox.mic_array.signals, axes=(1, 0))
        mixed_signal_max_value = np.max(np.abs(mixed_signal) + 1e-20)
        mixed_signal = (
            mixed_signal[: len(target_signal), :] / mixed_signal_max_value
        )

        # 根据线阵的几何结构计算延迟和波束形成器的转向矢量
        steer_vector = self._calc_steer_vector(mic_pos, mic_topo, target_seat)
        # 返回混合后的信号/仿真信号/导向向量
        return mixed_signal, premix_signals, steer_vector

    def _add_noise_in_cabin_gpu(
        self,
        target_signal,
        interferer_signal,
        itf_snr=0.0,
        noise_signal=None,
        noise_snr=0.0,
        mode="train",
        simulate_kwargs={},  # noqa: B006
        target_pos=None,
        itf_pos=None,
        mic_topo=[[-0.055, 0, 0], [0.055, 0, 0]],  # noqa: B006
    ):
        if self._local_rir:  # 查表获取rir
            group = "gpu_rir"
            head_key = f"gpu_rir_{target_pos}"
            idx = random.randint(0, 999)
            ref_key = f"{head_key}_{idx :>0{6}d}_ref"
            RIRs_ref = self.get_rirs_from_hdf5(group, ref_key)
            # 仿真成双通道
            stereo_ref = self.simulateTrajectory(target_signal, RIRs_ref)
            # 人声干扰信号
            stereo_itf = None
            if interferer_signal is not None:
                pass
            # 非人声干扰噪声
            stereo_noise = None
            if noise_signal is not None:
                noise_key = f"{head_key}_{idx :>0{6}d}_noise"
                RIRs_noise = self.get_rirs_from_hdf5(group, noise_key)
                stereo_noise = self.simulateTrajectory(
                    target_signal, RIRs_noise
                )
        else:
            if not hasattr(self, "simulator"):
                self.simulator = RIRSimulator(mixed_precision=False, lut=False)
            ret_itf_pos = interferer_signal is not None
            RT60, fixed_RT60 = self._random_rt60()
            (
                cab_size,
                mic_pos,
                accurate_mic_pos,
                (target_seat, itf_seat),
                (fixed_target_seat, fixed_itf_seat),
                noise_seat,
            ) = self._random_cabin(mic_topo, ret_itf_pos, target_pos, itf_pos)

            try:
                reflect_coeffs = self.simulator.beta_SabineEstimation(
                    cab_size, RT60
                )
                max_order = self.simulator.t2n(RT60, cab_size)

            except BaseException:
                reflect_coeffs = self.simulator.beta_SabineEstimation(
                    cab_size, fixed_RT60
                )
                max_order = self.simulator.t2n(fixed_RT60, cab_size)

            pos_ref = np.array([target_seat])
            RIRs_ref = self.simulator.simulateRIR(
                cab_size,
                reflect_coeffs,
                pos_src=pos_ref,
                pos_rcv=accurate_mic_pos,
                nb_img=max_order,
                Tmax=RT60,
                fs=16000,
            )
            # 仿真成双通道
            stereo_ref = self.simulator.simulateTrajectory(
                target_signal, RIRs_ref
            )
            # 人声干扰噪声
            stereo_itf = None
            if interferer_signal is not None:
                pos_itf = np.array([itf_seat])
                RIRs_itf = self.simulator.simulateRIR(
                    cab_size,
                    reflect_coeffs,
                    pos_src=pos_itf,
                    pos_rcv=accurate_mic_pos,
                    nb_img=max_order,
                    Tmax=RT60,
                    fs=16000,
                )
                stereo_itf = self.simulator.simulateTrajectory(
                    interferer_signal, RIRs_itf
                )

            # 非人声干扰噪声
            stereo_noise = None
            if noise_signal is not None:
                if simulate_kwargs.homogeneous_noise:
                    stereo_noise = signal_utils.multichannel_homogeneous_noise(
                        noise_signal
                    )
                else:
                    try:
                        noise_seat = np.array([noise_seat])
                        RIRs_noise = self.simulator.simulateRIR(
                            cab_size,
                            reflect_coeffs,
                            pos_src=noise_seat,
                            pos_rcv=accurate_mic_pos,
                            nb_img=max_order,
                            Tmax=RT60,
                            fs=16000,
                        )
                        stereo_noise = self.simulator.simulateTrajectory(
                            noise_signal, RIRs_noise
                        )
                    except BaseException:
                        logging.error(f'{"GPURIR noise synthesizing failed!"}')
                        stereo_noise = None

        premix_signals_stereo = [
            s for s in (stereo_ref, stereo_itf, stereo_noise) if s is not None
        ]
        premix_signals = [
            sig[: len(target_signal)].transpose()
            for sig in premix_signals_stereo
        ]
        if len(premix_signals) == 1:
            mixed_signal = premix_signals[0].transpose()
        else:
            premix_signals = np.stack(premix_signals, axis=0)
            # 指定 mix_signals 函数所需的系数
            premix_signals_label_noise = np.zeros_like(premix_signals)
            premix_signals_label_noise[:] = premix_signals[:]
            n_src = 1
            n_itf = 0
            if stereo_itf is not None:
                n_src += 1
                n_itf += 1
            if noise_signal is not None:
                n_src += 1
            mix_kwargs = {
                "snr": noise_snr,  # SNR target in decibels
                "sir": itf_snr,  # SIR target in decibels
                "n_src": n_src,
                "n_tgt": 1,
                "n_itf": n_itf,
                "ref_mic": 0,
            }
            mixed_signal = signal_utils.mix_signals(
                premix_signals, **mix_kwargs
            ).transpose()

        if self._normalize_signal:
            mean_value = np.mean(mixed_signal)
            mixed_signal = mixed_signal - mean_value
            max_value = np.max(np.abs(mixed_signal) + 1e-20)
            mixed_signal = mixed_signal[: len(target_signal), :] / max_value

        if simulate_kwargs.get("noisy_ref", False):
            mix_kwargs = {
                "snr": simulate_kwargs.ref_noise_snr,
                "sir": simulate_kwargs.ref_itf_snr,
                "n_src": 3,
                "n_tgt": 1,
                "n_itf": 1,
                "ref_mic": 0,
            }
            ref_signal = signal_utils.mix_signals(
                premix_signals_label_noise, **mix_kwargs
            ).transpose()

        steer_vector = None
        # steer_vector = self._calc_steer_vector(mic_pos, mic_topo, target_seat) # noqa: E501

        def _func(sig):
            sig = sig[: len(target_signal), 0]
            sig = sig / (sig.std() + EPS)
            return sig

        if simulate_kwargs.get("noisy_ref", False):
            premix_signals = []
            ref_signal = _func(ref_signal)
            premix_signals.append(ref_signal)
            for sig in premix_signals_stereo[1:]:
                premix_signals.append(_func(sig))
        else:
            # 信号值归一化并抽取仿真的 target_singal, interferer_signal[, noise_signal]
            premix_signals = [_func(sig) for sig in premix_signals_stereo]
        # 返回混合后的信号/仿真信号/导向向量
        return mixed_signal, premix_signals, steer_vector

    def get_rirs_from_hdf5(self, group, key):
        attrs = self.gpu_rir_reader[group][key].attrs
        beg_idx = attrs["beg_idx"]
        end_idx = attrs["end_idx"]
        length = attrs["length"]
        RIRs = self.gpu_rir_reader[group][key][beg_idx:end_idx]
        RIRs = [rir.reshape((1, 2, -1)) for rir in RIRs]
        RIRs = np.concatenate(RIRs, axis=2)
        RIRs = RIRs[:, :, :length]
        return RIRs

    def simulateTrajectory(
        self, source_signal, RIRs, timestamps=None, fs=None
    ):
        assert len(source_signal.shape) == 1
        assert len(RIRs.shape) == 3
        source_signal = source_signal.reshape(-1, 1)
        rirs_ref = RIRs[0, :, :].transpose()
        max_value = np.max(np.abs(rirs_ref) + 1e-20)
        rirs_ref = rirs_ref / max_value
        convolution = ss.convolve(
            rirs_ref[:, None, :], source_signal[:, :, None]
        )
        convolution = convolution.transpose(1, 2, 0)

        nSamples = len(source_signal)
        nPts = RIRs.shape[0]
        nRcv = RIRs.shape[1]
        lenRIR = RIRs.shape[2]

        assert (
            timestamps is None or fs is not None
        ), "fs must be indicated for custom timestamps"
        assert (
            timestamps is None or timestamps[0] == 0
        ), "The first timestamp must be 0"
        if timestamps is None:
            fs = nSamples / nPts
            timestamps = np.arange(nPts)

        w_ini = np.append((timestamps * fs).astype(int), nSamples)
        w_len = np.diff(w_ini)

        filtered_signal = np.zeros((nSamples + lenRIR - 1, nRcv))
        for m in range(nRcv):
            for n in range(nPts):
                filtered_signal[
                    w_ini[n] : w_ini[n + 1] + lenRIR - 1, m
                ] += convolution[n, m, 0 : w_len[n] + lenRIR - 1]

        return filtered_signal

    def __getstate__(self):
        state = self.__dict__.copy()
        if self._local_rir:
            del state["gpu_rir_reader"]
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._fork()


class NoiseSampler(object):
    """Noise Sampler.

    初始化噪声信息, 获取加噪信噪比和噪声ID.

    Args:
        noise_info: 噪声信息.
        noise_distribution: 噪声分布信息.
        snr_gen_kwargs: 信噪比信息.
        mode: 模式, 默认为"train".
    """

    def __init__(
        self,
        noise_info: dict,
        noise_distribution: edict,
        snr_gen_kwargs: edict,
        mode: str = "train",
    ):
        if mode == "train":
            random.seed()
        else:
            random.seed(777)
        self._noise_info = noise_info
        self.check_noise_args_validity(noise_info, noise_distribution)
        self.noise_cdf, self.noise_type_list = self.build_noise_cdf(
            noise_distribution
        )
        self._snr_gen_kwargs = snr_gen_kwargs

    def check_noise_args_validity(self, noise_info, noise_distribution):
        accu = 0.0
        for key in noise_distribution.keys():
            if key == "mixture":
                for mkey in noise_distribution[key].keys():
                    accu += noise_distribution[key][mkey]
            else:
                accu += noise_distribution[key]
        assert abs(accu - 1.0) < 0.0001

    def build_noise_cdf(self, noise_distribution):
        accu = 0.0
        noise_cdf = []
        noise_type_list = []
        for key in noise_distribution.keys():
            if key == "mixture":
                for mkey in noise_distribution[key].keys():
                    noise_type_list.append(mkey)
                    accu += noise_distribution[key][mkey]
                    noise_cdf.append(accu)
            else:
                noise_type_list.append(key)
                accu += noise_distribution[key]
                noise_cdf.append(accu)
        return noise_cdf, noise_type_list

    def get_snr(self, snr_gen_kwargs):
        if snr_gen_kwargs["distribution_type"] == "uniform":
            return random.uniform(
                snr_gen_kwargs["lowerbound"], snr_gen_kwargs["upperbound"]
            )
        elif snr_gen_kwargs["distribution_type"] == "gaussian":
            return random.uniform(
                snr_gen_kwargs["mean"], snr_gen_kwargs["stddev"]
            )
        else:
            raise ValueError(
                "SNR generation distribution name "
                "{} is not supported yet".format(
                    snr_gen_kwargs["distribution_name"]
                )
            )

    def get_noise_info(self):
        rnd = random.uniform(0, 1)
        noise_type = None
        for i, cdfv in enumerate(self.noise_cdf):
            if rnd <= cdfv:
                noise_type = self.noise_type_list[i]
                break

        sample_snr = self.get_snr(self._snr_gen_kwargs)

        return noise_type, sample_snr

    def get_noise_id(self, noise_type):
        assert noise_type in self._noise_info
        return random.choice(self._noise_info[noise_type])


class AudioNoiseRetriever(object):
    """AudioNoiseRetriever.

    初始化噪声读取器, 获取噪声数据.

    Args:
        hdf5_file: 噪声hdf5文件.
        ret_dtype: 读取噪声的数据类型.
        rec_content_type: 噪声文件的类型.
        rec_info: rec噪声文件路径.
        noise_lst_groups: rec噪声分组信息.
        noise_lst_prefix: rec噪声路径前缀.
        noise_info: hdf5噪声信息文件.
        noise_distribution: 噪声分布信息.
        signal_to_noise_snr_distribution: 加噪信噪比分布.
        noise_mixture_snr_kwargs: 噪声混合信噪比参数.
        mode: 模式, 默认为"train".
        noise_simulate: 设置为True, 则对噪声做仿真, 默认为False.
        gpu_rir_path: 缓存的rir信号文件路径.
        local_rir: 设置为True, 使用本地缓存的rir信号, 默认为False.
    """

    def __init__(
        self,
        hdf5_file: str,
        ret_dtype: str,
        rec_content_type: str,
        rec_info: edict,
        noise_lst_groups: str,
        noise_lst_prefix: str,
        noise_info: str,
        noise_distribution: edict,
        signal_to_noise_snr_distribution: edict,
        noise_mixture_snr_kwargs: edict,
        mode: str = "train",
        noise_simulate: bool = False,
        gpu_rir_path: str = None,
        local_rir: bool = False,
    ):
        self._rec_content_type = rec_content_type
        self._mixture_snr_kwargs = noise_mixture_snr_kwargs

        # 不同取数据方式的初始化
        if self._rec_content_type == "raw":
            self.initialize_noise_rec(rec_info)
            self._noise_type_info = self.initialize_noise_type_info(
                noise_lst_groups, noise_lst_prefix
            )
        elif self._rec_content_type == "hdf5":
            self._noise_type_info = self.initialize_noise_info(noise_info)
            self.noise_reader = HDF5WaveformReader(
                hdf5_file=hdf5_file,
                ret_dtype=ret_dtype,
                channel_first=False,
            )

        self.sampler = NoiseSampler(
            self._noise_type_info,
            noise_distribution,
            signal_to_noise_snr_distribution,
            mode,
        )
        self.audio_noise_augmentor = SignalAugmentor(
            normalize_signal=False,
            gpu_rir_path=gpu_rir_path,
            local_rir=local_rir,
        )
        self.noise_simulate = noise_simulate

    def initialize_noise_type_info(self, noise_lst_groups, noise_lst_prefix):
        with open(noise_lst_groups, "r", encoding="utf-8") as fc:
            noise_type_to_lst = edict(yaml.full_load(fc))
        noise_type_info = {}
        for noise_type, noise_lst in noise_type_to_lst.items():
            noise_sample_list = []
            for noise_sample in noise_lst:
                with open(
                    join(noise_lst_prefix, noise_sample), encoding="utf-8"
                ) as fh:
                    for line in fh:
                        noise_sample_list.append(
                            os.path.basename(line.rstrip()).replace(".wav", "")
                        )
            noise_type_info[noise_type] = noise_sample_list
        return noise_type_info

    def initialize_noise_info(self, noise_info):
        with open(noise_info, "r", encoding="utf-8") as fr:
            noise_infos = fr.readlines()
        noise_type_info = {}
        for info in noise_infos:
            info = info.rstrip("\r\t\n ").split()
            if len(info) != 2:
                continue
            noise_type = info[0]
            noise_id = info[1]
            noise_type_info.setdefault(noise_type, []).append(noise_id)
        return noise_type_info

    def initialize_noise_rec(self, rec_info):
        self.noise_record = MXRecord(
            uri=rec_info["rec_path"],
            idx_path=rec_info["idx_path"],
            writable=False,
            key_type=str,
        )

    def get_noise_mixture_snr(self):
        if self._mixture_snr_kwargs.distribution_type == "uniform":
            return random.uniform(
                self._mixture_snr_kwargs.lowerbound,
                self._mixture_snr_kwargs.upperbound,
            )
        else:
            raise NotImplementedError

    @require_packages("soundfile", "mxnet")
    def get_noise(self, target_length):
        noise_type, add_noise_snr = self.sampler.get_noise_info()
        current_noise_signal = None

        noise_signal_list = []
        if noise_type in self._noise_type_info:
            noise_type_list = [noise_type]
        else:
            noise_type_list = noise_type.split("-")
            assert len(noise_signal_list) <= 2
        for noise_type_i in noise_type_list:
            current_length = 0
            while current_length < target_length:
                noise_id = self.sampler.get_noise_id(noise_type_i)
                if self._rec_content_type == "nd":
                    sna = self.noise_record.read(noise_id)
                    noise_audio_signal = mx.nd.load_from_str(sna).asnumpy()
                elif self._rec_content_type == "raw":
                    # support wav, flac and other formats
                    sna = self.noise_record.read(noise_id)
                    noise_audio_signal, samplerate = sf.read(io.BytesIO(sna))
                elif self._rec_content_type == "hdf5":
                    need_length = target_length - current_length
                    attrs = self.noise_reader.read_attrs(noise_id)
                    duration = attrs["duration"]
                    samplerate = attrs["samplerate"]
                    noise_length = int(duration * samplerate)
                    if noise_length > need_length:
                        beg_ind = random.randint(0, noise_length - need_length)
                        end_ind = beg_ind + need_length
                    else:
                        beg_ind = 0
                        end_ind = noise_length
                    noise_audio_signal, _ = self.noise_reader.read_seg_audio(
                        noise_id,
                        beg_ind / (samplerate * 1.0),
                        end_ind / (samplerate * 1.0),
                    )
                else:
                    raise ValueError(
                        "The rec content record should be "
                        "selected from ['nd', 'raw', 'hdf5'] but \
                        the input is [{rec_content_type}]."
                        "Please check config.".format(
                            rec_content_type=self._rec_content_type
                        )
                    )
                if "record_noise" in noise_type_i:
                    noise_audio_signal = noise_audio_signal[:, [0, 1]]
                elif len(noise_audio_signal.shape) == 2:
                    nchannel = noise_audio_signal.shape[1]
                    sel_channel = [random.randint(0, nchannel - 1)]
                    noise_audio_signal = noise_audio_signal[:, sel_channel]

                if np.mean(noise_audio_signal ** 2) == 0:
                    continue

                if self.noise_simulate and len(noise_audio_signal.shape) == 1:
                    (
                        noise_audio_signal,
                        _,
                        _,
                    ) = self.audio_noise_augmentor.add_noise_in_cabin(
                        noise_audio_signal, np.zeros(noise_audio_signal.shape)
                    )
                    noise_audio_signal = (
                        noise_audio_signal / abs(noise_audio_signal).max()
                    )
                if current_noise_signal is None:
                    current_noise_signal = noise_audio_signal
                else:

                    noise_audio_signal = (
                        signal_utils.match_signal_to_target_signal_power(
                            noise_audio_signal, current_noise_signal
                        )
                    )

                    current_noise_signal = np.concatenate(
                        [current_noise_signal, noise_audio_signal], axis=0
                    )
                current_length = current_noise_signal.shape[0]
            noise_signal_list.append(
                signal_utils.randomly_cut_signal_to_target_length(
                    current_noise_signal, target_length
                )
                if current_length > target_length
                else current_noise_signal
            )

        if len(noise_signal_list) == 1:
            return noise_signal_list[0], add_noise_snr
        elif len(noise_signal_list) == 2:
            mixture_snr = self.get_noise_mixture_snr()
            (
                mixture_noise_signal,
                _,
            ) = self.audio_noise_augmentor.add_noise_linearly(
                noise_signal_list[0], noise_signal_list[1], mixture_snr
            )
            return mixture_noise_signal, add_noise_snr
        else:
            raise NotImplementedError


class RIRSimulator(object):
    """RIR Simulator.

    重构gpuRIR, 推荐使用RIRSimulator而不是直接使用gpuRIR.
    """

    polar_patterns = {
        "omni": 0,
        "homni": 1,
        "card": 2,
        "hypcard": 3,
        "subcard": 4,
        "bidir": 5,
    }

    @require_packages("gpuRIR_bind")
    def __init__(self, mixed_precision=False, lut=False):
        self.gpuRIR_bind = gpuRIR_bind()
        self.activateMixedPrecision(mixed_precision)
        self.activateLUT(lut)

    def beta_SabineEstimation(self, room_sz, T60, abs_weights=[1.0] * 6):
        """
        Estimation of the reflection coefficients needed to have the \
        desired reverberation time.

        Args:
            room_sz: 3 elements list or numpy array,
                size of the room (in meters).
            T60: Reverberation time of the room,
                seconds to reach 60dB attenuation.
            abs_weights: array_like with 6 elements, optional.
                Absoprtion coefficient ratios of the walls,
                the default is [1.0]*6.

        Returns:
            ndarray with 6 elements.
            Reflection coefficients of the walls as $[beta_{x0},
                beta_{x1}, beta_{y0}, beta_{y1}, beta_{z0}, beta_{z1}]$,
                where $beta_{x0}$ is the coeffcient of the wall parallel
                to the x axis closest to the origin of coordinates system
                and $beta_{x1}$ the farthest.
        """

        def t60error(x, T60, room_sz, abs_weights):
            alpha = x * abs_weights
            Sa = (
                (alpha[0] + alpha[1]) * room_sz[1] * room_sz[2]
                + (alpha[2] + alpha[3]) * room_sz[0] * room_sz[2]
                + (alpha[4] + alpha[5]) * room_sz[0] * room_sz[1]
            )
            V = np.prod(room_sz)
            if Sa == 0:
                return T60 - 0  # Anechoic chamber
            return abs(T60 - 0.161 * V / Sa)  # Sabine's formula

        abs_weights /= np.array(abs_weights).max()
        result = minimize(
            t60error, 0.5, args=(T60, room_sz, abs_weights), bounds=[[0, 1]]
        )
        return np.sqrt(1 - result.x * abs_weights).astype(np.float32)

    def att2t_SabineEstimator(self, att_dB, T60):
        """
        Estimation of the time for the RIR to reach a certain \
        attenuation using the Sabine model.

        Args:
            att_dB: float. Desired attenuation (in dB).
            T60: float. Reverberation time of the room,
                seconds to reach 60dB attenuation.

        Returns:
            float. Time (in seconds) to reach the desired attenuation.
        """

        return att_dB / 60.0 * T60

    def t2n(self, T, rooms_sz, c=343.0):
        """
        Estimation of the number of images needed for a correct RIR simulation.

        Args:
            T: float. RIRs length (in seconds).
            room_sz: 3 elements list or numpy array,
                Size of the room (in meters).
            c: float, optional. Speed of sound (the default is 343.0).

        Returns:
            3 elements list of integers.
            The number of images sources to compute in each dimension.
        """

        nb_img = 2 * T / (np.array(rooms_sz) / c)
        return [int(n) for n in np.ceil(nb_img)]

    def simulateRIR(
        self,
        room_sz,
        beta,
        pos_src,
        pos_rcv,
        nb_img,
        Tmax,
        fs,
        Tdiff=None,
        spkr_pattern="omni",
        mic_pattern="omni",
        orV_src=None,
        orV_rcv=None,
        c=343.0,
    ):
        """
        Room Impulse Responses (RIRs) simulation using the \
        Image Source Method (ISM).

        Args:
            room_sz: array_like with 3 elements.
                Size of the room (in meters).
            beta: array_like with 6 elements.
                Reflection coefficients of the walls as $[beta_{x0},
                    beta_{x1}, beta_{y0}, beta_{y1}, beta_{z0}, beta_{z1}]$,
                    where $beta_{x0}$ and $beta_{x1}$ are the reflection
                    coefficents of the walls orthogonal to the x axis at
                    x=0 and x=room_sz[0], respectively.
            pos_src, pos_rcv: ndarray with 2 dimensions and 3 columns.
                Position of the sources and the receivers (in meters).
            nb_img: array_like with 3 integer elements.
                Number of images to simulate in each dimension.
            Tmax: float. RIRs length (in seconds).
            fs: float. RIRs sampling frequency (in Hertz).
            Tdiff: float, optional.
                Time (in seconds) when the ISM is replaced by a diffuse
                    reverberation model.
                Default is Tmax (full ISM simulation).
            spkr_pattern:
                {"omni", "homni", "card", "hypcard", "subcard", "bidir"},
                    optional.
                Polar pattern of the sources (the same for all of them).
                    "omni": Omnidireccional (default).
                    "homni": Half omnidirectional, 1 in front of the
                        microphone, 0 backwards.
                    "card": Cardioid.
                    "hypcard": Hypercardioid.
                    "subcard": Subcardioid.
                    "bidir": Bidirectional, a.k.a. figure 8.
            mic_pattern:
                {"omni", "homni", "card", "hypcard", "subcard", "bidir"},
                    optional.
                Polar pattern of the receivers (the same for all of them).
                    "omni": Omnidireccional (default).
                    "homni": Half omnidirectional, 1 in front of the
                        microphone, 0 backwards.
                    "card": Cardioid.
                    "hypcard": Hypercardioid.
                    "subcard": Subcardioid.
                    "bidir": Bidirectional, a.k.a. figure 8.
            orV_src: ndarray with 2 dimensions and 3 columns or None, optional.
                Orientation of the sources as vectors pointing
                    in the same direction.
                None (default) is only valid for omnidirectional patterns.
            orV_rcv: ndarray with 2 dimensions and 3 columns or None, optional.
                Orientation of the receivers as vectors pointing
                    in the same direction.
                None (default) is only valid for omnidirectional patterns.
            c: float, optional.
                Speed of sound [m/s] (the default is 343.0).

        Returns:
            3D ndarray.
            The first axis is the source, the second the receiver and
                the third the time.

        Warnings:
            Asking for too much and too long RIRs
                (specially for full ISM simulations) may exceed
                the GPU memory and crash the kernel.
        """

        assert not (
            (pos_src >= room_sz).any() or (pos_src <= 0).any()
        ), "The sources must be inside the room"
        assert not (
            (pos_rcv >= room_sz).any() or (pos_rcv <= 0).any()
        ), "The receivers must be inside the room"
        assert (
            Tdiff is None or Tdiff <= Tmax
        ), "Tmax must be equal or greater than Tdiff"
        assert (
            mic_pattern in self.polar_patterns
        ), "mic_pattern must be omni, homni, card, hypcard, subcard or bidir"
        assert (
            mic_pattern == "omni" or orV_rcv is not None
        ), "the mics are not omni but their orientation is undefined"
        assert (
            spkr_pattern in spkr_pattern
        ), "spkr_pattern must be omni, homni, card, hypcard, subcard or bidir"
        assert (
            spkr_pattern == "omni" or orV_src is not None
        ), "the sources are not omni but their orientation is undefined"

        pos_src = pos_src.astype("float32", order="C", copy=False)
        pos_rcv = pos_rcv.astype("float32", order="C", copy=False)

        if Tdiff is None:
            Tdiff = Tmax
        if mic_pattern is None:
            mic_pattern = "omni"
        if spkr_pattern is None:
            spkr_pattern = "omni"
        if orV_rcv is None:
            orV_rcv = np.zeros_like(pos_rcv)
        else:
            orV_rcv = orV_rcv.astype("float32", order="C", copy=False)
        if orV_src is None:
            orV_src = np.zeros_like(pos_src)
        else:
            orV_src = orV_src.astype("float32", order="C", copy=False)

        return self.gpuRIR_bind.simulateRIR_bind(
            room_sz,
            beta,
            pos_src,
            pos_rcv,
            orV_src,
            orV_rcv,
            self.polar_patterns[spkr_pattern],
            self.polar_patterns[mic_pattern],
            nb_img,
            Tdiff,
            Tmax,
            fs,
            c,
        )

    def simulateTrajectory(
        self, source_signal, RIRs, timestamps=None, fs=None
    ):
        """
        Filter an audio signal by the RIRs of a motion trajectory \
        recorded with a microphone array.

        Args:
            source_signal: array_like. Signal of the moving source.
            RIRs: 3D ndarray.
                Room Impulse Responses generated with simulateRIR.
            timestamps: array_like, optional.
                Timestamp of each RIR [s]. By default, the RIRs are equispaced
                    through the trajectory.
            fs: float, optional.
                Sampling frequency (in Hertz). It is only needed for
                    custom timestamps.

        Returns:
            2D ndarray.
            Matrix with the signals captured by each microphone in each column.
        """

        nSamples = len(source_signal)
        nPts = RIRs.shape[0]
        nRcv = RIRs.shape[1]
        lenRIR = RIRs.shape[2]

        assert (
            timestamps is None or fs is not None
        ), "fs must be indicated for custom timestamps"
        assert (
            timestamps is None or timestamps[0] == 0
        ), "The first timestamp must be 0"
        if timestamps is None:
            fs = nSamples / nPts
            timestamps = np.arange(nPts)

        w_ini = np.append((timestamps * fs).astype(int), nSamples)
        w_len = np.diff(w_ini)
        segments = np.zeros((nPts, w_len.max()))
        for n in range(nPts):
            segments[n, 0 : w_len[n]] = source_signal[w_ini[n] : w_ini[n + 1]]
        segments = segments.astype("float32", order="C", copy=False)
        convolution = self.gpuRIR_bind.gpu_conv(segments, RIRs)

        filtered_signal = np.zeros((nSamples + lenRIR - 1, nRcv))
        for m in range(nRcv):
            for n in range(nPts):
                filtered_signal[
                    w_ini[n] : w_ini[n + 1] + lenRIR - 1, m
                ] += convolution[n, m, 0 : w_len[n] + lenRIR - 1]

        return filtered_signal

    def activateMixedPrecision(self, activate=True):
        """
        Activate the mixed precision mode, only for Pascal \
        GPU architecture or superior.

        Args:
            activate : bool, optional.
                True for activate and Flase for deactivate. True by default.
        """

        self.gpuRIR_bind.activate_mixed_precision_bind(activate)

    def activateLUT(self, activate=True):
        """Activate the lookup table for the sinc computations.

        Args:
            activate: bool, optional.
                True for activate and Flase for deactivate. True by default.
        """

        self.gpuRIR_bind.activate_lut_bind(activate)
