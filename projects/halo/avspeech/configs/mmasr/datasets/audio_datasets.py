# flake8: noqa

from typing import Iterable

import torchvision
import yaml
from common.default import tendis_kwargs
from easydict import EasyDict

from hat.data.datasets.avspeech.audio import (
    DirectWaveformReader,
    HDF5WaveformReader,
    TendisWaveformReader,
)
from hat.data.datasets.avspeech.basic_data_info import TendisBasicInfoReader
from hat.data.datasets.avspeech.info import MMASRInfoList, MMASRJsonInfoList
from hat.data.datasets.avspeech.processor import adapt_data_dict
from hat.data.datasets.avspeech_dataset import (
    CocktailDatasetV0,
    CocktailDatasetV2,
)
from hat.data.datasets.wenet_dataset import Dataset as make_wenet_dataset
from hat.data.datasets.wenet_dataset import Processor as WeNetProcessor
from hat.data.transforms.avspeech.audio import (
    ChannelSelect,
    FBank,
    ReSample,
    SpecAug,
)
from hat.data.transforms.avspeech.augment import (
    OnLineAugmentPipeline,
    OnLineJsonAugmentPipeline,
)
from hat.data.transforms.avspeech.denoise import PyHisfDenoiser
from hat.data.transforms.avspeech.text import LabelToTensor, Tokenize
from projects.halo.avspeech.utils.global_config import get_config


def get_wenet_train_dataset(config: EasyDict) -> Iterable:
    """

    Args:
        config (EasyDict):
            info: 数据集列表

    Returns:
        Iterable: 可以用做dataset的可迭代对象
    """

    conf = {
        "filter_conf": {
            "max_length": 1000,
            "min_length": 30,
            "token_max_length": 100,
            "token_min_length": 2,
        },
        "resample_conf": {"resample_rate": 16000},
        "speed_perturb": True,
        "fbank_conf": {
            "num_mel_bins": 80,
            "frame_shift": 10,
            "frame_length": 25,
            "dither": 1.0,
        },
        "spec_aug": True,
        "spec_sub": False,
        "spec_aug_conf": {
            "num_t_mask": 2,
            "num_f_mask": 2,
            "max_t": 50,
            "max_f": 30,
        },
        "shuffle": True,
        "shuffle_conf": {"shuffle_size": 10000},
        "sort": True,
        "sort_conf": {"sort_size": 2000},
        "batch_conf": {
            "batch_type": "dynamic",
            "batch_size": get_config("batch_size"),
            "max_frames_in_batch": 50000,
        },
        "no_pad": True,
    }

    dataset = WeNetProcessor(
        make_wenet_dataset(
            data_type="shard",
            data_list_file=config.info,
            symbol_table=get_config("symbol_table"),
            conf=conf,
            bpe_model=get_config("bpe_model_path"),
            non_lang_syms=None,
            partition=True,
        ),
        adapt_data_dict,
        downsample_rate=8,
    )

    return dataset


def get_tendis_ao_train_dataset(config: EasyDict) -> dict:
    """Get dataset for training.

    Args:
        config (EasyDict):
            use_hisf (bool): 是否使用hisf
            augment_config_path (str): 在线加噪的yaml配置文件
            info (str): 训练集的info路径
            noise_hdf5_file (str): 噪声hdf5路径
            rir (str): rir文件路径
            noise_info (str): 噪声的info路径

    Returns:
        dict: dataset的定义
    """

    if config.use_hisf:
        denoise_transform = dict(
            type=PyHisfDenoiser,
            pre_pad=True,
            channel_select=True,
            p=0.5,
        )
    else:
        denoise_transform = dict(type=ChannelSelect, channel=0)

    left_limit, right_limit = 0.3, 10

    with open(config.augment_config_path) as fr:
        augment_config = yaml.load(fr, Loader=yaml.FullLoader)

    dataset = dict(
        type=CocktailDatasetV0,
        audio_reader=dict(
            type=TendisWaveformReader,
            ret_dtype="float32",
            channel_first=True,
            tendis_kwargs=tendis_kwargs,
        ),
        image_reader=None,
        info_list=dict(
            type=MMASRInfoList,
            path=config.info,
            left_limit=left_limit,
            right_limit=right_limit,
        ),
        basic_info_reader=dict(
            type=TendisBasicInfoReader, tendis_kwargs=tendis_kwargs
        ),
        mode="train",
        vad_expand=0.2,
        transforms=dict(
            type=torchvision.transforms.Compose,
            transforms=[
                # 音频处理
                dict(type=ReSample, new_freq=16000),
                dict(
                    type=OnLineAugmentPipeline,
                    config=augment_config,
                    noise_hdf5_file=config.noise_hdf5_file,
                    ret_dtype="float32",
                    gpu_rir_path=config.rir,
                    noise_info=config.noise_info,
                    p=0.5,
                ),
                denoise_transform,
                dict(
                    type=FBank,
                    num_mel_bins=get_config("fbank_size"),
                    frame_shift=10,
                    frame_length=25,
                    dither=1.0,
                ),
                dict(
                    type=SpecAug,
                    num_t_mask=2,
                    num_f_mask=2,
                    max_t=50,
                    max_f=10,
                    p=1.0,
                ),
                # 标签处理
                dict(
                    type=Tokenize,
                    symbol_table=get_config("symbol_table"),
                    bpe_model_path=get_config("bpe_model_path"),
                    non_lang_syms={
                        "<blank>": 0,
                        "<unk>": 1,
                        "<sos/eos>": 5999,
                    },
                    split_with_space=False,
                ),
                dict(type=LabelToTensor),
            ],
        ),
    )

    return dataset


def get_json_ao_train_dataset(config):
    """Get dataset for training.

    Args:
        config (dict):
            use_hisf (bool): 是否使用hisf
            info (str): 训练集的info路径
            augment_config_path (str): 在线加噪的yaml配置文件
            noise_hdf5_file (str): 噪声hdf5路径
            rir (str): rir文件路径
            noise_info (str): 噪声的info路径

    Returns:
        dict: dataset的定义
    """

    if config.use_hisf:
        denoise_transform = dict(
            type=PyHisfDenoiser,
            pre_pad=True,
            channel_select=True,
            p=1.0,
        )
    else:
        denoise_transform = dict(type=ChannelSelect, channel=0)

    # if "augment_config_path" in config:
    with open(config.augment_config_path) as fr:
        augment_config = yaml.load(fr, Loader=yaml.FullLoader)
    # else:
    #     augment_config = None

    # if augment_config is not None:
    audio_aug = [
        dict(
            type=OnLineJsonAugmentPipeline,
            config=augment_config,
            noise_hdf5_file=config.noise_hdf5_file,
            ret_dtype="float32",
            gpu_rir_path=config.rir,
            noise_info=config.noise_info,
            p=0.5,
        ),
        denoise_transform,
    ]
    # else:
    #     audio_aug = [denoise_transform]

    dataset = dict(
        type=CocktailDatasetV2,
        audio_reader=dict(
            type=DirectWaveformReader,
            ret_dtype="float32",
            channel_first=True,
        ),
        image_reader=None,
        info_list=dict(
            type=MMASRJsonInfoList,
            path=config.info,
            left_limit=0.3,
            right_limit=10,
        ),
        mode="train",
        transforms=dict(
            type=torchvision.transforms.Compose,
            transforms=[
                # 音频处理
                dict(type=ReSample, new_freq=16000),
                *audio_aug,
                dict(
                    type=FBank,
                    num_mel_bins=get_config("fbank_size"),
                    frame_shift=10,
                    frame_length=25,
                    dither=1.0,
                ),
                dict(
                    type=SpecAug,
                    num_t_mask=2,
                    num_f_mask=2,
                    max_t=50,
                    max_f=10,
                    p=1.0,
                ),
                # 标签处理
                dict(
                    type=Tokenize,
                    symbol_table=get_config("symbol_table"),
                    bpe_model_path=get_config("bpe_model_path"),
                    non_lang_syms={
                        "<blank>": 0,
                        "<unk>": 1,
                        "<sos/eos>": 5999,
                    },
                    split_with_space=False,
                ),
                dict(type=LabelToTensor),
            ],
        ),
    )

    return dataset


def get_wenet_eval_dataset(config: EasyDict) -> Iterable:
    """

    Args:
        config (EasyDict):
            info: 数据集列表

    Returns:
        Iterable: 可以用做dataset的可迭代对象
    """

    conf = {
        "filter_conf": {
            "max_length": 2000,
            "min_length": 10,
            "token_max_length": 200,
            "token_min_length": 2,
        },
        "resample_conf": {"resample_rate": 16000},
        "speed_perturb": False,
        "fbank_conf": {
            "num_mel_bins": get_config("fbank_size"),
            "frame_shift": 10,
            "frame_length": 25,
            "dither": 0.0,
        },
        "spec_aug": False,
        "spec_sub": False,
        "shuffle": False,
        "sort": True,
        "sort_conf": {"sort_size": 500},
        "batch_conf": {
            "batch_type": "static",
            "batch_size": 8,
        },
        "no_pad": True,
    }

    dataset = WeNetProcessor(
        make_wenet_dataset(
            data_type="shard",
            data_list_file=config.info,
            symbol_table=get_config("symbol_table"),
            conf=conf,
            bpe_model=get_config("bpe_model_path"),
            non_lang_syms=None,
            partition=True,
        ),
        adapt_data_dict,
        downsample_rate=8,
    )

    return dataset


def get_tendis_ao_eval_dataset(config: EasyDict) -> dict:
    """Get dataset for validation or test.

    Args:
        config (EasyDict):
            use_hisf (bool): 是否使用hisf
            hdf5 (str): 验证集/测试集的info路径
            info (str): 测试集的info路径
            mode (str): val or test

    Returns:
        dict: dataset的定义
    """

    if config.use_hisf:
        denoise_transform = dict(
            type=PyHisfDenoiser,
            pre_pad=True,
            channel_select=True,
            p=1.0,
        )
    else:
        denoise_transform = dict(type=ChannelSelect, channel=0)

    dataset = dict(
        type=CocktailDatasetV0,
        audio_reader=dict(
            type=HDF5WaveformReader,
            hdf5_file=config.hdf5,
            ret_dtype="float32",
            channel_first=True,
        ),
        image_reader=None,
        info_list=dict(
            type=MMASRInfoList,
            path=config.info,
            left_limit=0.1,
            right_limit=100,
        ),
        basic_info_reader=dict(
            type=TendisBasicInfoReader, tendis_kwargs=tendis_kwargs
        ),
        mode=config.mode,
        vad_expand=0.2,
        transforms=dict(
            type=torchvision.transforms.Compose,
            transforms=[
                # 音频处理
                dict(type=ReSample, new_freq=16000),
                denoise_transform,
                dict(
                    type=FBank,
                    num_mel_bins=get_config("fbank_size"),
                    frame_shift=10,
                    frame_length=25,
                    dither=0.0,
                ),
                # 标签处理
                dict(
                    type=Tokenize,
                    symbol_table=get_config("symbol_table"),
                    bpe_model_path=get_config("bpe_model_path"),
                    non_lang_syms={
                        "<blank>": 0,
                        "<unk>": 1,
                        "<sos/eos>": 5999,
                    },
                    split_with_space=False,
                ),
                dict(type=LabelToTensor),
            ],
        ),
    )

    return dataset


def get_json_ao_eval_dataset(config):
    """Get dataset for validation or test.

    Args:
        config (EasyDict):
            use_hisf (bool): 是否使用hisf
            info (str): json文件的路径
            mode (str): val or test

    Returns:
        dict: dataset的定义
    """

    if config.use_hisf:
        denoise_transform = dict(
            type=PyHisfDenoiser,
            pre_pad=True,
            channel_select=True,
            p=1.0,
        )
    else:
        denoise_transform = dict(type=ChannelSelect, channel=0)

    dataset = dict(
        type=CocktailDatasetV2,
        audio_reader=dict(
            type=DirectWaveformReader,
            ret_dtype="float32",
            channel_first=True,
        ),
        image_reader=None,
        info_list=dict(
            type=MMASRJsonInfoList,
            path=config.info,
            left_limit=0.1,
            right_limit=100,
        ),
        mode=config.mode,
        transforms=dict(
            type=torchvision.transforms.Compose,
            transforms=[
                # 音频处理
                dict(type=ReSample, new_freq=16000),
                denoise_transform,
                dict(
                    type=FBank,
                    num_mel_bins=get_config("fbank_size"),
                    frame_shift=10,
                    frame_length=25,
                    dither=0.0,
                ),
                # 标签处理
                dict(
                    type=Tokenize,
                    symbol_table=get_config("symbol_table"),
                    bpe_model_path=get_config("bpe_model_path"),
                    non_lang_syms={
                        "<blank>": 0,
                        "<unk>": 1,
                        "<sos/eos>": 5999,
                    },
                    split_with_space=False,
                ),
                dict(type=LabelToTensor),
            ],
        ),
    )

    return dataset
