# flake8: noqa

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
from hat.data.datasets.avspeech.image import (
    DirectVideoImageReader,
    TendisVideoImageReader,
)
from hat.data.datasets.avspeech.info import MMASRInfoList, MMASRJsonInfoList
from hat.data.datasets.avspeech_dataset import (
    CocktailDatasetV0,
    CocktailDatasetV2,
)
from hat.data.transforms.avspeech.align import ForceAlign
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
from hat.data.transforms.avspeech.images import (
    BrightnessContrast,
    DownSampleAugment,
    FancyPCA,
    HueSaturateValue,
    ImageListNormalize,
    ImageListRandomCrop,
    ImageListRandomFlip,
    ImageListSpatialMask,
    ImageListStack,
    ImageListToYUV444,
    TimeMask,
    ToGray,
)
from hat.data.transforms.avspeech.text import LabelToTensor, Tokenize
from projects.halo.avspeech.utils.global_config import get_config


def get_train_dataset(config: EasyDict) -> dict:
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

    audio_transforms = [
        dict(type=ReSample, new_freq=16000),
        dict(
            type=OnLineAugmentPipeline,
            config=augment_config,
            noise_hdf5_file=config.noise_hdf5_file,
            ret_dtype="float32",
            gpu_rir_path=config.rir,
            noise_info=config.noise_info,
            p=config.get("noise_prob", 0.5),
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
    ]

    images_transforms = [
        dict(
            type=BrightnessContrast,
            brightness_limit=0.4,
            contrast_limit=0.4,
            brightness_by_max=False,
            p=0.3,
        ),
        dict(
            type=HueSaturateValue,
            hue_shift_limit=10,
            sat_shift_limit=30,
            val_shift_limit=0,
            p=0.2,
        ),
        dict(
            type=FancyPCA,
            alpha=0.8,
            p=0.3,
        ),
        dict(
            type=ToGray,
            p=0.3,
            rgb_data=False,
        ),
        # dict(type=SyntheticVideo),
        dict(type=ImageListStack, hwc2chw=True),
        dict(type=ImageListRandomFlip, p=0.5),
        dict(
            type=ImageListRandomCrop,
            scale=(0.8, 1.0),
            ratio=(1.0, 1.0),
            p=0.5,
        ),
        dict(
            type=TimeMask,
            p=0.5,
            max_frame=15,
            num_mask=2,
            replace_with_zero=False,
        ),
        dict(
            type=ImageListSpatialMask,
            p=0.2,
            mode="mean",
            min_width=32,
            max_width=48,
        ),
        dict(type=ImageListToYUV444),
        dict(
            type=ImageListNormalize,
            mean=[128.0, 128.0, 128.0],
            std=[128.0, 128.0, 128.0],
        ),
    ]

    dataset = dict(
        type=CocktailDatasetV0,
        audio_reader=dict(
            type=TendisWaveformReader,
            ret_dtype="float32",
            channel_first=True,
            tendis_kwargs=tendis_kwargs,
        ),
        image_reader=dict(
            type=TendisVideoImageReader,
            fps=25,
            origin_fps=30,
            zfill_num=6,
            missing_image_complete_mode="nearest",
            filter_rules={"missing_ratio": {"threshold": 0.3}},
            tendis_kwargs=tendis_kwargs,
        ),
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
        vad_expand=0,
        transforms=dict(
            type=torchvision.transforms.Compose,
            transforms=[
                # 音频处理
                *audio_transforms,
                # 强制对齐
                dict(type=ForceAlign, audio_left_extend=7),
                # 图片处理
                *images_transforms,
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


def get_json_train_dataset(config):
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

    with open(config.augment_config_path) as fr:
        augment_config = yaml.load(fr, Loader=yaml.FullLoader)

    audio_transforms = [
        # 音频处理
        dict(type=ReSample, new_freq=16000),
        dict(
            type=OnLineJsonAugmentPipeline,
            config=augment_config,
            noise_hdf5_file=config.noise_hdf5_file,
            ret_dtype="float32",
            gpu_rir_path=config.rir,
            noise_info=config.noise_info,
            p=config.get("noise_prob", 0.5),
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
    ]

    images_transforms = [
        dict(
            type=BrightnessContrast,
            brightness_limit=0.4,
            contrast_limit=0.4,
            brightness_by_max=False,
            p=0.3,
        ),
        dict(
            type=HueSaturateValue,
            hue_shift_limit=10,
            sat_shift_limit=30,
            val_shift_limit=0,
            p=0.2,
        ),
        dict(
            type=FancyPCA,
            alpha=0.8,
            p=0.3,
        ),
        dict(
            type=ToGray,
            p=0.5,
            rgb_data=False,
        ),
        dict(
            type=DownSampleAugment,
            p=0.5,
            min_size=(24, 24),
            max_size=(56, 56),
        ),
        dict(type=ImageListStack, hwc2chw=True),
        dict(type=ImageListRandomFlip, p=0.5),
        dict(
            type=TimeMask,
            p=0.5,
            max_frame=15,
            num_mask=2,
            replace_with_zero=False,
        ),
        dict(
            type=ImageListSpatialMask,
            p=0.2,
            mode="mean",
            min_width=32,
            max_width=48,
        ),
        dict(
            type=ImageListRandomCrop,
            scale=(0.8, 1.0),
            ratio=(1.0, 1.0),
            p=0.5,
        ),
        dict(type=ImageListToYUV444),
        dict(
            type=ImageListNormalize,
            mean=[128.0, 128.0, 128.0],
            std=[128.0, 128.0, 128.0],
        ),
    ]

    dataset = dict(
        type=CocktailDatasetV2,
        audio_reader=dict(
            type=DirectWaveformReader,
            ret_dtype="float32",
            channel_first=True,
        ),
        image_reader=dict(
            type=DirectVideoImageReader,
            fps=25,
        ),
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
                *audio_transforms,
                # 强制对齐
                dict(type=ForceAlign, audio_left_extend=7),
                # 图片处理
                *images_transforms,
                # 标签处理
                dict(
                    type=Tokenize,
                    symbol_table=get_config("symbol_table"),
                    bpe_model_path=get_config("bpe_model_path"),
                    split_with_space=False,
                ),
                dict(type=LabelToTensor),
            ],
        ),
    )

    return dataset


def get_json_train_face_dataset(config):
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

    with open(config.augment_config_path) as fr:
        augment_config = yaml.load(fr, Loader=yaml.FullLoader)

    audio_transforms = [
        dict(type=ReSample, new_freq=16000),
        dict(
            type=OnLineJsonAugmentPipeline,
            config=augment_config,
            noise_hdf5_file=config.noise_hdf5_file,
            ret_dtype="float32",
            gpu_rir_path=config.rir,
            noise_info=config.noise_info,
            p=config.get("noise_prob", 0.5),
            # p=0.5,
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
    ]

    images_transforms = [
        dict(
            type=BrightnessContrast,
            brightness_limit=0.4,
            contrast_limit=0.4,
            brightness_by_max=False,
            p=0.3,
        ),
        dict(
            type=HueSaturateValue,
            hue_shift_limit=10,
            sat_shift_limit=30,
            val_shift_limit=0,
            p=0.2,
        ),
        dict(
            type=FancyPCA,
            alpha=0.8,
            p=0.3,
        ),
        dict(
            type=ToGray,
            p=0.5,
            rgb_data=False,
        ),
        dict(
            type=DownSampleAugment,
            p=0.5,
            min_size=(24, 24),
            max_size=(56, 56),
        ),
        dict(type=ImageListStack, hwc2chw=True),
        dict(type=ImageListRandomFlip, p=0.5),
        dict(
            type=ImageListRandomCrop,
            scale=(0.8, 1.0),
            ratio=(1.0, 1.0),
            p=0.5,
        ),
        dict(type=ImageListToYUV444),
        dict(
            type=ImageListNormalize,
            mean=[128.0, 128.0, 128.0],
            std=[128.0, 128.0, 128.0],
        ),
    ]

    dataset = dict(
        type=CocktailDatasetV2,
        audio_reader=dict(
            type=DirectWaveformReader,
            ret_dtype="float32",
            channel_first=True,
        ),
        image_reader=dict(
            type=DirectVideoImageReader,
            fps=25,
        ),
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
                *audio_transforms,
                # 强制对齐
                dict(type=ForceAlign, audio_left_extend=7),
                # 图片处理
                *images_transforms,
                # 标签处理
                dict(
                    type=Tokenize,
                    symbol_table=get_config("symbol_table"),
                    bpe_model_path=get_config("bpe_model_path"),
                    split_with_space=False,
                ),
                dict(type=LabelToTensor),
            ],
        ),
    )

    return dataset


def get_eval_dataset(config: EasyDict) -> dict:
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
    audio_transforms = [
        dict(type=ReSample, new_freq=16000),
        denoise_transform,
        dict(
            type=FBank,
            num_mel_bins=get_config("fbank_size"),
            frame_shift=10,
            frame_length=25,
            dither=0.0,
        ),
    ]

    images_transforms = [
        dict(type=ImageListStack, hwc2chw=True),
        dict(type=ImageListToYUV444),
        dict(
            type=ImageListNormalize,
            mean=[128.0, 128.0, 128.0],
            std=[128.0, 128.0, 128.0],
        ),
    ]

    dataset = dict(
        type=CocktailDatasetV0,
        audio_reader=dict(
            type=HDF5WaveformReader,
            hdf5_file=config.hdf5,
            ret_dtype="float32",
            channel_first=True,
        ),
        image_reader=dict(
            type=TendisVideoImageReader,
            fps=25,
            origin_fps=30,
            zfill_num=6,
            missing_image_complete_mode="nearest",
            filter_rules={"missing_ratio": {"threshold": 0.3}},
            tendis_kwargs=tendis_kwargs,
        ),
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
                *audio_transforms,
                # 强制对齐
                dict(type=ForceAlign, audio_left_extend=7),
                # 图片处理
                *images_transforms,
                # 标签处理
                dict(
                    type=Tokenize,
                    symbol_table=get_config("symbol_table"),
                    bpe_model_path=get_config("bpe_model_path"),
                    split_with_space=False,
                ),
                dict(type=LabelToTensor),
            ],
        ),
    )

    return dataset


def get_json_eval_dataset(config):
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

    audio_transforms = [
        dict(type=ReSample, new_freq=16000),
        denoise_transform,
        dict(
            type=FBank,
            num_mel_bins=get_config("fbank_size"),
            frame_shift=10,
            frame_length=25,
            dither=0.0,
        ),
    ]

    images_transforms = [
        dict(type=ImageListStack, hwc2chw=True),
        dict(type=ImageListToYUV444),
        dict(
            type=ImageListNormalize,
            mean=[128.0, 128.0, 128.0],
            std=[128.0, 128.0, 128.0],
        ),
    ]

    dataset = dict(
        type=CocktailDatasetV2,
        audio_reader=dict(
            type=DirectWaveformReader,
            ret_dtype="float32",
            channel_first=True,
        ),
        image_reader=dict(
            type=DirectVideoImageReader,
            fps=25,
        ),
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
                *audio_transforms,
                # 强制对齐
                dict(type=ForceAlign, audio_left_extend=7),
                # 图片处理
                *images_transforms,
                # 标签处理
                dict(
                    type=Tokenize,
                    symbol_table=get_config("symbol_table"),
                    bpe_model_path=get_config("bpe_model_path"),
                    split_with_space=False,
                ),
                dict(type=LabelToTensor),
            ],
        ),
    )

    return dataset
