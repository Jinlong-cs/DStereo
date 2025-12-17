# flake8: noqa

import torchvision
from common.default import tendis_kwargs
from easydict import EasyDict

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
from hat.data.transforms.avspeech.images import (
    BrightnessContrast,
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
from hat.utils.root_helper import RootHelper
from projects.halo.avspeech.utils.global_config import get_config


def get_vo_train_dataset(config: EasyDict) -> dict:
    """

    Args:
        config (dict):
            info (str): 训练集的info

    Returns:
        dict: dataset的定义
    """

    left_limit, right_limit = 0.3, 10
    transforms = [
        # 图片处理
        dict(
            type=BrightnessContrast,
            brightness_limit=0.4,
            contrast_limit=0.4,
            brightness_by_max=False,
            p=0.3,
        ),
        dict(
            type=ImageListSpatialMask,
            p=0.2,
            mode="mean",
            min_width=32,
            max_width=48,
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
        dict(type=ImageListToYUV444),
        dict(
            type=ImageListNormalize,
            mean=[128.0, 128.0, 128.0],
            std=[128.0, 128.0, 128.0],
        ),
        # 标签处理
        dict(
            type=Tokenize,
            symbol_table=get_config("symbol_table"),
            bpe_model_path=get_config("bpe_model_path"),
            non_lang_syms=None,
            split_with_space=False,
        ),
        dict(type=LabelToTensor),
    ]

    dataset = dict(
        type=CocktailDatasetV0,
        audio_reader=None,
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
        vad_expand=0.2,
        transforms=dict(
            type=torchvision.transforms.Compose, transforms=transforms
        ),
    )

    return dataset


def get_vo_eval_dataset(config: EasyDict) -> dict:
    """

    Args:
        config (dict):
            mode (str): test or val
            info (str): 数据集的info路径

    Returns:
        dict: _description_
    """
    assert config.mode in ["test", "val"]

    left_limit, right_limit = 0.1, 100
    transforms = [
        # 图片处理
        dict(type=ImageListStack, hwc2chw=True),
        dict(type=ImageListToYUV444),
        dict(
            type=ImageListNormalize,
            mean=[128.0, 128.0, 128.0],
            std=[128.0, 128.0, 128.0],
        ),
        # 标签处理
        dict(
            type=Tokenize,
            symbol_table=get_config("symbol_table"),
            bpe_model_path=get_config("bpe_model_path"),
            non_lang_syms=None,
            split_with_space=False,
        ),
        dict(type=LabelToTensor),
    ]

    dataset = dict(
        type=CocktailDatasetV0,
        audio_reader=None,
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
        mode=config.mode,
        vad_expand=0.2,
        transforms=dict(
            type=torchvision.transforms.Compose, transforms=transforms
        ),
    )
    return dataset


def get_vo_json_train_dataset(config):
    left_limit, right_limit = 0.3, 10
    transforms = [
        # 图片处理
        dict(
            type=BrightnessContrast,
            brightness_limit=0.4,
            contrast_limit=0.4,
            brightness_by_max=False,
            p=0.3,
        ),
        dict(
            type=ImageListSpatialMask,
            p=0.2,
            mode="mean",
            min_width=32,
            max_width=48,
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
        dict(type=ImageListStack, hwc2chw=True),
        dict(type=ImageListRandomFlip, p=0.5),
        dict(
            type=ImageListRandomCrop,
            scale=(0.8, 1.0),
            ratio=(1.0, 1.0),
            p=0.5,
        ),
        # dict(
        #     type=TimeMask,
        #     p=0.5,
        #     max_frame=15,
        #     num_mask=2,
        #     replace_with_zero=False,
        # ),
        # # dict(type=SyntheticVideo, save_dir="./tmp_check_json/"),
        dict(type=ImageListToYUV444),
        dict(
            type=ImageListNormalize,
            mean=[128.0, 128.0, 128.0],
            std=[128.0, 128.0, 128.0],
        ),
        # 标签处理
        dict(
            type=Tokenize,
            symbol_table=get_config("symbol_table"),
            bpe_model_path=get_config("bpe_model_path"),
            non_lang_syms=None,
            split_with_space=False,
        ),
        dict(type=LabelToTensor),
    ]

    dataset = dict(
        type=CocktailDatasetV2,
        audio_reader=None,
        image_reader=dict(
            type=DirectVideoImageReader,
            fps=25,
        ),
        info_list=dict(
            type=MMASRJsonInfoList,
            path=config.info,
            left_limit=left_limit,
            right_limit=right_limit,
        ),
        mode="train",
        transforms=dict(
            type=torchvision.transforms.Compose, transforms=transforms
        ),
    )

    return dataset


def get_vo_json_eval_dataset(config):
    left_limit, right_limit = 0.1, 100
    transforms = [
        # 图片处理
        dict(type=ImageListStack, hwc2chw=True),
        # # dict(type=SyntheticVideo, save_dir="./tmp_check_json/"),
        dict(type=ImageListToYUV444),
        dict(
            type=ImageListNormalize,
            mean=[128.0, 128.0, 128.0],
            std=[128.0, 128.0, 128.0],
        ),
        # 标签处理
        dict(
            type=Tokenize,
            symbol_table=get_config("symbol_table"),
            bpe_model_path=get_config("bpe_model_path"),
            non_lang_syms=None,
            split_with_space=False,
        ),
        dict(type=LabelToTensor),
    ]

    dataset = dict(
        type=CocktailDatasetV2,
        audio_reader=None,
        image_reader=dict(
            type=DirectVideoImageReader,
            fps=25,
        ),
        info_list=dict(
            type=MMASRJsonInfoList,
            path=config.info,
            left_limit=left_limit,
            right_limit=right_limit,
        ),
        mode=config.mode,
        transforms=dict(
            type=torchvision.transforms.Compose, transforms=transforms
        ),
    )

    return dataset
