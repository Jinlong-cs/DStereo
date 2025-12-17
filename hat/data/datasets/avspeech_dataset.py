# Copyright (c) Horizon Robotics, All rights reserved.
"""AVSPEECH(多模语音)数据迭代器.

目前该模块下共有四个类:

CocktailDatasetV0   : 多模通用识别数据迭代器.
CocktailDatasetV2   : 多模通用识别JSON格式数据迭代器.
CocktailCmdDataset  : 多模命令词数据迭代器.
LipmoveDatasetV0    : 唇动检测数据迭代器.
"""

import logging
import pathlib
import random
from copy import deepcopy

import torch
from torch.utils import data

from hat.data.datasets.avspeech.audio import BaseWaveformReader
from hat.data.datasets.avspeech.basic_data_info import TendisBasicInfoReader
from hat.data.datasets.avspeech.image import BaseImageReader
from hat.data.datasets.avspeech.info import BaseInfoList, UttParser
from hat.data.datasets.avspeech.label import VadLabelMaskGeneratorV1
from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type.mxrecord import MXRecord
from hat.utils.package_helper import require_packages

try:
    import mxnet as mx
except ImportError:
    mx = None


@OBJECT_REGISTRY.register
class CocktailDatasetV0(data.Dataset):
    """CocktailDatasetV0.

    多模通用识别数据迭代器.

    Args:
        image_reader: 图像数据读取器.
        audio_reader: 音频数据读取器.
        info_list: 训练集label信息列表.
        basic_info_reader: 基础标签信息读取器.
        mode: 使用数据迭代器的模式, 默认为"train".
        transforms : 数据增强、数据格式转换器, 默认为None.
        vad_expand: vad向左右两边扩展的范围, 默认为0.0.
    """

    SUPPORT_MODE = ["train", "val", "test"]

    def __init__(
        self,
        image_reader: BaseImageReader,
        audio_reader: BaseWaveformReader,
        info_list: BaseInfoList,
        basic_info_reader: TendisBasicInfoReader,
        mode: str = "train",
        transforms=None,
        vad_expand: float = 0.0,
    ):
        self.image_reader = image_reader
        self.audio_reader = audio_reader
        self.info_list = info_list
        self.basic_info_reader = basic_info_reader
        self.utt_parser = UttParser()
        assert (
            mode in self.SUPPORT_MODE
        ), f"mode should be in {self.SUPPORT_MODE}, but get {mode}"
        self.mode = mode
        self.transforms = transforms
        self.vad_expand = vad_expand
        assert (
            self.vad_expand >= 0
        ), f"vad_expand shoud >= 0, but get {self.vad_expand}"
        assert image_reader is not None or audio_reader is not None

    def __getitem__(self, index) -> dict:
        info = deepcopy(self.info_list[index])
        logging.debug(f"Info: {info}")
        # 利用 info 信息解析其他数据
        matched = self.utt_parser.match(info["seg_utt"])
        logging.debug(f"matched: {matched}")
        snt_utt = f"{matched['dataset']}_{matched['batch']}-{matched['snt']}"
        basic_info = self.basic_info_reader.read_basic_data_info(snt_utt)

        if self.vad_expand > 0:
            info = self.expand_info_vad(info, basic_info)

        role = basic_info.get_seat_role()
        # 构造返回的数据dict
        data = {"role": role, "label": info["text"]}
        if self.mode == "test":
            if role == "pilot":
                cams = ["cam1"]
            elif role == "copilot":
                cams = ["cam5"]
        else:
            cams = basic_info.get_available_cam_ids()
            logging.debug(f"cams: {cams}")
            if self.mode == "train":
                random.shuffle(cams)
        # 获取图片序列
        cam = None
        if self.image_reader is not None:
            for cam in cams:
                video_utt = f"{matched['dataset']}_{matched['batch']}-{matched['snt']}_{cam}"  # noqa E501
                logging.debug(f"video_utt: {video_utt}")
                images = self.image_reader.read_seg_images(
                    video_utt, info["seg_beg"], info["seg_end"]
                )
                if images is not None:
                    break
            if images is None:
                return None
            logging.debug(f"images: {len(images)} {images[0].shape}")
            data["images"] = images
            data["images_lens"] = len(images)
        # 获取音频序列
        mics = [matched["mic"]]
        if mics[0] is None:
            mics = basic_info.get_available_mic_ids()
            logging.debug(f"mics: {mics}")
            if self.mode == "train":
                random.shuffle(mics)

        if self.audio_reader is not None:
            for mic in mics:
                audio_utt = f"{matched['dataset']}_{matched['batch']}-{matched['snt']}_{mic}"  # noqa E501

                logging.debug(f"audio_utt: {audio_utt}")
                waveform, samplerate = self.audio_reader.read_seg_audio(
                    audio_utt, info["seg_beg"], info["seg_end"]
                )
                if waveform is not None:
                    break
            if waveform is None:
                return None
            channel_dim = 0 if self.audio_reader.channel_first else 1
            logging.debug(f"audio: {waveform.shape}, {samplerate}")
            waveform = torch.tensor(waveform)
            logging.debug(f"audio: {waveform.shape}, {samplerate}")

            data["waveform"] = (waveform, samplerate)
            data["channel_dim"] = channel_dim
        # 获取label
        logging.debug(f"text: {info['text']}")
        key = f"{matched['dataset']}_{matched['batch']}-{matched['snt']}"
        if self.audio_reader is not None:
            key = key + f"_{mic}"
        if self.image_reader is not None:
            key = key + f"_{cam}"
        key = key + f"_{matched['index']}"  # noqa: E501
        data["key"] = key
        # 对数据进行各种处理
        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def expand_info_vad(self, info, basic_info):
        if self.mode == "train":
            vad_expand = (0.5 + random.random() / 2) * self.vad_expand
        else:
            vad_expand = self.vad_expand

        vad_list = basic_info.get_vad_segment_list()
        _duration = basic_info.get_audio_length()
        left_limit = 0
        right_limit = _duration
        seg_beg = info["seg_beg"]
        seg_end = info["seg_end"]
        for vad in vad_list:
            if vad[1] < seg_beg and vad[1] > left_limit:
                left_limit = vad[1]
            elif vad[0] > seg_end and vad[0] < right_limit:
                right_limit = vad[0]
        new_info = deepcopy(info)
        new_info["seg_beg"] = max(left_limit, seg_beg - vad_expand)
        new_info["seg_end"] = min(right_limit, seg_end + vad_expand)

        return new_info

    def __len__(self):
        return len(self.info_list)


class CocktailDatasetV2(data.Dataset):
    """CocktailDatasetV2.

    多模通用识别JSON格式数据迭代器.

    Args:
        image_reader: 图像数据读取器.
        audio_reader: 音频数据读取器.
        info_list: 训练集label信息列表.
        mode: 使用数据迭代器的模式, 默认为"train".
        transforms : 数据增强、数据格式转换器, 默认为None.
    """

    SUPPORT_MODE = ["train", "val", "test"]

    def __init__(
        self,
        image_reader: BaseImageReader,
        audio_reader: BaseWaveformReader,
        info_list: BaseInfoList,
        mode: str = "train",
        transforms=None,
    ):
        self.image_reader = image_reader
        self.audio_reader = audio_reader
        self.info_list = info_list
        assert (
            mode in self.SUPPORT_MODE
        ), f"mode should be in {self.SUPPORT_MODE}, but get {mode}"
        self.mode = mode
        self.transforms = transforms
        assert image_reader is not None or audio_reader is not None

    def __getitem__(self, index):
        item = self.info_list[index]
        data = {"label": item["text"]}
        # 读取图片列表
        key = None
        if self.image_reader is not None:
            images = self.image_reader.read_seg_images(item["mp4_path"])
            if images is None:
                return None
            logging.debug(f"images: {len(images)} {images[0].shape}")
            data["images"] = images
            data["images_lens"] = len(images)
            key = pathlib.Path(item["mp4_path"]).stem
        # 读取waveform
        if self.audio_reader is not None:
            waveform, samplerate = self.audio_reader.read_seg_audio(
                item["wav_path"]
            )
            if waveform is None:
                return None
            channel_dim = 0 if self.audio_reader.channel_first else 1
            logging.debug(f"audio: {waveform.shape}, {samplerate}")
            waveform = torch.tensor(waveform)
            logging.debug(f"audio: {waveform.shape}, {samplerate}")
            data["waveform"] = (waveform, samplerate)
            data["channel_dim"] = channel_dim
            wav_key = pathlib.Path(item["wav_path"]).stem
            key = wav_key if key is None else f"{key}_{wav_key}"
        data["key"] = key
        # 对数据进行各种数据增强和处理
        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def __len__(self):
        return len(self.info_list)


class CocktailCmdDataset(data.Dataset):
    """CocktailCmdDataset.

    多模命令词数据迭代器.

    Args:
        image_reader: 图像数据读取器.
        audio_reader: 音频数据读取器.
        info_list: 训练集label信息列表.
        basic_info_reader: 基础标签信息读取器.
        mode: 使用数据迭代器的模式, 默认为"train".
        transforms : 数据增强、数据格式转换器, 默认为None.
    """

    SUPPORT_MODE = ["train", "val", "test"]

    @require_packages("mxnet")
    def __init__(
        self,
        image_reader: BaseImageReader,
        audio_reader: MXRecord,
        info_list: BaseInfoList,
        basic_info_reader: TendisBasicInfoReader,
        mode: str = "train",
        transforms=None,
    ):
        self.image_reader = image_reader
        self.audio_reader = audio_reader
        self.info_list = info_list
        self.basic_info_reader = basic_info_reader
        self.utt_parser = UttParser()
        assert (
            mode in self.SUPPORT_MODE
        ), f"mode should be in {self.SUPPORT_MODE}, but get {mode}"
        self.mode = mode
        self.transforms = transforms

    def __getitem__(self, index) -> dict:
        info = self.info_list[index]
        logging.debug(f"Info: {info}")
        # 利用 info 信息解析其他数据
        matched = self.utt_parser.match(info["seg_utt"])
        logging.debug(f"matched: {matched}")
        snt_utt = f"{matched['dataset']}_{matched['batch']}-{matched['snt']}"
        basic_info = self.basic_info_reader.read_basic_data_info(snt_utt)
        cams = basic_info.get_available_cam_ids()
        logging.debug(f"cams: {cams}")
        cam = random.choice(cams) if self.mode == "train" else cams[0]
        # 获取图片序列
        video_utt = (
            f"{matched['dataset']}_{matched['batch']}-{matched['snt']}_{cam}"
        )
        logging.debug(f"video_utt: {video_utt}")
        images = self.image_reader.read_seg_images(
            video_utt, info["seg_beg"], info["seg_end"]
        )
        logging.debug(f"images: {len(images)} {images[0].shape}")
        # 获取音频序列
        rec_idx = random.choice(info["rec_idx"])
        s = self.audio_reader.read(rec_idx)
        audio = mx.nd.load_from_str(s).asnumpy()
        audio = torch.from_numpy(audio)
        logging.debug(f"audio_fbank: {audio.shape}")
        # 获取label
        logging.debug(f"text: {info['text']}")
        # 构造返回的数据dict
        key = f"{matched['dataset']}_{matched['batch']}-{matched['snt']}_{cam}_{matched['mic']}_{matched['index']}"  # noqa: E501
        data = {
            "key": key,
            "images": images,
            "audio": audio,
            "tokens": info["tokens"],
        }
        # 对数据进行各种处理
        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def __len__(self):
        return len(self.info_list)


@OBJECT_REGISTRY.register
class LipmoveDatasetV0(data.Dataset):
    """LipmoveDatasetV0.

    唇动检测数据迭代器.

    Args:
        image_reader: 图像数据读取器.
        info_list: 训练集label信息列表.
        basic_info_reader: 基础标签信息读取器.
        label_mask_generator: 训练标签及对应mask的生成器.
        mode: 使用数据迭代器的模式, 默认为"train".
        transforms : 数据增强、数据格式转换器, 默认为None.
    """

    SUPPORT_MODE = ["train", "val", "test"]

    def __init__(
        self,
        image_reader: BaseImageReader,
        info_list: BaseInfoList,
        basic_info_reader: TendisBasicInfoReader,
        label_mask_generator: VadLabelMaskGeneratorV1,
        mode: str = "train",
        transforms=None,
    ):
        self.image_reader = image_reader
        self.info_list = info_list
        self.basic_info_reader = basic_info_reader
        self.utt_parser = UttParser()
        assert (
            mode in self.SUPPORT_MODE
        ), f"mode should be in {self.SUPPORT_MODE}, but get {mode}"
        self.mode = mode
        self.transforms = transforms
        self.label_mask_generator = label_mask_generator

    def __getitem__(self, index) -> dict:
        info = self.info_list[index]
        logging.debug(f"Info: {info}")
        # 利用 info 信息解析其他数据
        matched = self.utt_parser.match(info["seg_utt"])
        logging.debug(f"matched: {matched}")
        snt_utt = f"{matched['dataset']}_{matched['batch']}-{matched['snt']}"
        basic_info = self.basic_info_reader.read_basic_data_info(snt_utt)
        if basic_info is None:
            cams = ["cam1"]
        else:
            cams = basic_info.get_available_cam_ids()
        logging.debug(f"cams: {cams}")
        cam = random.choice(cams) if self.mode == "train" else cams[0]
        # 获取图片序列
        video_utt = (
            f"{matched['dataset']}_{matched['batch']}-{matched['snt']}_{cam}"
        )
        logging.debug(f"video_utt: {video_utt}")
        images = self.image_reader.read_seg_images(
            video_utt, info["seg_beg"], info["seg_end"]
        )
        logging.debug(f"images: {len(images)} {images[0].shape}")
        label, mask, weight_mask = self.label_mask_generator.generate(
            info["seg_beg"], info["seg_end"], info["vad_beg"], info["vad_end"]
        )
        key = f"{matched['dataset']}_{matched['batch']}-{matched['snt']}_{cam}_{matched['index']}"  # noqa: E501
        if len(images) > 180:
            images, label = images[:180], label[:180]
        data = {
            "key": key,
            "images": images,
            "label": label,
            # "mask": mask,
            # "weight_mask": weight_mask,
        }
        # 对数据进行各种处理
        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def __len__(self):
        return len(self.info_list)
