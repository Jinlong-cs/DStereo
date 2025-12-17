# Copyright (c) Horizon Robotics. All rights reserved.

import os

import pytest
import torchvision

from hat.data.datasets.avspeech.label import VadLabelMaskGeneratorV1
from hat.registry import build_from_registry
from tests import HAT_BUCKET_PATH

try:
    import redis
except ImportError:
    redis = None

tendis_kwargs = dict(
    host="aidi-kv-cluster-01.hogpu.cc",
    port=7617,
    db=0,
    socket_connect_timeout=5000,
    password="pB9cA2eN1eD1lJ0",
    socket_keepalive=False,
)


@pytest.mark.skipif(redis is None, reason="need redis")
def test_cocktail_dataset():
    mmasr_train_data_path = os.path.join(
        HAT_BUCKET_PATH,
        "unit_test_data/avspeech/data/datasets/cocktail_dataset_v0_test_info.txt",  # noqa: E501
    )

    transforms = dict(
        type=torchvision.transforms.Compose,
        transforms=[
            dict(type="ReSample", new_freq=16000),
            dict(
                type="FBank",
                num_mel_bins=80,
                frame_shift=10,
                frame_length=25,
                dither=1.0,
            ),
            dict(type="ImageListStack", hwc2chw=True),
            dict(type="ImageListToYUV444"),
            dict(
                type="ImageListNormalize",
                mean=[128.0, 128.0, 128.0],
                std=[128.0, 128.0, 128.0],
            ),
        ],
    )

    data_cfg = dict(
        type="CocktailDatasetV0",
        audio_reader=dict(
            type="TendisWaveformReader",
            ret_dtype="float32",
            channel_first=True,
            tendis_kwargs=tendis_kwargs,
        ),
        image_reader=dict(
            type="TendisVideoImageReader",
            fps=25,
            origin_fps=30,
            zfill_num=6,
            missing_image_complete_mode="nearest",
            filter_rules={"missing_ratio": {"threshold": 0.3}},
            tendis_kwargs=tendis_kwargs,
        ),
        info_list=dict(
            type="MMASRInfoList",
            path=mmasr_train_data_path,
            left_limit=0.3,
            right_limit=10,
        ),
        basic_info_reader=dict(
            type="TendisBasicInfoReader", tendis_kwargs=tendis_kwargs
        ),
        mode="train",
        vad_expand=0.2,
        transforms=transforms,
    )
    dataset = build_from_registry(data_cfg)

    for idx, batch in enumerate(dataset):
        assert batch["audio"].shape[1] == 80
        assert batch["images"].shape[1:4] == (3, 96, 96)

        if idx > 2:
            break


@pytest.mark.skipif(redis is None, reason="need redis")
def test_lipmove_dataset():
    lipmove_train_data_path = os.path.join(
        HAT_BUCKET_PATH,
        "unit_test_data/avspeech/data/datasets/lipmove_dataset_test_info.txt",  # noqa: E501
    )

    transforms = dict(
        type=torchvision.transforms.Compose,
        transforms=[
            dict(type="ImageListStack", hwc2chw=True),
            dict(type="ImageListToYUV444"),
            dict(
                type="ImageListNormalize",
                mean=[128.0, 128.0, 128.0],
                std=[128.0, 128.0, 128.0],
            ),
        ],
    )

    data_cfg = dict(
        type="LipmoveDatasetV0",
        image_reader=dict(
            type="TendisVideoImageReader",
            missing_image_complete_mode="zero",
            tendis_kwargs=tendis_kwargs,
        ),
        info_list=dict(
            type="LipMoveInfoList",
            path=lipmove_train_data_path,
        ),
        basic_info_reader=dict(
            type="TendisBasicInfoReader",
            tendis_kwargs=tendis_kwargs,
        ),
        label_mask_generator=dict(type=VadLabelMaskGeneratorV1),
        transforms=transforms,
    )
    dataset = build_from_registry(data_cfg)

    for idx, batch in enumerate(dataset):
        assert batch["images"].shape[1:4] == (3, 96, 96)
        assert batch["images"].shape[0] == batch["label"].shape[0]

        if idx > 2:
            break
