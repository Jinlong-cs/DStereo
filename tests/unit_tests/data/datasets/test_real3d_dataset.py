# Copyright (c) Horizon Robotics. All rights reserved.

import os

import pytest

from hat.data.datasets.real3d_dataset import Real3DDataset, Real3DRecReader
from hat.registry import build_from_registry
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH

try:
    import auto_matrix
except ImportError:
    auto_matrix = None

try:
    import pycocotools
except ImportError:
    pycocotools = None


@pytest.mark.skipif(pycocotools is None, reason="need pycocotools")
@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_real3d_dataset():
    # test function get_category_id_dict
    category_id_dict = Real3DDataset.get_category_id_dict(3)
    imgdir = os.path.join(
        HAT_BUCKET_PATH,
        "unit_test_data/J5FSD/detection_3D/pandar64_distorted/train/images",  # noqa
    )
    jsonfile = os.path.join(
        HAT_BUCKET_PATH,
        "unit_test_data/J5FSD/detection_3D/pandar64_distorted/train/annotations/test.json",  # noqa
    )
    cfg = dict(
        type="Real3DDataset",
        paths=dict(
            img_dir=imgdir,
            anno_path=[
                jsonfile,
            ],
        ),
        num_classes=3,
        transforms=[
            dict(type="ImageTransform", size=(960, 512)),
            dict(type="ImageToTensor", from_numpy=True),
            dict(type="ConvertLayout", hwc2chw=True),
            dict(type="ImageBgrToYuv444", rgb_input=False),
            dict(type="ImageNormalize", mean=128.0, std=128.0),
            dict(
                type="Real3dTargetGenerator",
                num_classes=3,
                focal_length_default=1548.9354248046875,
                input_size=(960, 512),
                category_id_dict=category_id_dict,
                down_stride=4,
                max_objs=100,
            ),
            dict(type="ParseDataReal3DMultitask"),
        ],
        num_dist=8,
    )

    # test function construction
    dataset = build_from_registry(cfg)

    # test function __len__
    length_dataset = len(dataset)
    assert length_dataset == 1

    # test function __getitem__
    assert "image_name" in dataset[0]
    assert "image_height" in dataset[0]
    assert "image_width" in dataset[0]
    assert "imgs" in dataset[0]
    assert "image_id" in dataset[0]
    assert "ignore_mask" in dataset[0]
    assert "image_transform" in dataset[0]
    assert "target" in dataset[0]
    assert dataset[0]["image_height"] == 1080
    assert dataset[0]["image_width"] == 1920


@pytest.mark.skipif(
    auto_matrix is None, reason="auto-matrix is required"
)  # noqa
@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_real3d_rec_reader():
    rec_file = os.path.join(
        HAT_BUCKET_PATH,
        "unit_test_data/J5FSD/detection_3D/pandar64_distorted/rec/eval/eval_0821.rec",  # noqa
    )  # noqa
    reader = Real3DRecReader(rec_file)
    print(len(reader))
    for (image, label) in reader:
        print(image.shape)
        print(label.keys())
        break


@pytest.mark.skipif(
    auto_matrix is None, reason="auto-matrix is required"
)  # noqa
@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_real3d_dataset_rec():
    rec_file = os.path.join(
        HAT_BUCKET_PATH,
        "unit_test_data/J5FSD/detection_3D/pandar64_distorted/rec/eval/eval_0821.rec",  # noqa
    )  # noqa
    category_id_dict = Real3DDataset.get_category_id_dict(3)
    cfg = dict(
        type="Real3DDatasetRec",
        paths=[rec_file, rec_file],
        num_classes=3,
        transforms=[
            dict(type="ImageTransform", size=(960, 512)),
            dict(type="ImageToTensor", from_numpy=True),
            dict(type="ConvertLayout", hwc2chw=True),
            dict(type="ImageBgrToYuv444", rgb_input=False),
            dict(type="ImageNormalize", mean=128.0, std=128.0),
            dict(
                type="Real3dTargetGenerator",
                num_classes=3,
                focal_length_default=1548.9354248046875,
                input_size=(960, 512),
                category_id_dict=category_id_dict,
                origin_image_shape=(1080, 1920),
                down_stride=4,
                max_objs=100,
            ),
            dict(type="ParseDataReal3DMultitask"),
        ],
        num_dist=8,
    )

    # test function construction
    dataset = build_from_registry(cfg)

    # test function __len__
    length_dataset = len(dataset)
    assert length_dataset == 28000

    # test function __getitem__
    assert "image_name" in dataset[0]
    assert "image_height" in dataset[0]
    assert "image_width" in dataset[0]
    assert "imgs" in dataset[0]
    assert "image_id" in dataset[0]
    assert "ignore_mask" in dataset[0]
    assert "image_transform" in dataset[0]
    assert "target" in dataset[0]
    assert dataset[0]["image_height"] == 1080
    assert dataset[0]["image_width"] == 1920


@pytest.mark.skipif(
    auto_matrix is None, reason="auto-matrix is required"
)  # noqa
@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_real3d_dataset_rec_sign_light_cone():
    rec_file = os.path.join(
        HAT_BUCKET_PATH,
        "unit_test_data/J5FSD/detection_3D/sign_light_cone_real3d/rec/data_H3165_20220422_246944__front_0820__.rec",  # noqa
    )  # noqa
    category_id_dict = {
        100: -99,  # Unknown -> ignore
        101: 0,  # TrafficSign -> TrafficSign
        102: 1,  # TrafficLight -> TrafficLight
        103: -99,  # TrafficLightBulb -> ignore
        104: -99,  # LaneMarking -> ignore
        105: -99,  # StopLine  -> ignore
        106: -99,  # SpeedBump -> ignore
        107: -99,  # Pole -> ignore
        108: -99,  # CrossWalk -> ignore
        109: -99,  # Zone -> ignore
        110: -99,  # ParkingSlot -> ignore
        111: 2,  # TrafficCone -> TrafficCone
    }  # 'Dontcare' -> Ignore
    cfg = dict(
        type="Real3DDatasetRec",
        paths=[rec_file, rec_file],
        num_classes=3,
        transforms=[
            dict(type="ImageTransform", size=(960, 512)),
            dict(type="ImageToTensor", from_numpy=True),
            dict(type="ConvertLayout", hwc2chw=True),
            dict(type="ImageBgrToYuv444", rgb_input=False),
            dict(type="ImageNormalize", mean=128.0, std=128.0),
            dict(
                type="Real3dTargetGenerator",
                num_classes=3,
                focal_length_default=2411.0,
                input_size=(960, 512),
                category_id_dict=category_id_dict,
                origin_image_shape=(2160, 3840),
                down_stride=4,
                max_objs=100,
            ),
            dict(type="ParseDataReal3DMultitask"),
        ],
        num_dist=8,
    )

    # test function construction
    dataset = build_from_registry(cfg)

    # test function __len__
    length_dataset = len(dataset)
    assert length_dataset == 4857 * 2

    # test function __getitem__
    assert "image_name" in dataset[0]
    assert "image_height" in dataset[0]
    assert "image_width" in dataset[0]
    assert "imgs" in dataset[0]
    assert "image_id" in dataset[0]
    assert "ignore_mask" in dataset[0]
    assert "image_transform" in dataset[0]
    assert "target" in dataset[0]
    assert dataset[0]["image_height"] == 2160
    assert dataset[0]["image_width"] == 3840


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_2d_detection_pb_rec():
    rec_file = os.path.join(
        HAT_BUCKET_PATH,
        "unit_test_data/mono/face.rec",  # noqa
    )  # noqa
    cfg = dict(
        type="PbRec2DDataset",
        paths=[rec_file, rec_file],
        transforms=[
            dict(
                type="Resize",
                img_scale=(540, 960),
                keep_ratio=False,
            ),
            dict(
                type="PresetCrop",
                crop_top=220,
                crop_bottom=128,
                crop_left=0,
                crop_right=0,
            ),
            dict(type="RandomFlip", px=0.5),
            dict(
                type="CenterNetTargetGenerator",
                input_size=(960, 192),
                head_channels=dict(hm=1, wh=2),
                down_stride=4,
            ),
        ],
    )

    # test function construction
    dataset = build_from_registry(cfg)

    # test function __len__
    length_dataset = len(dataset)
    assert length_dataset == 63140

    # test function __getitem__
    assert "img" in dataset[0]
    assert "labels" in dataset[0]
    assert "hm" in dataset[0]["labels"]
    assert "wh" in dataset[0]["labels"]
    assert "ignore_mask" in dataset[0]["labels"]
    assert dataset[0]["img"].shape == (3, 192, 960)
    assert dataset[0]["labels"]["hm"].shape == (1, 48, 240)
    assert dataset[0]["labels"]["wh"].shape == (2, 48, 240)
    assert dataset[0]["labels"]["ignore_mask"].shape == (1, 48, 240)
