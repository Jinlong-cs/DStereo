# Copyright (c) Horizon Robotics. All rights reserved.

import os
import pickle
import shutil
import time

import pytest

from hat.data.datasets.lmdb_auto2d import (
    Auto2D2lmdb,
    Auto2dFromLMDB,
    AutoDetPacker,
    AutoSegPacker,
)
from hat.utils.package_helper import check_packages_available
from tests import BasicAlgorithm_BUCKET_EXISTS, BasicAlgorithm_BUCKET_PATH

det_data_path = os.path.join(
    BasicAlgorithm_BUCKET_PATH,
    "J5_FSD/2dvision/adasmini_v2/vehicle_rear/detection/" "720_v1/",
)
seg_data_path = os.path.join(
    BasicAlgorithm_BUCKET_PATH, "J5_FSD/2dvision/adasmini_v2/parsing/720_v1/"
)
lmdb_path = os.path.join(
    BasicAlgorithm_BUCKET_PATH,
    "J5_FSD/2dvision/HDLT/lmdb/detection/vehicle_rear/"
    "adasmini_vehicle_rear_val",
)

raw_det_data_path = os.path.join(
    BasicAlgorithm_BUCKET_PATH,
    "Neural_ISP/unit_test_data/" "dataset_lmdb_auto2d/test_data/det/",
)
raw_det_lmdb_path = os.path.join(
    BasicAlgorithm_BUCKET_PATH,
    "Neural_ISP/unit_test_data/" "dataset_lmdb_auto2d/test_data/det/lmdb/eval",
)

raw_seg_data_path = os.path.join(
    BasicAlgorithm_BUCKET_PATH,
    "Neural_ISP/unit_test_data/" "dataset_lmdb_auto2d/test_data/seg/",
)
raw_seg_lmdb_path = os.path.join(
    BasicAlgorithm_BUCKET_PATH,
    "Neural_ISP/unit_test_data/" "dataset_lmdb_auto2d/test_data/seg/lmdb/eval",
)


@pytest.mark.skipif(
    not BasicAlgorithm_BUCKET_EXISTS, reason="path does not exist"
)
@pytest.mark.skipif(
    not check_packages_available("pycocotools", raise_exception=False),
    reason="need pycocotools",
)
def test_auto_det_packer():
    needed_meta_keys = [
        "file_name",
        "height",
        "width",
        "id",
        "gt_bboxes",
        "gt_classes",
        "img",
        "bit_nums_upper",
        "bit_nums_lower",
        "mipi",
        "channels",
        "raw_pattern",
        "pre_offset",
    ]
    dataset = AutoDetPacker(raw_det_data_path, "eval")
    assert len(dataset) == 10
    assert isinstance(dataset[0], dict)
    for key in needed_meta_keys:
        assert key in dataset[0].keys()


@pytest.mark.skipif(
    not BasicAlgorithm_BUCKET_EXISTS, reason="path does not exist"
)
@pytest.mark.skipif(
    not check_packages_available("pycocotools", raise_exception=False),
    reason="need pycocotools",
)
def test_auto_seg_packer():
    needed_meta_keys = [
        "file_name",
        "height",
        "width",
        "id",
        "img",
        "gt_seg",
        "bit_nums_upper",
        "bit_nums_lower",
        "mipi",
        "channels",
        "raw_pattern",
        "pre_offset",
    ]
    dataset = AutoSegPacker(raw_seg_data_path, "eval")
    assert len(dataset) == 10
    assert isinstance(dataset[0], dict)
    for key in needed_meta_keys:
        assert key in dataset[0].keys()


@pytest.mark.skipif(
    not BasicAlgorithm_BUCKET_EXISTS, reason="path does not exist"
)
def test_auto2d_2lmdb():
    try:
        time_stamp = time.strftime(
            "%Y%m%d%H%M%S", time.localtime(int(time.time()))
        )

        Auto2D2lmdb(
            "./test_auto2d_2lmdb/{}".format(time_stamp),
            raw_det_data_path,
            "eval",
            "det",
            0,
        )
        if os.path.exists("./test_auto2d_2lmdb"):
            shutil.rmtree("./test_auto2d_2lmdb")

        Auto2D2lmdb(
            "./test_auto2d_2lmdb/{}".format(time_stamp),
            raw_seg_data_path,
            "eval",
            "seg",
            0,
        )
        if os.path.exists("./test_auto2d_2lmdb"):
            shutil.rmtree("./test_auto2d_2lmdb")

    except BaseException:
        raise AssertionError()


@pytest.mark.skipif(
    not BasicAlgorithm_BUCKET_EXISTS, reason="path does not exist"
)
def test_auto2d_from_lmdb():
    try:
        dataset = Auto2dFromLMDB(
            raw_det_lmdb_path, None, 10, return_orig_img=True
        )
        dataset = pickle.dumps(dataset)
        dataset = pickle.loads(dataset)
        assert len(dataset) == 10
        assert isinstance(dataset[0], dict)
        needed_meta_keys = [
            "img_name",
            "img_height",
            "img_width",
            "img_id",
            "img",
            "color_space",
            "layout",
            "bit_nums_upper",
            "bit_nums_lower",
            "raw_pattern",
        ]
        for key in needed_meta_keys:
            assert key in dataset[0].keys()

        dataset = Auto2dFromLMDB(
            raw_seg_lmdb_path, None, 10, return_orig_img=True
        )
        dataset = pickle.dumps(dataset)
        dataset = pickle.loads(dataset)
        assert len(dataset) == 10
        assert isinstance(dataset[0], dict)
        needed_meta_keys = [
            "img_name",
            "img_height",
            "img_width",
            "img_id",
            "img",
            "color_space",
            "layout",
            "bit_nums_upper",
            "bit_nums_lower",
            "raw_pattern",
            "gt_seg",
        ]
        for key in needed_meta_keys:
            assert key in dataset[0].keys()
    except BaseException:
        raise AssertionError()
