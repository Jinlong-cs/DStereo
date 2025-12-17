# Copyright (c) Horizon Robotics. All rights reserved.
import os

import pytest

from hat.data.datasets.densebox_dataset_with_roilist import (  # noqa
    DenseboxWithRoilistDataset,
    LegacyDenseBoxWithRoilistImageRecordDataset,
)
from hat.registry import build_from_registry
from tests import HAT_BUCKET_PATH

ADASMINI_ROOT = os.path.join(
    HAT_BUCKET_PATH, "unit_test_data/mono_person_detection_2pe"
)
ADASMINI_AVAILABLE = os.path.exists(ADASMINI_ROOT)


def test_densebox_dataset():
    rec_file = "person_mini.rec"
    anno_file = "person_mini.anno.pb_rec"
    roilist_file = "person_mini.roilist.pb_rec"
    config = dict(
        type="LegacyDenseBoxWithRoilistImageRecordDataset",
        rec_path=os.path.join(ADASMINI_ROOT, rec_file),
        anno_path=os.path.join(ADASMINI_ROOT, anno_file),
        roi_list_path=os.path.join(ADASMINI_ROOT, roilist_file),
    )
    dataset = build_from_registry(config)
    for idx in range(0, len(dataset)):
        data = dataset[idx]
        assert isinstance(data, dict)
        assert "img" in data
        assert "anno" in data
        assert "roi_list" in data
        break


def test_densebox_2pe_with_roilist_dataset():
    rec_file = "person_mini.rec"
    anno_file = "person_mini.anno.pb_rec"
    roilist_file = "person_mini.roilist.pb_rec"
    config = dict(
        type="DenseboxWithRoilistDataset",
        data_path=os.path.join(ADASMINI_ROOT, rec_file),
        anno_path=os.path.join(ADASMINI_ROOT, anno_file),
        roi_list_path=os.path.join(ADASMINI_ROOT, roilist_file),
        to_rgb=True,
        task_type="detection",
    )
    dataset = build_from_registry(config)
    for idx in range(0, len(dataset)):
        data = dataset[idx]
        assert isinstance(data, dict)
        assert "img" in data
        assert "anno" in data
        assert "roi_list" in data
        break


if __name__ == "__main__":
    pytest.main(["-v", __file__])
