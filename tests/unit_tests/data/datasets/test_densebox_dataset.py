# Copyright (c) Horizon Robotics. All rights reserved.
import os

import pytest

from hat.data.datasets.densebox_dataset import LegacyDenseBoxImageRecordDataset
from hat.registry import build_from_registry
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH

ADASMINI_ROOT = os.path.join(HAT_BUCKET_PATH, "unit_test_data/J5FSD/adasmini")
ADASMINI_AVAILABLE = os.path.exists(ADASMINI_ROOT)


@pytest.mark.skipif(
    LegacyDenseBoxImageRecordDataset is None,
    reason="auto_matrix or horizon_plugin_pytorch >= 1.0.0 is required.",
)  # noqa
@pytest.mark.skipif(not ADASMINI_AVAILABLE, reason="adas-mini is required")
@pytest.mark.parametrize(
    ["return_orig_img", "remove_det_duplicate"],
    [
        pytest.param(False, False),
        pytest.param(True, False),
        pytest.param(False, True),
        pytest.param(True, True),
    ],
)
def test_auto2d_dataset_det(return_orig_img, remove_det_duplicate):
    rec_file = "vehicle_rear/detection/720_v1/rec/val.rec"
    json_file = "vehicle_rear/detection/720_v1/rec/val.json"
    config = dict(
        type="DenseboxDataset",
        data_path=os.path.join(ADASMINI_ROOT, rec_file),
        anno_path=os.path.join(ADASMINI_ROOT, json_file),
        task_type="detection",
        with_img_buf=True,
        return_orig_img=return_orig_img,
        remove_det_duplicate=remove_det_duplicate,
    )
    dataset = build_from_registry(config)
    # datset_len = len(dataset)
    for data in dataset:
        assert isinstance(data, dict)
        assert "gt_bboxes" in data
        assert "img_buf" in data
        assert "gt_classes" in data
        if return_orig_img:
            assert "orig_img" in data
        else:
            assert "orig_img" not in data
        break


@pytest.mark.skipif(
    LegacyDenseBoxImageRecordDataset is None,
    reason="auto_matrix or horizon_plugin_pytorch >= 1.0.0 is required.",
)  # noqa
@pytest.mark.skipif(not ADASMINI_AVAILABLE, reason="adas-mini is required")
@pytest.mark.parametrize(
    ["return_orig_gt_seg"],
    [
        pytest.param(False),
        pytest.param(True),
    ],
)
def test_auto2d_dataset_seg(return_orig_gt_seg):
    rec_file = "parsing/720_v1/rec/val.rec"
    json_file = "parsing/720_v1/rec/val.json"
    config = dict(
        type="DenseboxDataset",
        data_path=os.path.join(ADASMINI_ROOT, rec_file),
        anno_path=os.path.join(ADASMINI_ROOT, json_file),
        task_type="segmentation",
        with_img_buf=True,
        return_orig_gt_seg=return_orig_gt_seg,
    )
    dataset = build_from_registry(config)
    # datset_len = len(dataset)
    for data in dataset:
        assert isinstance(data, dict)
        assert "gt_seg" in data
        assert "img_buf" in data
        assert data["gt_seg"].ndim == 2
        if return_orig_gt_seg:
            assert "orig_gt_seg" in data
        else:
            assert "orig_gt_seg" not in data
        break


@pytest.mark.skipif(
    LegacyDenseBoxImageRecordDataset is None,
    reason="auto_matrix or horizon_plugin_pytorch >= 1.0.0 is required.",
)  # noqa
@pytest.mark.skipif(not ADASMINI_AVAILABLE, reason="adas-mini is required")
def test_dataset_using_matrix_gluon():
    rec_file = "vehicle_rear/detection/720_v1/rec/val.rec"
    json_file = "vehicle_rear/detection/720_v1/rec/val.json"
    data_path = os.path.join(ADASMINI_ROOT, rec_file)
    anno_path = os.path.join(ADASMINI_ROOT, json_file)
    recs = [data_path, data_path]
    jsons = [anno_path, anno_path]

    config = dict(
        type="DenseboxDataset",
        data_path=recs[0],
        anno_path=jsons[0],
        task_type="detection",
    )
    dataset = build_from_registry(config)
    assert hasattr(dataset, "__len__")

    for data in dataset:
        assert isinstance(data, dict)
        print(data.keys())
        break


@pytest.mark.skipif(
    LegacyDenseBoxImageRecordDataset is None,
    reason="auto_matrix or horizon_plugin_pytorch >= 1.0.0 is required.",
)  # noqa
@pytest.mark.skipif(
    not HAT_BUCKET_EXISTS, reason="HDLTAlgorithm bucket is required"
)
def test_multiclass_densebox_dataset():
    rec_file = "data/rec_data/sp_sod/sod_ut/val_aframe_v1.6.2.rec"
    anno_file = "data/rec_data/sp_sod/sod_ut/val_aframe_v1.6.2.anno.pb_rec"
    rec_file = os.path.join(HAT_BUCKET_PATH, rec_file)
    anno_file = os.path.join(HAT_BUCKET_PATH, anno_file)

    config = dict(
        type="MulticlassDenseboxDataset",
        data_path=rec_file,
        anno_path=anno_file,
        class_id_map={
            9: 0,
            13: 1,
            14: 2,
        },
        task_type="detection",
        ignore_hard=True,
        use_ignore=True,
    )
    dataset = build_from_registry(config)
    assert hasattr(dataset, "__len__")

    max_iter = 100
    for i, data in enumerate(dataset):
        if i > max_iter:
            break
        assert isinstance(data, dict)
