import os

import pytest

try:
    import auto_matrix
except ImportError:
    auto_matrix = None

from hat.data.datasets.auto2d_anno_dataset import (
    Auto2dFromRawJson,
    DatasetFromAnno,
    RCNNDetDatasetFromAnno,
)
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH

DETECTION_REC = os.path.join(
    HAT_BUCKET_PATH, "data/rec_data/sp_sod/sod_ut/val.anno.pb_rec"
)
RCNN_KPS_REC = os.path.join(HAT_BUCKET_PATH, "data/sp_vdvru/kps_ut/val.pb_rec")


@pytest.mark.skipif(
    auto_matrix is None,
    reason="auto_matrix is required.",
)  # noqa
@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="hdfs bucket is required")
@pytest.mark.parametrize(
    ["anno_path", "task_type"],
    [
        pytest.param(DETECTION_REC, "detection"),
        pytest.param(RCNN_KPS_REC, "rcnn_kps"),
    ],
)
def test_dataset_without_anno(anno_path, task_type):
    dataset = DatasetFromAnno(
        anno_path,
        class_id_map={},
        transforms=None,
        task_type=task_type,
        buf_only=False,
        to_rgb=True,
        return_orig_img=True,
        ignore_hard=True,
        use_ignore=True,
        pad_topk=100,
    )
    max_iter = 100
    for i, data in enumerate(dataset):
        if i > max_iter:
            break
        assert isinstance(data, dict)
        if task_type != "rcnn_kps":
            assert "gt_bboxes" in data or "gt_boxes" in data
            assert "gt_classes" in data
            assert "ig_bboxes" in data
            assert len(data["gt_bboxes"]) >= 100
        assert "ori_image" in data


RCNN_DET_REC = os.path.join(
    HAT_BUCKET_PATH, "data/sp_vdvru/roi_det_ut/val.anno.pb_rec"
)


@pytest.mark.skipif(
    auto_matrix is None,
    reason="auto_matrix is required.",
)  # noqa
@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="hdfs bucket is required")
@pytest.mark.parametrize(
    ["anno_path"],
    [
        pytest.param(RCNN_DET_REC),
    ],
)
def test_rcnn_det_dataset_without_anno(anno_path):
    dataset = RCNNDetDatasetFromAnno(
        anno_path,
        class_id_map={},
        transforms=None,
        buf_only=True,
        to_rgb=True,
        return_orig_img=True,
        ignore_hard=True,
        use_ignore=True,
        pad_topk=100,
        lt_point_id=0,
        rb_point_id=2,
        parent_lt_point_id=10,
        parent_rb_point_id=12,
    )
    max_iter = 100
    for i, data in enumerate(dataset):
        if i > max_iter:
            break
        assert isinstance(data, dict)
        assert "gt_bboxes" in data or "gt_boxes" in data
        assert "ori_image" in data


DETECTION_JSON = os.path.join(
    HAT_BUCKET_PATH, "data/sp_vdvru/det_ut/person_positive/data.json"
)
DETECTION_EMPTY_JSON = os.path.join(
    HAT_BUCKET_PATH, "data/sp_vdvru/det_ut/person_empty/data.json"
)


@pytest.mark.skipif(
    auto_matrix is None,
    reason="auto_matrix is required.",
)  # noqa
@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="hdfs bucket is required")
@pytest.mark.parametrize(
    ["anno_path", "task_type"],
    [
        pytest.param(DETECTION_JSON, "detection"),
        pytest.param(DETECTION_EMPTY_JSON, "detection"),
    ],
)
def test_dataset_from_raw_json(anno_path, task_type):
    dataset = Auto2dFromRawJson(
        anno_path,
        transforms=None,
        task_type=task_type,
        buf_only=True,
        to_rgb=True,
        return_orig_img=True,
    )
    max_iter = 100
    for i, data in enumerate(dataset):
        if i > max_iter:
            break
        assert isinstance(data, dict)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
