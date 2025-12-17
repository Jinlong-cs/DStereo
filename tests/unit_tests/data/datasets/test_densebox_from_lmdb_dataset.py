import os

import pytest

from hat.registry import build_from_registry
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH


@pytest.mark.skipif(
    not HAT_BUCKET_EXISTS,
    reason="requiring HAT_BUCKET bucket",
)
def test_densebox_from_lmdb_dataset():
    # test detection data
    det_root = f"{HAT_BUCKET_PATH}/users/zihan.qiu/jenkins_test_packed_data/cyclist_detection"  # noqa
    det_config = dict(
        type="DenseboxFromLMDBDataset",
        idx_path=os.path.join(det_root, "idx"),
        img_path=os.path.join(det_root, "img"),
        anno_path=os.path.join(det_root, "anno"),
        decode_img=True,
        task_type="detection",
        class_id=7,
        category=0,
        to_rgb=True,
        ignore_hard=False,
        use_ignore=True,
        transform_ignore_bboxes=True,
        return_orig_img=False,
        return_orig_gt_seg=False,
        remove_det_duplicate=True,
        abandon_other_category=True,
        transforms=[dict(type="Resize", img_scale=(1000, 2000))],
    )
    det_dataset = build_from_registry(det_config)
    det_data = det_dataset[0]
    assert isinstance(det_data, dict)
    assert "img_name" in det_data
    assert "img_height" in det_data
    assert "img_width" in det_data
    assert "img_id" in det_data
    assert "img" in det_data
    assert "color_space" in det_data
    assert "layout" in det_data
    assert "img_shape" in det_data
    assert "gt_bboxes" in det_data
    assert "gt_classes" in det_data
    assert "ig_bboxes" in det_data
    assert det_data["img_shape"] == (1000, 1600, 3)

    # test semantic parsing data
    parsing_data_root = f"{HAT_BUCKET_PATH}/users/zihan.qiu/jenkins_test_packed_data/semantic_parsing"  # noqa
    parsing_data_config = dict(
        type="DenseboxFromLMDBDataset",
        idx_path=os.path.join(parsing_data_root, "idx"),
        img_path=os.path.join(parsing_data_root, "img"),
        anno_path=os.path.join(parsing_data_root, "anno"),
        decode_img=True,
        task_type="segmentation",
        to_rgb=True,
        return_orig_gt_seg=False,
        transforms=[dict(type="Resize", img_scale=(1000, 2000))],
    )
    parsing_dataset = build_from_registry(parsing_data_config)
    parsing_data = parsing_dataset[0]
    assert isinstance(parsing_data, dict)
    assert "img_name" in parsing_data
    assert "img_height" in parsing_data
    assert "img_width" in parsing_data
    assert "img_id" in parsing_data
    assert "img" in parsing_data
    assert "color_space" in parsing_data
    assert "layout" in parsing_data
    assert "img_shape" in parsing_data
    assert "gt_seg" in parsing_data
    assert parsing_data["img_shape"] == (1000, 1500, 3)
    assert parsing_data["gt_seg"].shape == parsing_data["img_shape"][:2]
