import os
import pickle

import numpy as np
import pytest

from hat.data.transforms.transform_3d import AffineTransform, BBoxGenerate
from tests import HAT_BUCKET_PATH

try:
    import pycocotools
except ImportError:
    pycocotools = None

root = f"{HAT_BUCKET_PATH}/users/zihan.qiu/jenkins_test_transform_data"


def test_3d_bbox_generate_transforms():
    input_data_path = os.path.join(root, "origin_data_dense.pickle")
    with open(input_data_path, "rb") as f:
        data = pickle.load(f)
    crop_roi_3d = (0, 100, 1920, 1124)
    transform = BBoxGenerate(
        use_bbox2d=True,
        filtered_name="__front_0820__",
        undistort_2dcenter=True,
        crop_roi=crop_roi_3d,
    )

    data = transform(data)
    assert isinstance(data, dict)
    assert data["img"].shape == (1280, 1920, 3)
    assert data["bboxes"].shape == (1, 4)
    assert data["dims"].shape == (1, 3)
    assert data["cls_ids"].shape == (1,)


def test_3d_bbox_generate_with_camera_standardization_transforms():
    input_data_path = os.path.join(root, "bbox_generate_camera_stand.pkl")
    with open(input_data_path, "rb") as f:
        data = pickle.load(f)
    transform = BBoxGenerate(
        use_bbox2d=True,
        filtered_name="__front_0820__",
        undistort_2dcenter=True,
    )

    data = transform(data)
    assert isinstance(data, dict)
    assert data["img"].shape == (1280, 1920, 3)
    assert data["bboxes"].shape == (1, 4)
    assert data["dims"].shape == (1, 3)
    assert data["cls_ids"].shape == (1,)
    assert data["uv_map"].shape == (640, 960, 2)


@pytest.mark.skipif(pycocotools is None, reason="need pycocotools")
def test_3d_affine_transforms():
    input_data_path = os.path.join(root, "transform_bboxgenerate_output.pkl")
    with open(input_data_path, "rb") as f:
        data = pickle.load(f)
    crop_roi_3d = (0, 100, 1920, 1124)
    input_wh = (960, 512)
    transform = AffineTransform(
        input_wh=input_wh,
        keep_res=False,
        shift=np.array([0, 0], dtype=np.float32),
        max_objs=100,
        keep_aspect_ratio=False,
        crop_roi=crop_roi_3d,
    )

    data = transform(data)
    assert isinstance(data, dict)
    assert data["img"].shape == (3, 512, 960)
    assert data["img_wh"][0] == data["img"].shape[2]
    assert data["img_wh"][1] == data["img"].shape[1]
    assert np.all(data["im_hw"] == data["img"].shape[1:])
    assert data["trans_matrix"].shape == (2, 3)


@pytest.mark.skipif(pycocotools is None, reason="need pycocotools")
def test_3d_affine_transforms_empty_bbox():
    input_data_path = os.path.join(root, "affine_empty_bbox_input.pkl")
    with open(input_data_path, "rb") as f:
        data = pickle.load(f)
    crop_roi_3d = (0, 100, 1920, 1124)
    input_wh = (960, 512)
    transform = AffineTransform(
        input_wh=input_wh,
        keep_res=False,
        shift=np.array([0, 0], dtype=np.float32),
        max_objs=100,
        keep_aspect_ratio=False,
        crop_roi=crop_roi_3d,
    )

    data = transform(data)
    assert isinstance(data, dict)
    assert data["img"].shape == (3, 512, 960)
    assert data["img_wh"][0] == data["img"].shape[2]
    assert data["img_wh"][1] == data["img"].shape[1]
    assert np.all(data["im_hw"] == data["img"].shape[1:])
    assert data["trans_matrix"].shape == (2, 3)
