# Copyright (c) Horizon Robotics. All rights reserved.
from copy import deepcopy

import numpy as np
import pytest
from cv2 import flip

try:
    import albumentations  # noqa F401
except ImportError:
    AlBU_AVAILABLE = False
else:
    AlBU_AVAILABLE = True

from hat.registry import build_from_registry
from tests.utils import gen_fake_transforms_data


@pytest.mark.parametrize(
    ["rgb_data", "nc"],
    [
        pytest.param(
            False,
            3,
        )
    ],
)
def test_gaze_yuv_transform(rgb_data, nc):
    h = 200
    w = 200
    data = gen_fake_transforms_data(w, h, layout="hwc")
    config = dict(type="GazeYUVTransform", rgb_data=rgb_data, nc=nc)

    gaze_yuv_transform = build_from_registry(config)
    data = gaze_yuv_transform(data)
    assert data["img"].shape == (h, w, 3)


@pytest.mark.parametrize(
    ["px", "py", "rotate_3d_augm"],
    [
        pytest.param(1.0, 0.0, True),
        pytest.param(1.0, 0.0, False),
    ],
)
def test_random_flip(px, py, rotate_3d_augm):
    h = 200
    w = 200
    data = gen_fake_transforms_data(w, h, layout="hwc")
    config = dict(type="RandomFlip", px=px, py=py)
    if rotate_3d_augm:
        gaze_label = {
            "gt_head_pose": np.random.randn(3),
            "gt_gaze": np.random.randn(4),
            "gt_eye_ldmk": np.random.randn(42, 2),
            "gt_eye_bbox": np.random.randn(4),
            "gt_loss_weight": 1.0,
            "gt_face_ldmks": np.random.randn(68, 2),
            "intrinsics_K": np.random.randn(3, 3),
            "origin_image_shape": [h, w],
        }
        data["gaze_label"] = gaze_label
    else:
        data["horizon_img"] = np.random.randn(*data["img"].shape)
        data["vertical_img"] = np.random.randn(*data["img"].shape)
        data["mirror_horizon_img"] = np.random.randn(*data["img"].shape)
        data["mirror_vertical_img"] = np.random.randn(*data["img"].shape)
        gaze_label = {
            "gt_gaze": np.random.randn(4),
            "gt_normed_eye_ldmk": np.random.randn(42, 2),
            "gt_head_pose": np.random.randn(3),
            "gt_gazemap": np.random.randn(200, 200, 3),
        }
        data["gaze_label"] = gaze_label

    random_flip = build_from_registry(config)
    ori_data = deepcopy(data)
    data = random_flip(data)
    gt_gaze = ori_data["gaze_label"]["gt_gaze"]
    gt_gaze = np.array([gt_gaze[2], -gt_gaze[3], gt_gaze[0], -gt_gaze[1]])
    assert (data["gaze_label"]["gt_gaze"] == gt_gaze).all()
    assert (
        data["gaze_label"]["gt_head_pose"]
        == ori_data["gaze_label"]["gt_head_pose"] * [1, -1, -1]
    ).all()
    if not rotate_3d_augm:
        assert (
            ori_data["gaze_label"]["gt_gazemap"][:, ::-1, :]
            == data["gaze_label"]["gt_gazemap"]
        ).all()
    assert (data["img"] == flip(ori_data["img"], 1)).all()


@pytest.mark.parametrize(
    ["size", "area", "ratio", "prob", "is_train"],
    [
        pytest.param(
            (192, 320),
            (0.08, 1.0),
            (3.0 / 4.0, 4.0 / 3.0),
            1.0,
            True,
        ),
        pytest.param(
            (192, 320),
            (0.08, 1.0),
            (3.0 / 4.0, 4.0 / 3.0),
            1.0,
            False,
        ),
    ],
)
def test_gaze_random_crop_resize(size, area, ratio, prob, is_train):
    h = 200
    w = 200
    data = gen_fake_transforms_data(w, h, layout="hwc")
    data["horizon_img"] = np.random.randn(w, h)
    data["vertical_img"] = np.random.randn(w, h)
    data["gaze_label"] = {
        "gt_normed_eye_ldmk": np.random.randn(42, 2),
        "gt_gazemap": np.random.randn(200, 200, 3),
    }
    config = dict(
        type="GazeRandomCropWoResize",
        size=size,
        area=area,
        ratio=ratio,
        prob=prob,
        is_train=is_train,
    )

    gaze_random_crop_resize = build_from_registry(config)
    data = gaze_random_crop_resize(data)
    assert isinstance(data["img"], np.ndarray)
    assert data["img"].shape[2] == 3


@pytest.mark.parametrize(
    ["minimum", "maximum"],
    [
        pytest.param(
            0.0,
            255.0,
        )
    ],
)
def test_clip(minimum, maximum):
    h = 200
    w = 200
    data = gen_fake_transforms_data(w, h, layout="hwc")
    config = dict(type="Clip", minimum=minimum, maximum=maximum)

    clip = build_from_registry(config)
    data = clip(data)
    assert data["img"].min() >= minimum
    assert data["img"].max() <= maximum


@pytest.mark.skipif(not AlBU_AVAILABLE, reason="albu is required")
@pytest.mark.parametrize(
    ["brightness", "contrast", "saturation", "hue"],
    [pytest.param(0.5, (0.5, 1.5), (0.5, 1.5), 0.1)],
)
def test_random_color_jitter(brightness, contrast, saturation, hue):
    h = 200
    w = 200
    data = gen_fake_transforms_data(w, h, layout="hwc")
    config = dict(
        type="RandomColorJitter",
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

    random_color_jitter = build_from_registry(config)
    ori_data = deepcopy(data)
    data = random_color_jitter(data)
    assert data["img"].shape == ori_data["img"].shape


if __name__ == "__main__":
    pytest.main(["-s", __file__])
