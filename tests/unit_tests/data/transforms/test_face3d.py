# Copyright (c) Horizon Robotics. All rights reserved.
import numpy as np
import pytest

from hat.data.transforms.face3d import (
    BboxEncodingCLIFF,
    CropRoIJitter,
    ImageCLAHE,
    PositionEncoding,
    RandomRotateCrop,
    VirtualCameraNorm,
)

try:
    import kornia
except ImportError:
    kornia = None


def gen_fake_data(bbox, img_shape):
    data = {}
    h, w, c = img_shape
    num_ldmk = 10
    data["img"] = np.random.randint(0, 255, img_shape, dtype=np.uint8)
    data["gt_bboxes"] = np.array(bbox)
    data["gt_ldmk"] = np.random.uniform(0, 1, (num_ldmk, 3))
    data["gt_ldmk"][:, 0] *= w
    data["gt_ldmk"][:, 1] *= h
    data["gt_ldmk"][:, 2] = 1
    data["gt_mask"] = np.random.choice(np.arange(2), (h, w)).astype(np.uint8)
    data["gt_img"] = data["img"].copy()
    data["img_shape"] = data["img"].shape
    focal_lenth = np.random.randint(500, 2000)
    data["intrinsic"] = np.array(
        [[focal_lenth, 0, w / 2], [0, focal_lenth, h / 2], [0, 0, 1]]
    )
    data["distortion"] = np.random.uniform(0, 1, (1, 5))
    data["roi_offset"] = np.array([[0, 0]])
    data["raw_img_shape"] = data["img"].shape
    data["save_crop"] = False
    data["gt_pupil_ellipse_param"] = np.array([0.5, 0.5, 0.5, 0.6, 0.5])
    return data


@pytest.mark.skipif(kornia is None, reason="need kornia")
@pytest.mark.parametrize(
    ["bbox", "img_shape", "net_input_size", "net_target_size", "norm_method"],
    [
        pytest.param(
            [50, 100, 200, 200],
            (256, 256, 3),
            (128, 128),
            (128, 128),
            "longside_square",
        )
    ],
)
def test_random_rotate_crop(
    bbox, img_shape, net_input_size, net_target_size, norm_method
):
    data = gen_fake_data(bbox, img_shape)
    rotate_crop = RandomRotateCrop(
        net_input_size=net_input_size,
        rot_prob=1.0,
        rot_angle_range=30,
        center_shift_prob=1.0,
        center_shift_range=0.01,
        norm_ratio=1.25,
        norm_method=norm_method,
        norm_jitter_range=0.25,
        net_target_size=net_target_size,
        base_len=1.0,
    )
    result = rotate_crop(data)
    assert result["img"].shape[0] == net_input_size[1]
    assert result["img"].shape[1] == net_input_size[0]
    assert result["gt_mask"].shape[1] == net_target_size[1]
    assert result["gt_mask"].shape[2] == net_target_size[0]
    assert result["gt_img"].shape[1] == net_target_size[1]
    assert result["gt_img"].shape[2] == net_target_size[0]
    assert result["gt_pupil_ellipse_param"].shape == (5,)


@pytest.mark.skipif(kornia is None, reason="need kornia")
@pytest.mark.parametrize(
    ["bbox", "img_shape", "net_input_size", "net_target_size", "use_distort"],
    [
        pytest.param(
            [100, 150, 200, 300], (400, 400, 3), (128, 128), (160, 160), True
        ),
        pytest.param(
            [100, 150, 200, 300], (400, 400, 3), (128, 128), (160, 160), False
        ),
    ],
)
def test_virtual_camera_norm(
    bbox, img_shape, net_input_size, net_target_size, use_distort
):
    data = gen_fake_data(bbox, img_shape)
    norm = VirtualCameraNorm(
        net_input_size,
        net_target_size,
        norm_ratio=1.2,
        norm_method="longside_square",
        use_dist=use_distort,
        virtual_intrinsic=None,
    )
    result = norm(data)
    assert result["img"].shape[0] == net_input_size[1]
    assert result["img"].shape[1] == net_input_size[0]
    assert result["gt_mask"].shape[1] == net_target_size[1]
    assert result["gt_mask"].shape[2] == net_target_size[0]
    assert result["gt_img"].shape[1] == net_target_size[1]
    assert result["gt_img"].shape[2] == net_target_size[0]


@pytest.mark.parametrize(
    ["img_shape", "concat_img"],
    [[(128, 160, 3), True], [(256, 192, 1), False]],
)
def test_postition_encoding(img_shape, concat_img):
    h, w, c = img_shape
    net_input_size = (w, h)
    data = gen_fake_data([100, 200, 200, 300], img_shape)
    data["virtual_bbox"] = data["gt_bboxes"].copy()
    position_encoding = PositionEncoding(net_input_size, True, concat_img)
    result = position_encoding(data)
    assert result["pos_map_h"].shape == (h, w)
    assert result["pos_map_v"].shape == (h, w)
    if concat_img:
        assert result["img"].shape == (h, w, c + 2)
    else:
        assert result["img"].shape == (h, w, c)


def test_bbox_encoding_cliff():
    add_cliff_info = BboxEncodingCLIFF()
    bbox = [100, 150, 200, 200]
    data = gen_fake_data(bbox, (300, 300, 3))
    data["input_bbox"] = bbox
    result = add_cliff_info(data)
    assert result["cliff_info"].shape == (4, 1, 1)


def test_imgclahe():
    imgclahe = ImageCLAHE()
    bbox = [100, 150, 200, 200]
    data = gen_fake_data(bbox, (300, 300, 3))
    res = imgclahe(data)
    assert res["img"].shape == (300, 300, 3)
    assert res["gt_img"].shape == (300, 300, 3)


def test_crop_roi_jitter():
    roi_jitter = CropRoIJitter(
        jitter_prob=1.0, exp_ratio=1.1, exp_jitter=0.2, center_shift=0.2
    )
    bbox = [100, 150, 150, 199]
    data = gen_fake_data(bbox, (300, 300, 3))
    error = 0
    count = 0
    while error < 1e-6:
        res = roi_jitter(data)
        error += (abs(res["gt_bboxes"] - np.array(bbox))).sum()
        count += 1
        if count > 50:
            assert 0
