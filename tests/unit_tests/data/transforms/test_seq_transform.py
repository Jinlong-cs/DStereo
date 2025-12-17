# Copyright (c) Horizon Robotics. All rights reserved.

from copy import deepcopy

import numpy as np
import pytest
import torch

from hat.data.transforms.seq_transform import (
    SeqAlbuImageOnlyTransform,
    SeqAugmentHSV,
    SeqBgrToYuv444,
    SeqNormalize,
    SeqPad,
    SeqRandomFlip,
    SeqRandomSizeCrop,
    SeqResize,
    SeqToFasterRCNNData,
    SeqToTensor,
)

try:
    import albumentations
except ImportError:
    albumentations = None


def gen_fake_seq_data(
    w, h, layout, fill_value=None, return_tensor=False, c=3, seq_length=2
):
    data_list = []
    for _ in range(seq_length):
        assert layout in ["hwc", "chw"]
        data = {}
        if layout == "hwc":
            img_shape = (h, w, c)
            flow_shape = (h, w, 2)
        else:
            img_shape = (c, h, w)
            flow_shape = (2, h, w)
        if fill_value is None:
            if return_tensor:
                img = torch.randint(0, 255, img_shape, dtype=torch.uint8)
                gt_seg = torch.randint(0, 10, (h, w), dtype=torch.uint8)
                gt_flow = torch.randn(flow_shape, dtype=torch.float32)
            else:
                img = np.random.randint(0, 255, img_shape, dtype=np.uint8)
                gt_seg = np.random.randint(0, 10, (h, w), dtype=np.uint8)
                gt_flow = np.random.random_sample(flow_shape)
        else:
            if return_tensor:
                img = torch.full(img_shape, fill_value, dtype=torch.uint8)
                gt_seg = torch.full((h, w), fill_value, dtype=torch.uint8)
                gt_flow = torch.full(
                    flow_shape, fill_value, dtype=torch.float32
                )
            else:
                img = np.full(img_shape, fill_value, dtype=np.uint8)
                gt_seg = np.full((h, w), fill_value, dtype=np.uint8)
                gt_flow = np.full(flow_shape, fill_value, dtype=np.float32)
        data["img"] = img
        data["img_height"] = h
        data["img_width"] = w
        data["img_shape"] = img.shape
        data["img_id"] = 0
        data["color_space"] = "bgr"
        data["layout"] = layout
        data["gt_seg"] = gt_seg
        data["gt_bboxes"] = np.array([[10, 10, 20, 20], [20, 20, 40, 40]])
        data["gt_classes"] = np.array([1, 0])
        data["gt_ids"] = np.array([0, 1])
        data["gt_flow"] = gt_flow
        data_list.append(data)
    seq_data = {"frame_data_list": data_list}
    return seq_data


def test_to_seg_totensor_bgrtoyuv444_normalize():
    data = gen_fake_seq_data(640, 352, layout="hwc")
    totensor = SeqToTensor(to_yuv=False)
    seqbgrtoyuv444 = SeqBgrToYuv444()
    seqnormalize = SeqNormalize(mean=128.0, std=128.0)
    data = totensor(data)
    data = seqbgrtoyuv444(data)
    data_result = seqnormalize(data)
    for data in data_result["frame_data_list"]:
        assert isinstance(data["img"], torch.Tensor)
        assert isinstance(data["gt_bboxes"], torch.Tensor)
        assert isinstance(data["gt_classes"], torch.Tensor)
        assert isinstance(data["gt_seg"], torch.Tensor)
        assert isinstance(data["gt_flow"], torch.Tensor)
        assert isinstance(data["gt_ids"], torch.Tensor)
        assert data["img"].shape == (3, 352, 640)


@pytest.mark.parametrize(
    ["w", "h", "c", "layout", "min_size", "max_size"],
    [
        pytest.param(500, 600, 3, "hwc", 300, 600),
        pytest.param(300, 400, 3, "chw", 100, 500),
    ],
)
def test_seq_random_size_crop(w, h, c, layout, min_size, max_size):
    data = gen_fake_seq_data(w, h, layout=layout, c=c)
    seq_random_croper = SeqRandomSizeCrop(min_size, max_size)
    data_result = seq_random_croper(data).copy()
    for croped_data in data_result["frame_data_list"]:
        if layout == "hwc":
            assert croped_data["img"].shape[0] <= max_size
            assert croped_data["img"].shape[1] <= max_size
            assert croped_data["img"].shape[0] >= min_size
            assert croped_data["img"].shape[1] >= min_size
        else:
            assert croped_data["img"].shape[1] <= max_size
            assert croped_data["img"].shape[2] <= max_size
            assert croped_data["img"].shape[1] >= min_size
            assert croped_data["img"].shape[2] >= min_size


@pytest.mark.parametrize(
    ["px", "py", "c"],
    [
        pytest.param(0, 1, 3),
        pytest.param(1, 0, 3),
        pytest.param(0, 1, 6),
        pytest.param(1, 0, 6),
    ],
)
def test_random_flip(px, py, c):
    seq_data = gen_fake_seq_data(300, 400, layout="hwc", c=c)
    src_data_list = deepcopy(seq_data)
    seq_fliper = SeqRandomFlip(px, py)
    flip_data_list = seq_fliper(seq_data)
    flip_flip_data_list = seq_fliper(deepcopy(flip_data_list))
    for flip_flip_data, src_data in zip(
        flip_flip_data_list["frame_data_list"],
        src_data_list["frame_data_list"],
    ):
        assert np.allclose(flip_flip_data["img"], src_data["img"])
        assert np.allclose(flip_flip_data["gt_bboxes"], src_data["gt_bboxes"])
        assert np.allclose(flip_flip_data["gt_seg"], src_data["gt_seg"])
        assert np.allclose(flip_flip_data["gt_flow"], src_data["gt_flow"])


def test_hsv_aug():
    seq_data = gen_fake_seq_data(512, 512, layout="hwc")
    seq_data = SeqAugmentHSV()(seq_data)
    for data in seq_data["frame_data_list"]:
        assert "img" in data


@pytest.mark.parametrize(
    [
        "origin_size",
        "max_scale",
        "keep_ratio",
        "multiscale_mode",
    ],
    [
        pytest.param(
            (3840, 2160),
            (1080, 2280),
            True,
            "max_size",
        ),
        pytest.param(
            (1280, 720),
            (1080, 2280),
            True,
            "max_size",
        ),
        pytest.param(
            (1920, 1080),
            (1080, 2280),
            True,
            "max_size",
        ),
        pytest.param(
            (3824, 2048),
            (1080, 2280),
            True,
            "max_size",
        ),
    ],
)
def test_seq_resize(origin_size, max_scale, keep_ratio, multiscale_mode):
    seq_data = gen_fake_seq_data(*origin_size, layout="hwc")
    resize_data = SeqResize(
        max_scale=max_scale,
        keep_ratio=keep_ratio,
        multiscale_mode=multiscale_mode,
    )(seq_data)
    for data in resize_data["frame_data_list"]:
        assert data["img"].shape[0] <= max_scale[0]
        assert data["img"].shape[1] <= max_scale[1]


@pytest.mark.parametrize(
    ["size", "divisor", "pad_val", "seg_pad_val", "target_h", "target_w"],
    [
        pytest.param((500, 600), 1, 100, 100, 500, 600),
        pytest.param((500, 600), 64, 100, 100, 512, 640),
    ],
)
def test_pad(size, divisor, pad_val, seg_pad_val, target_h, target_w):
    seq_data = gen_fake_seq_data(300, 400, layout="hwc", fill_value=100)
    target_data = gen_fake_seq_data(
        target_w, target_h, layout="hwc", fill_value=100
    )
    padder = SeqPad(size, divisor, pad_val, seg_pad_val)
    padded_data = padder(seq_data).copy()
    for data, target_data in zip(
        padded_data["frame_data_list"], target_data["frame_data_list"]
    ):
        assert np.allclose(data["img"], target_data["img"])
        assert np.allclose(data["gt_seg"], target_data["gt_seg"])


@pytest.mark.skipif(
    albumentations is None, reason="albumentations is required"
)
@pytest.mark.parametrize(
    [
        "albu_params",
    ],
    [
        pytest.param(
            [
                dict(
                    name="GaussNoise",
                    var_limit=50.0,
                    p=0.5,
                ),
                dict(
                    name="Blur",
                    p=0.2,
                    blur_limit=(3, 15),
                ),
            ]
        ),
    ],
)
def test_seq_albu_image_only_transform(albu_params):
    data = gen_fake_seq_data(640, 352, layout="hwc")
    data_ori = deepcopy(data)
    cvter = SeqAlbuImageOnlyTransform(albu_params)
    data_after = cvter(data)
    for ori_data, after_data in zip(
        data_ori["frame_data_list"], data_after["frame_data_list"]
    ):
        assert np.allclose(ori_data["gt_bboxes"], after_data["gt_bboxes"])


@pytest.mark.parametrize(
    [
        "max_gt_boxes_num",
        "max_ig_regions_num",
    ],
    [
        pytest.param(1, 1),
        pytest.param(1000, 1000),
    ],
)
def test_to_faster_rcnn_data(max_gt_boxes_num, max_ig_regions_num):
    data = gen_fake_seq_data(640, 352, layout="hwc")
    data_ori = deepcopy(data)

    cvter = SeqToFasterRCNNData(
        max_gt_boxes_num=max_gt_boxes_num,
        max_ig_regions_num=max_ig_regions_num,
    )
    seq_faster_rcnn_data = cvter(data)
    for ori_data, after_data in zip(
        data_ori["frame_data_list"], seq_faster_rcnn_data["frame_data_list"]
    ):
        num_gt = (ori_data["gt_classes"] > 0).sum()
        assert len(after_data["gt_boxes"]) == max_gt_boxes_num
        assert len(after_data["ig_regions"]) == max_ig_regions_num
        if num_gt > max_gt_boxes_num:
            assert 0 == after_data["gt_boxes"][0][-1]
            assert 0 == after_data["gt_boxes_num"][0]
        else:
            assert num_gt == after_data["gt_boxes_num"][0]


if __name__ == "__main__":
    pytest.main(["-s", __file__])
