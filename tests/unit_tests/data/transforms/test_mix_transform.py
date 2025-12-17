# Copyright (c) Horizon Robotics. All rights reserved.
import pytest

from hat.data.transforms.detection import DetMosaic
from tests.utils import gen_fake_transforms_data


@pytest.mark.parametrize(
    [
        "cache_num",
        "img_h",
        "img_w",
    ],
    [pytest.param(5, 1408, 1152), pytest.param(6, 704, 576)],
)
def test_plain_copy_paste(cache_num, img_h, img_w):
    det_mosaic = DetMosaic(
        img_scale=[i // 2 for i in (img_h, img_w)],
        center_ratio_range=(0.75, 1.25),
        p=1,
    )
    for i in range(cache_num):
        data = gen_fake_transforms_data(img_w, img_h, layout="hwc")
        data["gt_bboxes"] = data["gt_bboxes"] + i * 10
        tmp = det_mosaic(data)
    assert len(tmp["gt_bboxes"]) <= 8
    assert len(tmp["gt_classes"]) <= 8
    assert tmp["img_height"] == img_h
    assert tmp["img_width"] == img_w


if __name__ == "__main__":
    pytest.main(["-s", __file__])
