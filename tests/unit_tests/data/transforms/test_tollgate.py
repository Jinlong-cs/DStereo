# -*- coding: utf-8 -*-
# Copyright (c) Horizon Robotics. All rights reserved.
import numpy as np
import pytest

from hat.registry import build_from_registry
from tests.utils import gen_fake_transforms_data


@pytest.mark.parametrize(
    ["size"], [pytest.param([480, 256]), pytest.param([960, 512])]
)
def test_tollgate_transform(size):
    cfg = dict(
        type="OvalHmTargetGenerator",
        img_scale=(size[1], size[0]),
        encode_lmks=2,
        stride=4,
        sigma=2.0,
        ng_weights=1.0,
    )
    transform_image = build_from_registry(cfg)
    data = gen_fake_transforms_data(*size, "hwc")
    data["gt_lines"] = np.array(
        [[[284.150, 138.705, 16.144]], [[222.129, 138.847, 13.161]]]
    )
    data["img_name"] = None
    data_res = transform_image(data)

    tags = [
        "gt_heatmap",
        "gt_offset",
        "gt_offset_weight",
        "gt_heatmap_weight",
        "img",
        "img_name",
    ]
    for tag in tags:
        assert tag in data_res

    assert (
        data_res["img"].shape[1] == size[1]
        and data_res["img"].shape[2] == size[0]
    )


if __name__ == "__main__":
    pytest.main(["-s", __file__])
