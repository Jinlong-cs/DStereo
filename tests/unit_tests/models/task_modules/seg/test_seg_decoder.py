# Copyright (c) Horizon Robotics. All rights reserved.

import pytest

from hat.models.task_modules.seg import SegDecoder
from hat.utils.apply_func import _as_list
from tests.utils import gen_fake_seg_data

transforms = [
    dict(type="Resize", img_scale=(640, 640), keep_ratio=True),
    dict(type="FixedCrop", size=(0, 0, 640, 640)),
]
inverse_transform_key = ["scale_factor", "crop_offset", "before_crop_shape"]


@pytest.mark.parametrize(
    ["h", "w", "strides", "num_classes", "output_names", "decode_strides"],
    [
        pytest.param(640, 640, (4, 8, 16, 32, 64), 2, "pred_seg", 4),
        pytest.param(
            640, 640, (8, 16, 32, 64), 5, ["seg_8", "seg_16"], [8, 16]
        ),
    ],
)
def test_fcos_decoder(
    h, w, strides, num_classes, output_names, decode_strides
):
    pred, label = gen_fake_seg_data(h, w, strides, num_classes)
    seg_decoder = SegDecoder(
        strides,
        decode_strides=decode_strides,
        output_names=output_names,
        transforms=transforms,
        inverse_transform_key=inverse_transform_key,
    )
    results = seg_decoder(pred, label)
    for output_name in _as_list(output_names):
        assert output_name in results
    for _, out in results.items():
        assert out.shape[-2:] == (h, w)
