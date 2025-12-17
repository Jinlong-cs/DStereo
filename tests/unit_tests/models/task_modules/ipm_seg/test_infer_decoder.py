# Copyright (c) Horizon Robotics. All rights reserved.

import pytest

from hat.models.task_modules.ipm_seg import SemSegDecoder
from tests.utils import gen_fake_seg_data


@pytest.mark.parametrize(
    ["output_name"],
    [
        pytest.param("pred_seg"),
    ],
)
def test_fcos_decoder(output_name):
    h, w = 896, 896
    stride = [4]
    num_classes = 21
    pred, label = gen_fake_seg_data(h, w, stride, num_classes)
    seg_decoder = SemSegDecoder(output_name)
    results = seg_decoder(tuple(pred), label)
    assert output_name in results
    for _, out in results.items():
        assert out.shape[-2:] == (h, w)
