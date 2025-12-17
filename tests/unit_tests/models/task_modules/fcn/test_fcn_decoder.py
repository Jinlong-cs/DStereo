# Copyright (c) Horizon Robotics. All rights reserved.

import pytest

from hat.models.task_modules.fcn import FCNDecoder
from tests.utils import gen_fake_feats


@pytest.mark.parametrize(
    ["up_scale"],
    [pytest.param(8), pytest.param(4)],
)
def test_fcn_decoder(up_scale):
    w, h = 512, 1024
    strides = [8]
    pred, _, _ = gen_fake_feats(w, h, strides, n=2, feat_channels=19)
    fcn_decoder = FCNDecoder(
        upsample_output_scale=up_scale,
    )

    out = fcn_decoder(pred[0])
    assert out.size(1) == (h / 8) * up_scale
    assert out.size(2) == (w / 8) * up_scale
