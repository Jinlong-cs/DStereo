# Copyright (c) Horizon Robotics. All rights reserved.

import pytest

from hat.models.task_modules.ipm_seg import IPMHeadParser
from tests.utils import gen_fake_feats


@pytest.mark.parametrize(
    ["out_strides"],
    [
        pytest.param([4, 8, 16]),
        pytest.param([4]),
    ],
)
def test_seg_target(out_strides):
    w, h = 896, 896
    feats, _, _ = gen_fake_feats(w, h, out_strides, feat_channels=32)
    seg_head_parser = IPMHeadParser(out_strides=out_strides)
    ipm_head_preds = seg_head_parser(feats)
    for ipm_head_pred in ipm_head_preds:
        assert ipm_head_pred.shape[-1] == w
        assert ipm_head_pred.shape[-1] == h
