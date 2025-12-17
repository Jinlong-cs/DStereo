# Copyright (c) Horizon Robotics. All rights reserved.
import os

import pytest

from hat.registry import build_from_registry
from tests import HAT_BUCKET_PATH

HAT_BUCKET_EXISTS = os.path.exists(HAT_BUCKET_PATH)


@pytest.mark.skipif(not HAT_BUCKET_PATH, reason="need hat bucket")
def test_depthrec_dataset():
    rec_file = os.path.join(
        HAT_BUCKET_PATH, "data/rec_data/sp_depth/depth_ut/train.rec"
    )
    assert os.path.exists(rec_file)
    depth_scale = 256.0
    config = dict(
        type="DepthDatasetRec",
        paths=[rec_file],
        to_rgb=True,
        depth_scale_factor=depth_scale,
        select_sample=False,
        with_parsing=False,
        mode="train",
    )

    dataset = build_from_registry(config)
    for data in dataset:
        assert isinstance(data, dict)
        assert "gt_depth" in data
        break
