# Copyright (c) Horizon Robotics. All rights reserved.

import pytest
import torch

from hat.models.task_modules.track2d import RCNNTrackSplitHead
from tests.unit_tests.models.base import qat_test, qtensor_test


@pytest.mark.parametrize(
    [
        "in_channel",
        "pw_num_filter2",
        "feat_len",
    ],
    [
        pytest.param(64, 128, 64),
        pytest.param(128, 64, 128),
    ],
)
def test_RCNNTrackSplitHead(
    in_channel,
    pw_num_filter2,
    feat_len,
):
    input_shape = (16, 32)
    x = torch.randn((1, in_channel, *input_shape))

    split_head = RCNNTrackSplitHead(
        in_channel=in_channel,
        pw_num_filter2=pw_num_filter2,
        feat_len=feat_len,
        bn_kwargs={},
    )
    output = split_head(x)
    assert output["track_feat"].size(1) == feat_len

    # test share_bn fuse_model
    input_qat = qtensor_test(x)
    qat_test(split_head, input_qat, with_quantized=False)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
