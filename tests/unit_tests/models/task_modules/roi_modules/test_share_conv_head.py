# Copyright (c) Horizon Robotics. All rights reserved.

import pytest
import torch

from hat.models.task_modules.roi_modules.share_conv_head import ShareConv
from tests.unit_tests.models.base import qat_test, qtensor_test


@pytest.mark.parametrize(
    ["group_base", "num_conv", "in_num_filter", "shared_conv_method"],
    [
        pytest.param(4, 2, 32, "BasicVargBlockDownUp"),
        pytest.param(8, 0, 64, "BasicVargBlockDownUp"),
        pytest.param(4, 2, 128, "BasicVargBlockDownUp"),
        pytest.param(8, 0, 128, "BasicVargBlockDownUp"),
        pytest.param(8, 2, 64, "BasicVargBlock"),
        pytest.param(16, 0, 128, "Conv"),
    ],
)
def test_share_conv_head(
    group_base,
    num_conv,
    in_num_filter,
    shared_conv_method,
):
    input = torch.randn((1, in_num_filter, 64, 64))

    share_conv = ShareConv(
        in_num_filter=in_num_filter,
        num_conv=num_conv,
        group_base=group_base,
        shared_conv_method=shared_conv_method,
    )
    output = share_conv(input)

    assert output.size() == (1, 128, 64, 64)

    # test share_bn fuse_model
    input_qat = qtensor_test(input)
    qat_test(share_conv, input_qat, with_quantized=False)
