# Copyright (c) Horizon Robotics. All rights reserved.

import pytest
import torch

from hat.models.task_modules.iris import IrisSingleBranchHead


@pytest.mark.parametrize(
    [
        "alpha",
        "classfier_num",
        "start_stage",
        "end_stage",
    ],
    [
        pytest.param(0.25, 2, 4, 6),
    ],
)
def test_hmbackbone(
    alpha,
    classfier_num,
    start_stage,
    end_stage,
):

    in_chls = [
        None,
        None,
        None,
        None,
        [int(x * alpha) for x in [64, 64, 64]],
        [int(x * alpha) for x in [48, 32]],
    ]
    out_chls = [
        None,
        None,
        None,
        None,
        [int(x * alpha) for x in [64, 64, 48]],
        [int(x * alpha) for x in [32, 32]],
    ]

    model = IrisSingleBranchHead(
        bn_kwargs={},
        alpha=alpha,
        bias=True,
        classifier_num=classfier_num,
        start_stage=start_stage,
        end_stage=end_stage,
        use_pool=False,
        in_chls=in_chls,
        out_chls=out_chls,
        pre_channels=4,
    )

    batch_size = 64
    dummy_data = torch.randn(batch_size, 64, 6, 10)
    y = model(dummy_data)

    assert y.shape == (batch_size, classfier_num, 1, 1)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
