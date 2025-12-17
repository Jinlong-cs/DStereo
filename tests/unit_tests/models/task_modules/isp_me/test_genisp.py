# Copyright (c) Horizon Robotics. All rights reserved.
import pytest

from hat.models.task_modules.isp_me import GenISP
from tests.utils import gen_fake_torch_randn_data


@pytest.mark.parametrize("in_channels", [3])
@pytest.mark.parametrize("out_channels", [3, 4])
@pytest.mark.parametrize("split_transform", [True, False])
def test_genisp(in_channels, out_channels, split_transform):
    batch_size = 2
    height = 512
    width = 960
    data = {}
    data["img"] = gen_fake_torch_randn_data(
        (batch_size, in_channels, height, width),
    )
    data["me_in_img"] = gen_fake_torch_randn_data(
        (batch_size, in_channels, 256, 256),
    )
    model = GenISP(in_channels, out_channels, split_transform)
    y = model(data)
    assert y.shape == (batch_size, out_channels, height, width)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
