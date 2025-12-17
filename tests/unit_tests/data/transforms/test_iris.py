# Copyright (c) Horizon Robotics. All rights reserved.

import numpy as np
import pytest

from hat.data.transforms.iris import IrisMtlTrans
from tests.utils import (
    check,
    check_range,
    check_shape,
    check_type,
    gen_fake_transforms_data,
)


@pytest.mark.parametrize("input_size", [([192, 320])])
def test_iris_mtl(input_size):
    h = 120
    w = 200
    dummy_data = gen_fake_transforms_data(h, w, layout="hwc")
    iris_mtl_trans = IrisMtlTrans(input_size=input_size)
    aug_data = iris_mtl_trans(dummy_data).copy()
    check(
        aug_data["img"], check_shape, shape=(input_size[0], input_size[1], 3)
    )
    check(aug_data["img"], check_range, min=-1, max=1)
    check(aug_data["img"], check_type, instance=np.ndarray)
    assert aug_data["img"].dtype == np.float32


if __name__ == "__main__":
    pytest.main(["-s", __file__])
