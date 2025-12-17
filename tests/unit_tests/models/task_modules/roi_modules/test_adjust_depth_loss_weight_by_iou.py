# Copyright (c) Horizon Robotics. All rights reserved.
import numpy as np
import pytest
import torch

from hat.models.task_modules.roi_modules.roi_3d_loss import (
    generate_depth_weight,
)


def test_generate_depth_weight():
    input = [
        [0.5],
        [1.0],
    ]
    output = [
        [1.64872127],
        [2.71828183],
    ]
    input = torch.from_numpy(np.array(input))
    expected_result = torch.from_numpy(np.array(output))
    computed_result = generate_depth_weight(input)

    computed_result_sample = computed_result[:, 0, 0, 0].view(-1, 1)

    assert torch.allclose(computed_result_sample, expected_result, rtol=1.0e-3)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
