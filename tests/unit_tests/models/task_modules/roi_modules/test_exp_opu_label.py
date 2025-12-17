# Copyright (c) Horizon Robotics. All rights reserved.
import numpy as np
import pytest
import torch

from hat.models.task_modules.roi_modules.roi_3d_loss import exp_iou_label


def test_exp_iou_label():
    input = [
        [0.5],
        [1.0],
    ]
    output = [
        [0.6567],
        [0.8479],
    ]
    input = torch.from_numpy(np.array(input))
    expected_result = torch.from_numpy(np.array(output))
    computed_result = exp_iou_label(input)

    assert torch.allclose(computed_result, expected_result, rtol=1.0e-3)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
