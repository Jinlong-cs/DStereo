# Copyright (c) Horizon Robotics. All rights reserved.
import numpy as np
import pytest

from hat.models.task_modules.roi_modules.roi_3d_loss import cal_iou_by_polygon


def test_rotation_iou():
    gt = [
        [1, 0],
        [1, 2],
        [2, 2],
        [2, 0],
    ]
    pred = [
        [0, 1],
        [1, 2],
        [2, 1],
        [1, 0],
    ]
    gt = np.array(gt)
    pred = np.array(pred)
    rotation_iou = cal_iou_by_polygon(gt, pred)
    computed_result = [rotation_iou]
    expected_result = [0.3333333]
    print(computed_result)
    assert np.allclose(computed_result, expected_result, rtol=1.0e-5)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
