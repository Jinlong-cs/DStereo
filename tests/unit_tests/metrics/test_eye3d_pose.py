# Copyright (c) Horizon Robotics. All rights reserved.
import pytest
import torch

from hat.metrics.eye3d_pose import Eye3dPoseAngleAndDist


def test_metric():
    pred = {
        "pred_pose": torch.zeros([32, 6, 1, 1]),
        "pred_eye3d": torch.zeros([32, 6, 1, 1]),
    }
    label = {
        "gt_pose": torch.ones([32, 6, 1, 1]),
        "gt_eye3d": torch.ones([32, 6, 1, 1]) * 2,
    }
    metric = Eye3dPoseAngleAndDist(num_mode=2)
    metric.update(label, pred)
    name, val = metric.get()
    assert len(name) == len(val)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
