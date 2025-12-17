# Copyright (c) Horizon Robotics. All rights reserved.
import pytest
import torch

from hat.metrics.gaze_metric import (
    AngleDifferenceMetric,
    EyeLdmksDist,
    GazeModelTestMetric,
)

labels = {
    "gt_gaze": torch.randn(21, 4),
    "gt_glass": torch.zeros(21),
    "gt_normed_eye_ldmk": torch.randn(42, 2),
}
preds = {
    "gaze": torch.randn(21, 4),
    "glass_gaze": torch.randn(21, 4),
    "glass_cls": torch.randn(21, 2),
    "eye_ldmk": torch.randn(42, 2),
}


@pytest.mark.parametrize(
    "labels, preds",
    [
        (labels, preds),
    ],
)
def test_angle_difference(labels, preds):
    angle = AngleDifferenceMetric(name="left_eye", use_glass=True)
    angle.update(labels, preds)
    name, result = angle.get()
    assert result > 0


@pytest.mark.parametrize(
    "labels, preds",
    [
        (labels, preds),
    ],
)
def test_eye_ldmk(labels, preds):
    model_val = EyeLdmksDist(name="angle")
    eye_ldmk = preds["eye_ldmk"].view(1, 84, 1, 1)
    eye_ldmk = [
        eye_ldmk[0, :16, 0, 0].view(-1, 16, 1, 1),
        eye_ldmk[0, 16:32, 0, 0].view(-1, 16, 1, 1),
        eye_ldmk[0, 32:42, 0, 0].view(-1, 10, 1, 1),
        eye_ldmk[0, 42:58, 0, 0].view(-1, 16, 1, 1),
        eye_ldmk[0, 58:74, 0, 0].view(-1, 16, 1, 1),
        eye_ldmk[0, 74:84, 0, 0].view(-1, 10, 1, 1),
    ]
    preds["eye_ldmk"] = eye_ldmk
    model_val.update(labels["gt_normed_eye_ldmk"], preds)
    name, result = model_val.get()
    assert result > 0


@pytest.mark.parametrize(
    "labels, preds",
    [
        (labels, preds),
    ],
)
def test_gaze_model_test_metric(labels, preds):
    model_val = GazeModelTestMetric(name="gaze")
    model_val.update(labels, preds)
    name, result = model_val.get()


if __name__ == "__main__":
    pytest.main(["-s", __file__])
