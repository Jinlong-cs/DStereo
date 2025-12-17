import numpy as np
import pytest
import torch

from hat.metrics.face3d import (
    Eye3dMAE,
    Face3dSTD,
    PoseMAE,
    transfer_head_pose_axes,
)

LABELS = {
    "gt_pose": torch.randn((10, 3)),
    "eye3d_left": torch.randn(10, 3),
    "eye3d_right": torch.randn(10, 3),
}

PREDS = {
    "global_pose": torch.randn((10, 3)),
    "eye3d_left": torch.randn((10, 3)),
    "eye3d_right": torch.randn((10, 3)),
}


def test_posemae_pp():
    pose_mae = PoseMAE(model_type="pp")
    pose_mae.update(LABELS, PREDS)
    _, val = pose_mae.get()
    assert sum(val) > 1e-5


def test_eyelocmae_pp():
    eye_mae = Eye3dMAE()
    eye_mae.update(LABELS, PREDS)
    _, val = eye_mae.get()
    assert sum(val) > 1e-5


@pytest.mark.parametrize(
    ["pose_euler"],
    [
        pytest.param(np.random.uniform(-180, 180, (10, 3))),
        pytest.param((torch.rand((10, 3)) - 0.5) * 180),
    ],
)
def test_transfer_head_pose_axes(pose_euler):
    output = transfer_head_pose_axes(pose_euler)
    assert type(pose_euler) == type(output)


def test_face3d_std():
    face3d_std = Face3dSTD("pp")
    face3d_std.update(LABELS, PREDS)
    _, stds = face3d_std.get()
    assert len(stds) == 9
    for std in stds:
        assert std >= 0
