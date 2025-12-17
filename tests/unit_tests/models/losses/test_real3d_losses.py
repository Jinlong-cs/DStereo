# Copyright (c) Horizon Robotics. All rights reserved.

import pytest
import torch

from hat.models.losses.real3d_losses import (
    Real3DLoss,
    angle_multibin_loss,
    dep_l1_loss,
    hm_focal_loss,
    hm_l1_loss,
    mask_l1_loss,
)


@pytest.mark.parametrize(
    ["pred", "gt", "ignore_mask", "target_loss"],
    [
        pytest.param(
            0.98 * torch.ones(1, 1, 128, 240),
            0.98 * torch.ones(1, 1, 128, 240),
            torch.ones(1, 1, 128, 240),
            0.0,
        ),
        pytest.param(
            0.98 * torch.ones(1, 1, 128, 240),
            0.98 * torch.ones(1, 1, 128, 240),
            torch.zeros(1, 1, 128, 240),
            0.0185,
        ),
        pytest.param(
            0.5 * torch.ones(1, 1, 128, 240),
            0.5 * torch.ones(1, 1, 128, 240),
            torch.zeros(1, 1, 128, 240),
            332.7107,
        ),
        pytest.param(
            0.02 * torch.ones(1, 1, 128, 240),
            0.02 * torch.ones(1, 1, 128, 240),
            torch.zeros(1, 1, 128, 240),
            0.2290,
        ),
    ],
)
def test_hm_focal_loss(pred, gt, ignore_mask, target_loss):
    loss = hm_focal_loss(pred, gt, ignore_mask)
    assert torch.abs(loss - target_loss) < 1e-4


@pytest.mark.parametrize(
    [
        "output",
        "target",
        "weight_mask",
        "ignore_mask",
        "heatmap_type",
        "target_loss",
    ],
    [
        pytest.param(
            torch.zeros(1, 2, 128, 240),
            torch.ones(1, 2, 128, 240),
            torch.ones(1, 1, 128, 240),
            torch.zeros(1, 1, 128, 240),
            "heatmap",
            2.0,
        ),
        pytest.param(
            torch.ones(1, 2, 128, 240),
            torch.ones(1, 2, 128, 240),
            torch.ones(1, 1, 128, 240),
            torch.zeros(1, 1, 128, 240),
            "heatmap",
            0.0,
        ),
        pytest.param(
            torch.zeros(1, 2, 128, 240),
            0.5 * torch.ones(1, 2, 128, 240),
            torch.ones(1, 1, 128, 240),
            torch.zeros(1, 1, 128, 240),
            "heatmap",
            1.0,
        ),
    ],
)
def test_hm_l1_loss(
    output, target, weight_mask, ignore_mask, heatmap_type, target_loss
):
    loss = hm_l1_loss(output, target, weight_mask, ignore_mask, heatmap_type)
    assert torch.abs(loss - target_loss) < 1e-4


@pytest.mark.parametrize(
    [
        "output",
        "target",
        "weight_mask",
        "ignore_mask",
        "heatmap_type",
        "max_dep",
        "target_loss",
    ],
    [
        pytest.param(
            torch.zeros(1, 1, 128, 240),
            torch.ones(1, 1, 128, 240),
            torch.ones(1, 1, 128, 240),
            torch.zeros(1, 1, 128, 240),
            "heatmap",
            150,
            1.2841,
        ),
        pytest.param(
            torch.zeros(1, 1, 128, 240),
            torch.zeros(1, 1, 128, 240),
            torch.ones(1, 1, 128, 240),
            torch.zeros(1, 1, 128, 240),
            "heatmap",
            150,
            0.0,
        ),
    ],
)
def test_dep_l1_loss(
    output,
    target,
    weight_mask,
    ignore_mask,
    heatmap_type,
    max_dep,
    target_loss,
):
    loss = dep_l1_loss(
        output, target, weight_mask, ignore_mask, heatmap_type, max_dep
    )
    assert torch.abs(loss - target_loss) < 1e-4


def test_dynamic_loss_weight():
    real3d_loss = Real3DLoss(
        max_dep=150.0,
        loss_weights={"hm": 1.0, "dep": 2.0},
        heatmap_type=None,
        use_dynamic_weight=False,
        init_values=None,
        norm_type="direct",
    )
    assert not real3d_loss.use_dynamic_weight
    assert real3d_loss.norm_type == "direct"
    assert real3d_loss.init_values["hm"] == 0.0

    real3d_loss = Real3DLoss(
        max_dep=150.0,
        loss_weights={"hm": 1.0, "dep": 2.0},
        heatmap_type=None,
        use_dynamic_weight=True,
        init_values={"hm": 10.0, "dep": 20.0},
        norm_type="l1",
    )
    assert real3d_loss.use_dynamic_weight
    assert real3d_loss.norm_type == "l1"
    assert real3d_loss.init_values["hm"] == 10.0
    assert real3d_loss.init_values["dep"] == 20.0
    assert real3d_loss.init_values["rot"] == 0.0
    assert real3d_loss.dynamic_weights["hm"].scale == 10.0
    assert real3d_loss.dynamic_weights["dep"].scale == 20.0


def test_mask_l1_loss():
    p = torch.tensor([0, 1, 2, 4], dtype=torch.float32)
    t = torch.tensor([10, 1, 2, 2], dtype=torch.float32)
    mask = torch.tensor([1, 0, 0, 1], dtype=torch.float32)
    loss = mask_l1_loss(p, t, mask)
    assert loss == 6.0
    mask = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
    loss = mask_l1_loss(p, t, mask)
    assert loss == 0.0


def test_multibin_loss():
    n_bin = 2
    max_objs = 4
    pred_cls = torch.zeros((2, max_objs, n_bin), dtype=torch.float32)
    pred_offset = torch.zeros((2, max_objs, n_bin * 2), dtype=torch.float32)
    mask = (
        torch.tensor([0, 1, 1, 0], dtype=torch.float32)
        .reshape((1, max_objs))
        .repeat(2, 1)
    )
    gt_cls = (
        torch.tensor([[0, 0], [1, 0], [0, 1], [0, 0]], dtype=torch.float32)
        .reshape((1, max_objs, 2))
        .repeat(2, 1, 1)
    )
    gt_offset = (
        torch.tensor(
            [[0, 0], [0.52, 0], [0, 1.047], [0, 0]], dtype=torch.float32
        )
        .reshape((1, max_objs, 2))
        .repeat(2, 1, 1)
    )
    loss = angle_multibin_loss(pred_cls, pred_offset, gt_cls, gt_offset, mask)
    assert isinstance(loss, torch.Tensor)
