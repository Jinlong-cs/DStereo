# Copyright (c) Horizon Robotics. All rights reserved.
import numpy as np
import pytest
import torch

from hat.models.losses.cross_entropy_loss import (
    CEWithHardMining,
    CEWithWeightMap,
    CrossEntropyLoss,
    SoftTargetCrossEntropy,
    cross_entropy,
)
from hat.utils.seed import seed_everything


def test_cross_entropy():
    pred = torch.Tensor(
        [
            [0.0241, 0.7897, 0.8505, 0.1845, 0.9831],
            [0.3923, 0.3279, 0.8779, 0.6180, 0.2927],
            [0.7145, 0.2136, 0.1793, 0.1577, 0.0701],
        ]
    )
    target = torch.tensor([0, 4, 1])
    loss = cross_entropy(pred, target)
    assert torch.abs(loss - 1.9189) < 1e-4

    weight = torch.Tensor([0.2, 0.3, 0.5])
    loss = cross_entropy(pred, target, weight)
    assert torch.abs(loss - 1.8432) < 1e-4

    weight = torch.Tensor([0.0, 0.0, 0.0])
    loss = cross_entropy(pred, target, weight)
    assert loss == 0.0


def test_cross_entropy_loss():
    pred_list = [
        torch.Tensor([0.4475, 1.6851, -1.6667, 0.7341, -2.0608]),
        torch.Tensor([0.0953, -0.3291, 0.3174, 0.6713, 1.3140]),
    ]
    target_list = [
        torch.Tensor([0.4900, 0.2658, 0.1319, 0.9753, 0.2768]),
        torch.Tensor([0.0152, 0.7072, 0.5548, 0.5564, 0.5547]),
    ]
    target_loss = [
        torch.FloatTensor([0.7246]).squeeze(),
        torch.FloatTensor([0.7475]).squeeze(),
    ]

    cross_entropy_loss = CrossEntropyLoss(use_sigmoid=True)
    for ii, (pred, target) in enumerate(zip(pred_list, target_list)):
        loss = cross_entropy_loss(pred, target)
        assert torch.abs(loss - target_loss[ii]) < 1e-4


def test_multi_output_cross_entropy_loss():
    pred_list = [
        torch.Tensor([0.4475, 1.6851, -1.6667, 0.7341, -2.0608]).view((1, -1)),
        torch.Tensor([0.0953, -0.3291, 0.3174, 0.6713, 1.3140]).view((1, -1)),
    ]
    target = torch.Tensor([0])
    target_loss = 7.7523
    class_weight = [2.0, 2.0, 2.0, 2.0, 1.0]

    cross_entropy_loss = CrossEntropyLoss(
        loss_name="loss_cls", use_sigmoid=False, class_weight=class_weight
    )
    loss = cross_entropy_loss(pred_list, target)
    assert torch.abs(loss["loss_cls"] - target_loss) < 1e-4


def test_soft_target_cross_entropy():
    pred = torch.Tensor(
        [
            [0.4475, 1.6851, -1.6667, 0.7341, -2.0608],
            [0.0953, -0.3291, 0.3174, 0.6713, 1.3140],
        ]
    )
    target = torch.Tensor([[0.5], [0.4]])

    cross_entropy_loss = SoftTargetCrossEntropy(loss_name="loss_cls")

    loss = cross_entropy_loss(pred, target)
    assert "loss_cls" in loss
    assert torch.abs(loss["loss_cls"] - torch.tensor(4.7794)) < 1e-4

    pred = torch.Tensor(
        [
            [0.4475, 1.6851, -1.6667, 0.7341, -2.0608],
            [0.0953, -0.3291, 0.3174, 0.6713, 1.3140],
        ]
    )
    target = torch.Tensor([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])
    cross_entropy_loss = SoftTargetCrossEntropy(loss_name="loss_cls")
    loss = cross_entropy_loss(pred, target)
    assert "loss_cls" in loss
    assert torch.abs(loss["loss_cls"] - torch.tensor(1.2082)) < 1e-4


def test_ceweightmap_loss():
    seed_everything(17)
    pred = torch.Tensor(np.random.randn(2, 4, 16, 16)).float()
    target = torch.Tensor(np.full((2, 16, 16), 2)).float()
    target_loss = 1.7021

    ce_weightmap_loss = CEWithWeightMap(
        use_sigmoid=False,
        loss_weight=1,
        reduction="mean",
        loss_name="ce_loss",
        ignore_index=255,
        num_class=4,
        auto_class_weight=False,
        weight_min=0.5,
        weight_noobj=0.75,
    )

    loss = ce_weightmap_loss(pred, target)
    assert torch.abs(loss["ce_loss"] - target_loss) < 1e-4


@pytest.mark.parametrize("raw_cls_num", [7])
def test_ceweightmap_loss_with_remap(raw_cls_num):
    seed_everything(17)
    pred = torch.Tensor(np.random.randn(2, raw_cls_num, 16, 16)).float()
    target = torch.Tensor(np.full((2, 16, 16), 2)).long()
    target_loss = 1.2762
    remap_dict = {
        "old2newmap": {
            0: 0,
            1: 1,
            2: 1,
            3: 1,
            4: 1,
            5: 1,
            6: 1,
        },
        "raw_cls_num": raw_cls_num,
    }
    ce_weightmap_loss = CEWithWeightMap(
        use_sigmoid=False,
        loss_weight=1,
        reduction="mean",
        loss_name="ce_loss",
        ignore_index=255,
        num_class=2,
        auto_class_weight=False,
        weight_min=0.5,
        weight_noobj=0.75,
        remap_params=remap_dict,
    )

    loss = ce_weightmap_loss(pred, target)
    assert torch.abs(loss["ce_loss"] - target_loss) < 1e-4


hard_neg_mining_cfg = dict(
    keep_pos=True,
    neg_ratio=0.5,
    hard_ratio=0.5,
    min_keep_num=1,
)


@pytest.mark.parametrize("hard_neg_mining_cfg", [hard_neg_mining_cfg, None])
@pytest.mark.parametrize("norm_type", ["fbg_elt", "fg_elt"])
def test_ce_with_hard_ming_loss(hard_neg_mining_cfg, norm_type):
    pred = torch.Tensor([[0.0], [0.0], [1.0], [1.0]])
    target = torch.Tensor([[0], [0], [1], [1]])
    weight = torch.Tensor([[1], [1], [1], [1]])

    cross_entropy_loss = CEWithHardMining(
        use_sigmoid=True,
        norm_type=norm_type,
        reduction="mean",
        hard_neg_mining_cfg=hard_neg_mining_cfg,
    )

    loss = cross_entropy_loss(pred, target, weight)
    if norm_type == "fg_elt":
        assert torch.allclose(loss, torch.tensor(0.5032) * 2)
    if norm_type == "bfg_elt":
        assert torch.allclose(loss, torch.tensor(0.5032))


if __name__ == "__main__":
    pytest.main(["-s", __file__])
