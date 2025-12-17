# Copyright (c) Horizon Robotics. All rights reserved.

import random

import numpy as np
import pytest
import torch

from hat.models.losses.auto_assign_loss import AutoAssignLoss
from hat.models.task_modules.autoassign import AutoAssignTarget
from hat.registry import build_from_registry
from tests.utils import gen_fake_det_label_pred_data

INF = 1e8


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


@pytest.mark.parametrize(
    ["h", "w", "strides", "num_classes", "force_topk"],
    [
        pytest.param(640, 640, (4, 8, 16, 32, 64), 2, False),
    ],
)
def test_autoassign_loss(h, w, strides, num_classes, force_topk):
    # set random seed
    setup_seed(20)
    autoassign_target = AutoAssignTarget(
        strides=strides,
        norm_on_bbox=False,
        center_prior=build_from_registry(
            dict(
                type="CenterPrior",
                force_topk=force_topk,
                num_classes=num_classes,
                strides=strides,
            )
        ),
        cls_out_channels=num_classes,
    )
    label, pred = gen_fake_det_label_pred_data(h, w, strides, num_classes)
    target = autoassign_target(label, pred)
    autoassign_loss = AutoAssignLoss(
        loss_name=["loss_pos", "loss_neg", "loss_center"],
        pos_loss=build_from_registry(
            dict(
                type="PosLoss",
                reg_loss=dict(
                    type="GIoULoss",
                    loss_name="loss_bbox",
                    loss_weight=5.0,
                    reduction="none",
                ),
                loss_weight=0.25,
            )
        ),
        neg_loss=build_from_registry(dict(type="NegLoss", loss_weight=0.75)),
        center_loss=build_from_registry(
            dict(type="CenterLoss", loss_weight=0.75)
        ),
    )
    loss = autoassign_loss(pred, target)
    assert abs(loss["loss_pos"].item() - 1.5766) < 1e-4
    assert abs(loss["loss_neg"].item() - 31.0864) < 1e-4
    assert abs(loss["loss_center"].item() - 0.0269) < 1e-4


@pytest.mark.parametrize(
    ["h", "w", "strides", "num_classes", "force_topk", "max_avg_num_gt"],
    [
        pytest.param(640, 640, (4, 8, 16, 32, 64), 2, False, 1),
    ],
)
def test_restrict_num_gt_autoassign_loss(
    h, w, strides, num_classes, force_topk, max_avg_num_gt
):
    # set random seed
    setup_seed(13)
    autoassign_target = AutoAssignTarget(
        strides=strides,
        norm_on_bbox=False,
        center_prior=build_from_registry(
            dict(
                type="CenterPrior",
                force_topk=force_topk,
                num_classes=num_classes,
                strides=strides,
            )
        ),
        cls_out_channels=num_classes,
        max_avg_num_gt=max_avg_num_gt,
    )
    label, pred = gen_fake_det_label_pred_data(h, w, strides, num_classes)
    target = autoassign_target(label, pred)
    autoassign_loss = AutoAssignLoss(
        loss_name=["loss_pos", "loss_neg", "loss_center"],
        pos_loss=build_from_registry(
            dict(
                type="PosLoss",
                reg_loss=dict(
                    type="GIoULoss",
                    loss_name="loss_bbox",
                    loss_weight=5.0,
                    reduction="none",
                ),
                loss_weight=0.25,
            )
        ),
        neg_loss=build_from_registry(dict(type="NegLoss", loss_weight=0.75)),
        center_loss=build_from_registry(
            dict(type="CenterLoss", loss_weight=0.75)
        ),
    )
    loss = autoassign_loss(pred, target)
    assert abs(loss["loss_pos"].item() - 1.6070) < 1e-4
    assert abs(loss["loss_neg"].item() - 45.5246) < 1e-4
    assert abs(loss["loss_center"].item() - 0.0271) < 1e-4
