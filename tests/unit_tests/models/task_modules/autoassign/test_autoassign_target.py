# Copyright (c) Horizon Robotics. All rights reserved.

import random

import numpy as np
import pytest
import torch

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
def test_autoassign_target(h, w, strides, num_classes, force_topk):
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
    target = target[0]
    assert target["cls_scores"][0].shape[-1] == num_classes
    assert (
        target["cls_scores"][0].shape[0] == target["objectnesses"][0].shape[0]
    )
    assert target["label_weights_list"][0].shape[-1] == num_classes
    assert (
        target["center_prior_weight_list"][0].shape[0]
        == target["inside_gt_bbox_mask_list"][0].shape[0]
    )
    assert target["decoded_bbox_preds_list"][0].shape[-1] == 4
    assert target["decoded_target_preds_list"][0].shape[-1] == 4
