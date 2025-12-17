# Copyright (c) Horizon Robotics. All rights reserved.

import numpy as np
import torch

from hat.models.task_modules.depth import (
    ConfL1Loss,
    DepthConfidenceCombinedLoss,
    DepthL1Loss,
    DepthVNLLoss,
    SmoothDepthLoss,
)
from hat.utils.seed import seed_everything


def test_conf_l1_loss():
    seed_everything(17)
    height, width = 16, 16
    pred_depth = torch.from_numpy(
        np.random.randint(2, 2 ** 16, [2, 1, height, width]) / 256.0
    )
    target_depth = pred_depth + torch.from_numpy(
        np.random.randn(2, 1, height, width)
    )
    pred_conf = torch.sigmoid(torch.randn([2, 1, height, width]))
    conf_l1_loss = ConfL1Loss(
        max_gt_depth=150.0,
        use_weight_map=False,
        loss_weight=1.0,
        loss_name="conf_l1_loss",
    )
    loss = conf_l1_loss(pred_conf, pred_depth, {"gt_depth": target_depth})
    target_loss = 0.4805
    assert torch.abs(loss["conf_l1_loss"] - target_loss) < 1e-4


def test_depth_l1_loss():
    seed_everything(17)
    height, width = 16, 16
    pred_depth = torch.from_numpy(
        np.random.randint(2, 2 ** 16, [2, 1, height, width]) / 256.0
    )
    target_depth = pred_depth + torch.from_numpy(
        np.random.randn(2, 1, height, width)
    )
    dpeht_l1_loss = DepthL1Loss(
        loss_weight=1.0,
        max_gt_depth=150,
        beta=1e-6,
        use_weight_map=False,
        loss_name="depth_l1_loss",
    )
    loss = dpeht_l1_loss(pred_depth, {"gt_depth": target_depth})
    target_loss = 0.8245
    assert torch.abs(loss["depth_l1_loss"] - target_loss) < 1e-4


def test_depth_conf_loss():
    high_depth = 150.0
    gt_scale = 1.0
    seed_everything(17)
    height, width = 16, 16
    pred_depth = torch.from_numpy(
        np.random.randint(2, 2 ** 16, [2, 1, height, width]) / 256.0
    )
    target_depth = pred_depth + torch.from_numpy(
        np.random.randn(2, 1, height, width)
    )
    pred_conf = torch.sigmoid(torch.randn([2, 1, height, width]))
    preds = torch.cat([pred_depth, pred_conf], dim=1)
    depth_conf_loss = DepthConfidenceCombinedLoss(
        depth_loss=DepthL1Loss(
            max_gt_depth=high_depth / gt_scale,
            loss_weight=1.0,
            loss_name="depth_l1_loss",
            use_weight_map=True,
        ),
        conf_loss=ConfL1Loss(
            max_gt_depth=high_depth / gt_scale,
            loss_weight=1.0,
            loss_name="conf_l1_loss",
            use_weight_map=False,
        ),
        depth_loss_name="depth_loss_s1",
        conf_loss_name="conf_loss_s1",
        loss_weight=[20.0, 2.0],
    )
    loss = depth_conf_loss(preds, {"gt_depth": target_depth})
    target_loss_depth = 16.4899
    target_loss_conf = 0.9609

    assert torch.abs(loss["depth_loss_s1"] - target_loss_depth) < 1e-4
    assert torch.abs(loss["conf_loss_s1"] - target_loss_conf) < 1e-4


def test_smooth_depth_loss():
    seed_everything(17)
    height, width = 16, 16
    pred_depth = torch.from_numpy(
        np.random.randint(2, 2 ** 16, [2, 1, height, width]) / 256.0
    )
    origin_img = torch.from_numpy(np.random.randn(2, 1, height, width))
    smooth_depth_loss = SmoothDepthLoss(
        do_normalization=True,
        loss_weight=1.0,
        loss_name="smooth_loss",
    )
    loss = smooth_depth_loss(pred_depth, {"origin_img": origin_img})
    target_loss = 0.5538
    assert torch.abs(loss["smooth_loss"] - target_loss) < 1e-4


def test_depth_vnl_loss():
    seed_everything(17)
    height, width = 288, 352
    pred = torch.from_numpy(
        np.random.randint(2, 2 ** 16, [2, 1, height, width]) / 256.0
    )
    target = pred + torch.from_numpy(np.random.randn(2, 1, height, width))
    virtual_cam_params = np.array(
        [[110, 0, 176], [0, 110, 144], [0, 0, 1]], dtype=np.float32
    )
    vnl_sample_ratio = 0.6
    gt_scale = 1.0
    depth_type = "Cylindrical"
    deptn_vnl_loss = DepthVNLLoss(
        focal_x=int(virtual_cam_params[0, 0]),
        focal_y=int(virtual_cam_params[1, 1]),
        input_size=[height, width],
        delta_cos=0.867,
        delta_diff_x=0.01,
        delta_diff_y=0.01,
        delta_diff_z=0.01,
        delta_z=0.00001,
        sample_ratio=vnl_sample_ratio,
        gt_scale=gt_scale,
        scale_pred=True,
        depth_type=depth_type,
        loss_name="vnl_loss",
    )
    loss = deptn_vnl_loss(pred, {"gt_depth": target})
    target_loss = 0.0102
    assert torch.abs(loss["vnl_loss"] - target_loss) < 1e-4
