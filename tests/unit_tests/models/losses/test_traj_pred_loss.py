# Copyright (c) Horizon Robotics. All rights reserved.

import os
import pickle

import numpy as np
import pytest
import torch

from hat.models.losses.traj_pred_loss import (
    BasicMultipathLoss,
    BehavHeadLoss,
    DenseTNTloss,
    HTTPHeatmapLoss,
    HTTPOffsetLoss,
    MTPLoss,
    SGNetLoss,
    SoftMultipathLoss,
    TrackValidHeadLoss,
    TrajConfidenceLoss,
    TrajOffsetRegLoss,
)
from projects.prediction.configs.processed_dataset import HAT_UNITTEST_PREFIX
from tests import SD_AlGORITHM_BUCKET_EXISTS, SD_AlGORITHM_BUCKET_PATH


@pytest.mark.skipif(
    not SD_AlGORITHM_BUCKET_EXISTS, reason="requiring SD_Algorithm bucket"
)
def test_traj_pred_loss():
    test_loss_data_path = os.path.join(
        SD_AlGORITHM_BUCKET_PATH,
        f"{HAT_UNITTEST_PREFIX}/outputs/test_loss_data_new.pkl",
    )
    with open(test_loss_data_path, "rb") as f:
        raw_output = pickle.load(f)
        model_output = {
            key.split("head_")[1] if "head_" in key else key: value
            for key, value in raw_output.items()
        }

    loss_func = BasicMultipathLoss(head_weight=1, reg_loss_scale=1)
    loss = loss_func(model_output)
    assert "reg_loss" in loss
    assert "cls_loss" in loss
    soft_loss_func = SoftMultipathLoss(head_weight=1, reg_loss_scale=1)
    soft_loss = soft_loss_func(model_output)
    assert "reg_loss" in soft_loss
    assert "cls_loss" in soft_loss


@pytest.mark.parametrize(
    ["output"],
    [
        pytest.param(
            {
                "lat_behav_probs": torch.randn(9, 3),
                "lon_behav_probs": torch.randn(9, 3),
                "lat_behav_gts": torch.tensor([0, 1, 2, 0, 0, 1, 0, 0, 2]),
                "lon_behav_gts": torch.tensor([0, 1, 2, 0, 0, 1, 0, 0, 2]),
                "track_ids": [[-42, 0, 1], [-42, 2, 3], [-42, 4, 5]],
                "behav_track_ids": [[-42, 0, 1], [-42, 2, 3], [-42, 4, 5]],
            }
        )
    ],
)
def test_behav_pred_loss(output):
    loss_func = BehavHeadLoss(
        lat_head_weight=1,
        lon_head_weight=1,
        lat_class_weight=[1.0, 1.0, 1.0],
        lon_class_weight=[1.0, 1.0, 1.0],
    )
    ret = loss_func(output)
    assert "lat_behav_pred_loss" in ret
    assert "lon_behav_pred_loss" in ret


@pytest.mark.parametrize(
    [
        "pred",
        "target",
        "passable_mask",
        "heatmap_loss",
    ],
    [
        pytest.param(
            torch.zeros(2, 192, 192),
            torch.zeros(2, 192, 192),
            None,
            0,
        ),
        pytest.param(
            torch.ones(2, 192, 192),
            torch.ones(2, 192, 192),
            None,
            0,
        ),
        pytest.param(
            torch.ones(2, 192, 192) * 0.5,
            torch.zeros(2, 192, 192),
            torch.ones(2, 192, 192),
            -torch.log(torch.Tensor([0.5 + 1e-12])) * 0.5 ** 2,
        ),
        pytest.param(
            torch.ones(2, 192, 192) * 0.5,
            torch.ones(2, 192, 192) * 0.9,
            torch.ones(2, 192, 192),
            -torch.log(torch.Tensor([0.5 + 1e-12])) * 0.9 ** 4 * 0.5 ** 2,
        ),
    ],
)
def test_http_heatmap_loss(
    pred,
    target,
    passable_mask,
    heatmap_loss,
):
    loss_func = HTTPHeatmapLoss(
        focal_loss=True,
        pos_threshold=0.7,
        hard_neg_threshold=0.1,
        alpha=2,
        beta=4,
        eps=1e-12,
    )
    ret = loss_func(pred, target, passable_mask)
    assert ret - heatmap_loss < 1e-5


@pytest.mark.parametrize(
    ["offset", "offset_gt", "heatmap_gt", "loss"],
    [
        pytest.param(
            torch.from_numpy(np.random.uniform(size=[2, 2, 100, 100])),
            torch.from_numpy(np.random.uniform(size=[2, 2, 100, 100])),
            torch.zeros(2, 100, 100),
            0,
        ),
        pytest.param(
            torch.from_numpy(np.indices([10, 10]) - 5)[None],
            torch.from_numpy(np.indices([10, 10]) - 4)[None],
            torch.ones(2, 10, 10),
            2,
        ),
        pytest.param(
            torch.from_numpy(np.indices([10, 10]) - 5)[None],
            torch.from_numpy(np.indices([10, 10]) - 3)[None],
            torch.ones(2, 10, 10),
            4,
        ),
    ],
)
def test_http_offset_loss(offset, offset_gt, heatmap_gt, loss):
    loss_func = HTTPOffsetLoss(
        pos_threshold=0.7,
        loss_weight=1,
        reduction="mean",
    )
    ret = loss_func(offset, offset_gt, heatmap_gt)
    assert ret - loss < 1e-5


@pytest.mark.parametrize(
    ["confidence", "traj_pred_decoded", "traj_gt", "future_mask", "loss"],
    [
        pytest.param(
            torch.Tensor([[0.5, 0.5]]),
            torch.Tensor([[[[0.5, 0.5], [1, 1]], [[0.8, 0.8], [1.2, 1.2]]]]),
            torch.Tensor([[[0, 0], [1, 1]]]),
            None,
            torch.Tensor([(0.936507170518 + 0.864031767191) / 2]),
        ),
        pytest.param(
            torch.Tensor([[0]]),
            torch.arange(20, dtype=torch.float32).reshape(1, 1, 10, 2),
            torch.arange(20, dtype=torch.float32).reshape(1, 10, 2) + 1,
            None,
            torch.Tensor([6.7175636227866]),
        ),
    ],
)
def test_traj_confidence_loss(
    confidence, traj_pred_decoded, traj_gt, future_mask, loss
):
    loss_func = TrajConfidenceLoss(
        loss_weight=1, pred_norm=False, reduction="mean"
    )
    ret = loss_func(confidence, traj_pred_decoded, traj_gt, future_mask)
    assert ret - loss < 1e-5


@pytest.mark.parametrize(
    [
        "traj_pred",
        "traj_gt",
        "endpoint",
        "future_mask",
        "passable_mask",
        "loss",
    ],
    [
        pytest.param(
            torch.arange(320, dtype=torch.float32).reshape([2, 4, 20, 2]),
            torch.arange(80, dtype=torch.float32).reshape([2, 20, 2]),
            torch.arange(16, dtype=torch.float32).reshape([2, 4, 2]),
            torch.ones(2, 20),
            torch.ones(2, 100, 100),
            torch.Tensor([76040.0547]),
        )
    ],
)
def test_traj_offset_reg_loss(
    traj_pred, traj_gt, endpoint, future_mask, passable_mask, loss
):
    loss_func = TrajOffsetRegLoss(
        sigma=1,
        pos_threshold=2,
        loss_weight=1,
        reduction="mean",
    )
    ret = loss_func(traj_pred, traj_gt, endpoint, future_mask, passable_mask)
    assert ret - loss < 1e-5


def test_MTPLoss():
    ground_truth = torch.stack(
        [
            torch.arange(12) * 0.142857,
            torch.arange(12) * 3.141592653589793,
        ],
        dim=1,
    )[None, ...]

    loss_func = MTPLoss(use_variance=False)

    prediction_full = torch.arange(60).reshape(1, 1, 12, 5).repeat(1, 6, 1, 1)
    prediction_full = prediction_full * torch.arange(1, 7).view(1, 6, 1, 1)
    probs = torch.arange(1, 7).view(1, 6) / torch.arange(1, 7).sum()

    input_dict = {
        "traj": prediction_full,
        "probs": probs,
    }
    target_dict = {
        "traj": ground_truth,
    }

    ret = loss_func(input_dict, target_dict)

    assert float(ret) - 28.9865 < 1e-4

    loss_func.use_variance = True

    ret = loss_func(input_dict, target_dict)

    assert float(ret) - (-0.0476) < 1e-4


@pytest.mark.parametrize(
    ["output"],
    [
        pytest.param(
            {
                "pred_traj": torch.arange(
                    416 * 10, dtype=torch.float32
                ).reshape([13, 10, 8, 4]),
                "pred_cxcympb": torch.arange(
                    416 * 10, dtype=torch.float32
                ).reshape([13, 10, 8, 4]),
                "target_traj": torch.ones(13, 1, 8, 4),
                "target_cxcympb": torch.ones(13, 1, 8, 4),
                "all_goal_trajs": torch.ones(13, 1, 8, 4),
                "goal_cxcympb": torch.ones(13, 1, 8, 4),
                "KLD": torch.tensor([1.0]),
                "probabilities": torch.ones(13, 1, 1, 10),
            }
        )
    ],
)
def test_SGNet_loss(output):
    loss_func = SGNetLoss(
        KLD_weight=1,
        goal_weight=1,
        pred_weight=1,
        prob_weight=1,
        bbox_loss_weight=1,
        iou_loss_weight=1,
        mpb_loss_weight=1,
        eps=1e-6,
    )
    ret = loss_func(output)
    assert "KLD_loss" in ret
    assert "goal_loss" in ret
    assert "pred_loss" in ret
    assert "total_loss" in ret

    loss_func = SGNetLoss(
        KLD_weight=1,
        goal_weight=1,
        pred_weight=1,
        prob_weight=1,
        bbox_loss_weight=1,
        iou_loss_weight=1,
        mpb_loss_weight=1,
        iou_loss_type="ciou_loss",
        eps=1e-6,
    )
    ret = loss_func(output)
    assert "KLD_loss" in ret
    assert "goal_loss" in ret
    assert "pred_loss" in ret
    assert "total_loss" in ret

    loss_func = SGNetLoss(
        KLD_weight=1,
        goal_weight=1,
        pred_weight=1,
        prob_weight=1,
        bbox_loss_weight=1,
        iou_loss_weight=1,
        mpb_loss_weight=1,
        iou_loss_type="giou_loss",
        eps=1e-6,
    )
    ret = loss_func(output)
    assert "KLD_loss" in ret
    assert "goal_loss" in ret
    assert "pred_loss" in ret
    assert "total_loss" in ret

    loss_func = SGNetLoss(
        KLD_weight=1,
        goal_weight=1,
        pred_weight=1,
        prob_weight=1,
        bbox_loss_weight=1,
        iou_loss_weight=1,
        mpb_loss_weight=1,
        iou_loss_type="diou_loss",
        eps=1e-6,
    )
    ret = loss_func(output)
    assert "KLD_loss" in ret
    assert "goal_loss" in ret
    assert "pred_loss" in ret
    assert "total_loss" in ret


@pytest.mark.parametrize(
    ["output"],
    [
        pytest.param(
            {
                "track_valid_cls": torch.rand(32, 3, 1, 1),
                "track_id_gt": torch.ones(32),
                "batch_history_valid_class": torch.ones(32),
            }
        )
    ],
)
def test_track_valid_head_loss(output):
    loss_func = TrackValidHeadLoss(
        head_weight=1,
    )
    ret = loss_func(output)
    assert "track_valid_loss" in ret


def test_densetnt_loss():
    predict_traj_for_train = torch.Tensor(
        [
            [
                [0.9140, 0.3939],
                [0.7222, 0.1991],
                [0.5644, 0.75136],
            ],
            [
                [0.8459, 0.2419],
                [0.2227, 0.8864],
                [0.6678, 0.2443],
            ],
        ]
    )
    predict_traj_for_train = predict_traj_for_train.unsqueeze(1)
    future_trajectories = torch.Tensor(
        [
            [
                [
                    [0.4135, 0.6855],
                    [0.1176, 0.4234],
                    [0.4502, 0.8952],
                ]
            ],
            [
                [
                    [0.9196, 0.7138],
                    [0.3535, 0.3894],
                    [0.0233, 0.7960],
                ]
            ],
        ]
    )
    valid_masks = torch.Tensor(
        [
            [True, True, True],
            [True, True, True],
        ]
    )
    predict_goal_scores = torch.Tensor(
        [
            [
                [0.3, 0.7],
            ],
            [
                [0.8, 0.2],
            ],
        ]
    )
    predict_goal_scores = predict_goal_scores.unsqueeze(1)
    predict_road_scores = torch.Tensor(
        [
            [
                [0.6, 0.4],
            ],
            [
                [0.1, 0.9],
            ],
        ]
    )
    predict_road_scores = predict_road_scores.unsqueeze(1)
    nearest_goal_idxs = torch.Tensor(
        [
            [
                [1],
            ],
            [
                [0],
            ],
        ]
    )
    nearest_goal_idxs = nearest_goal_idxs.unsqueeze(-1)
    nearest_road_idxs = torch.Tensor(
        [
            [
                [1],
            ],
            [
                [0],
            ],
        ]
    )
    nearest_road_idxs = nearest_road_idxs.unsqueeze(-1)
    target_loss = torch.FloatTensor([1.5431])
    loss = DenseTNTloss()
    input = {
        "valid_masks": valid_masks,
        "predict_trajs": predict_traj_for_train,
        "predict_goal_scores": predict_goal_scores,
        "predict_road_scores": predict_road_scores,
        "nearest_goal_idxs": nearest_goal_idxs,
        "nearest_road_idxs": nearest_road_idxs,
        "future_trajectories": future_trajectories,
    }
    cal_loss = loss(input)
    assert torch.abs(target_loss - cal_loss["total_loss"]) < 1e-4
