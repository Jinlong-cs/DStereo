# Copyright (c) Horizon Robotics. All rights reserved.

import os
import pickle
import random

import numpy as np
import pytest
import torch

from hat.metrics.traj_pred_metrics import (
    BehavPredMetric,
    DenseTNTtrajMetric,
    TrajPredFPVPedMetric,
    TrajPredMetric,
)
from projects.prediction.configs.processed_dataset import HAT_UNITTEST_PREFIX
from tests import SD_AlGORITHM_BUCKET_EXISTS, SD_AlGORITHM_BUCKET_PATH


@pytest.mark.skipif(
    not SD_AlGORITHM_BUCKET_EXISTS, reason="requiring SD_Algorithm bucket"
)
def test_traj_pred_metrics():
    data_path = os.path.join(
        SD_AlGORITHM_BUCKET_PATH,
        f"{HAT_UNITTEST_PREFIX}/outputs/prediction_inference_0709/",
    )

    probs = np.load(os.path.join(data_path, "test_output_anchor_probs.npy"))
    means = np.load(
        os.path.join(data_path, "test_output_anchor_mean_vars.npy")
    )
    with open(os.path.join(data_path, "anchors.pkl"), "rb") as f:
        anchors = pickle.load(f)
    num_obj, num_anchors, _, _ = probs.shape
    anchors = anchors[num_anchors][None, :, :, :]
    traj_len = anchors.shape[2]
    means = means.reshape([num_obj, num_anchors, traj_len, -1])
    means = means[:, :, :, 0:2] + anchors
    probs = probs[:, :, 0, 0]
    pred_outputs = {
        "normal_head_pred_trajs": torch.tensor(means),
        "normal_head_gt_trajs": torch.zeros_like(torch.tensor(means)),
        "normal_head_traj_probs": torch.tensor(probs),
        "normal_head_timestep_masks": torch.ones([num_obj, traj_len]),
    }

    k_values = [1, 5, 10, 64]
    name = "normal_head"
    head_name_dict = {
        "traj_head": name,
    }
    metric = TrajPredMetric(
        head_names=head_name_dict, k_values=k_values, name=name
    )
    metric.update(pred_outputs)
    result = metric.get()
    assert result is not None
    for k in k_values:
        ade_metric_name = "head_{}_min_ade_{}".format(name, k)
        fde_metric_name = "head_{}_min_fde_{}".format(name, k)
        miss_rate_metric_name = "head_{}_miss_rate_{}".format(name, k)
        assert ade_metric_name in result[0]
        assert fde_metric_name in result[0]
        assert miss_rate_metric_name in result[0]


@pytest.mark.skipif(
    not SD_AlGORITHM_BUCKET_EXISTS, reason="requiring SD_Algorithm bucket"
)
def test_behav_pred_metrics(tmpdir):
    name = "behav_head"
    save_dir = tmpdir
    output = {
        f"{name}_pred_lat_behav": torch.tensor(
            [random.choice([0, 1, 2]) for _ in range(9)]
        ),
        f"{name}_pred_lon_behav": torch.tensor(
            [random.choice([0, 1, 2]) for _ in range(9)]
        ),
        f"{name}_lat_behav_gts": torch.tensor([0, 1, 2, 0, 0, 1, 0, 0, 2]),
        f"{name}_lon_behav_gts": torch.tensor([0, 1, 2, 0, 0, 1, 0, 0, 2]),
        "tag_array": torch.zeros(9, 13),
    }

    metric = BehavPredMetric(
        name="behav_head",
        F_score_beta=2,
        F_score_weights=[0.2, 0.4, 0.4],
        save_dir=save_dir,
    )
    name = metric.name
    metric.update(output)
    result = metric.get()
    assert result is not None
    names = []
    names.append("{}_global_latbehav_F-score".format(name))
    names.append("{}_global_keeplane_auc".format(name))
    names.append("{}_global_keeplane_F-score".format(name))
    names.append("{}_global_keeplane_recall".format(name))
    names.append("{}_global_lchange_auc".format(name))
    names.append("{}_global_lchange_F-score".format(name))
    names.append("{}_global_lchange_recall".format(name))
    names.append("{}_global_rchange_auc".format(name))
    names.append("{}_global_rchange_F-score".format(name))
    names.append("{}_global_rchange_recall".format(name))
    names.append("{}_global_lonbehav_acc".format(name))
    names.append("{}_global_keepvelo_recall".format(name))
    names.append("{}_global_speedup_recall".format(name))
    names.append("{}_global_slowdown_recall".format(name))

    for name in names:
        assert name in result[0]


@pytest.mark.skipif(
    not SD_AlGORITHM_BUCKET_EXISTS, reason="requiring SD_Algorithm bucket"
)
def test_traj_pred_fpvped_metrics():
    output1 = {
        "pred_traj": torch.randn([13, 10, 8, 4]),
        "pred_cxcympb": torch.randn([13, 10, 8, 4]),
        "target_traj": torch.randn([13, 1, 8, 4]),
        "target_cxcympb": torch.randn([13, 1, 8, 4]),
        "probabilities": torch.randn([13, 1, 1, 10]),
    }
    output2 = {
        "pred_traj": torch.randn([13, 10, 8, 4]),
        "pred_cxcympb": torch.randn([13, 10, 8, 4]),
        "target_traj": torch.randn([13, 1, 8, 4]),
        "target_cxcympb": torch.randn([13, 1, 8, 4]),
        "probabilities": torch.randn([13, 1, 1, 10]),
    }
    metrics_type = {
        "05": 3,
        "10": 5,
        "15": 8,
    }
    top_k_values = [
        (
            1,
            2,
        ),
        (1,),
    ]
    for k_vals in top_k_values:
        for output in [output1, output2]:
            metric = TrajPredFPVPedMetric(
                metrics_type=metrics_type, top_k_values=k_vals
            )
            name = metric.name
            metric.update(output)
            ret_names, ret_values = metric.get()
            assert ret_names is not None
            assert ret_values is not None
            metrics_type = metric.metrics_type
            k_vals = metric.top_k_values
            names = []
            for metric in metrics_type:
                for _, k in enumerate(k_vals):
                    for _, stat in enumerate(["avg", "90th", "95th"]):
                        names.append(
                            "{}_min_rmse{}_{}_{}".format(name, metric, k, stat)
                        )
                        names.append(
                            "{}_min_ade_center{}_{}_{}".format(
                                name, metric, k, stat
                            )
                        )
                        names.append(
                            "{}_min_ade_mpb{}_{}_{}".format(
                                name, metric, k, stat
                            )
                        )
                        names.append(
                            "{}_min_fde_center{}_{}_{}".format(
                                name, metric, k, stat
                            )
                        )
                        names.append(
                            "{}_min_fde_mpb{}_{}_{}".format(
                                name, metric, k, stat
                            )
                        )
            assert len(ret_values) == len(k_vals) * len(metrics_type) * 5 * 3
            for name in names:
                assert name in ret_names
            for name in ret_names:
                assert name in names


def test_densetnt_traj_pred_metrics():
    test_input = {
        "end_points": torch.randn([20, 2, 1, 1]),
        "goal_coords": torch.randn([20, 2, 1, 2048]),
        "predict_trajs": torch.randn([20, 5, 12, 2]),
        "predict_goal_scores": torch.randn([20, 1, 1, 2048]),
        "real_scores": torch.randn([20, 5, 1, 1]),
        "future_trajectories": torch.randn([20, 1, 12, 2]),
        "valid_masks": torch.ones([20, 12]).bool(),
    }
    k_values = [1, 5]
    name = "normal_head"
    head_name_dict = {
        "traj_head": name,
    }

    metric = DenseTNTtrajMetric(
        head_names=head_name_dict,
        k_values=k_values,
        name=name,
        goal_coords_scale=0.04,
    )
    metric.update(test_input)
    result = metric.get()
    assert result is not None
    for k in k_values:
        ade_metric_name = "head_{}_min_ade_{}".format(name, k)
        fde_metric_name = "head_{}_min_fde_{}".format(name, k)
        miss_rate_metric_name = "head_{}_miss_rate_{}".format(name, k)
        assert ade_metric_name in result[0]
        assert fde_metric_name in result[0]
        assert miss_rate_metric_name in result[0]
