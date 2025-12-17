import torch

from hat.metrics.planning_metrics import PlanningRewardMetric


def test_planning_metric():
    plan_metric = PlanningRewardMetric(
        name=["svf_diff", "cdf_dist"],
    )

    bs = 4
    grid_dim = 65

    pred_svf = torch.randn(bs, 1, grid_dim, grid_dim)
    target_svf = torch.randn(bs, 1, grid_dim, grid_dim)

    outputs = {"reward_head_svf": pred_svf, "plan_svf_path": target_svf}

    plan_metric.update(outputs)

    assert hasattr(plan_metric, "svf_diff")
    assert hasattr(plan_metric, "cdf_dist")
    assert plan_metric.svf_diff[0] == torch.abs(pred_svf - target_svf).mean()
