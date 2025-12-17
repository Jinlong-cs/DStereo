# Copyright (c) Horizon Robotics. All rights reserved.

import os

import pytest
import torch

from hat.callbacks.planning_viz import PlanningMetricUpdater, PlanningViz
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_planning_viz():
    bs = 4
    img_h, img_w, img_resolution = 512, 512, 0.2
    grid_dim = 65
    bev_ego_center = (362, 256)
    ckpt_dir = os.path.join(
        HAT_BUCKET_PATH,
        "users/bikun.wang/unit_test/viz_outputs",
    )

    image_dir = os.path.join(ckpt_dir, "image")

    viz_callback = PlanningViz(
        height=img_h,
        width=img_w,
        resolution=img_resolution,
        bev_ego_center=bev_ego_center,
        video_save_path=ckpt_dir,
        video_name="viz_planning",
        video_fps=1,
        viz_on_every_epoch=False,
        viz_epoch=0,
    )
    rendered_map = torch.randn(bs, 3, img_h, img_w)
    rendered_obs = torch.randn(bs, 4, img_h, img_w)
    rendered_ego = torch.randn(bs, 4, img_h, img_w)
    svf_goal = torch.randn(bs, 1, grid_dim, grid_dim)
    plan_fut_state = torch.randn(bs, 12, 6)
    batch_timestamp = [1000, 2000, 3000, 4000]
    svf = torch.randn(bs, 1, grid_dim, grid_dim)

    batch_data = {
        "rendered_map": rendered_map,
        "rendered_obs": rendered_obs,
        "plan_rendered_ego": rendered_ego,
        "plan_svf_goal": svf_goal,
        "plan_fut_state": plan_fut_state,
        "lcf_timestamp": batch_timestamp,
    }

    batch_out = {"reward_head_svf": svf}

    viz_callback.render(batch_data, batch_out)

    assert os.path.exists(os.path.join(image_dir, "1000.jpg"))
    assert os.path.exists(os.path.join(image_dir, "2000.jpg"))
    assert os.path.exists(os.path.join(image_dir, "3000.jpg"))
    assert os.path.exists(os.path.join(image_dir, "4000.jpg"))


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_planning_metric_updater():
    ckpt_dir = os.path.join(
        HAT_BUCKET_PATH,
        "users/bikun.wang/unit_test/viz_outputs",
    )

    def update_metric(metrics, batch, model_outs):
        for metric in metrics:
            metric.update(model_outs)

    metric_updater = PlanningMetricUpdater(
        metric_update_func=update_metric,
        step_log_freq=10,
        epoch_log_freq=1,
        log_prefix="Validation_uni_test",
        save_metric_path=ckpt_dir,
    )

    log = metric_updater._get_one_group_log([], [])

    assert len(log) > 0
    assert os.path.exists(os.path.join(ckpt_dir, "metric.txt"))
