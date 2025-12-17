# Copyright (c) Horizon Robotics. All rights reserved.

import numpy as np
import torch

from hat.data.collates.planning_collates import collate_p3c_plan


def test_collate_p3c_plan():

    batchsize = 4
    context_fram = 4
    traj_len = 12
    img_h = 512
    img_w = 512
    state_dim = 6
    grid_dim = 65
    horizon = 50
    action = 9

    road_map = np.random.random((img_h, img_w))
    rendered_obs = np.random.random((img_h, img_w, context_fram))
    rendered_obs_fut = torch.from_numpy(
        np.random.random((img_h, img_w, traj_len))
    )
    plan_fut_state = np.random.random((traj_len, state_dim))
    plan_his_state = np.random.random((context_fram, state_dim))
    plan_rendered_ego = np.random.random((img_h, img_w, traj_len))
    plan_svf_path = np.random.random((1, grid_dim, grid_dim))
    plan_svf_goal = np.random.random((1, grid_dim, grid_dim))
    plan_waypts_expert = np.random.random((horizon, 2))
    plan_grid_idcs = np.random.random((horizon, 2))
    plan_bc_targets = np.random.random((horizon, action, grid_dim, grid_dim))
    plan_ego_motion = np.random.random((2, grid_dim, grid_dim))

    data = {
        "road_map": road_map,
        "rendered_obs": rendered_obs,
        "rendered_obs_fut": rendered_obs_fut,
        "plan_fut_state": plan_fut_state,
        "plan_his_state": plan_his_state,
        "plan_rendered_ego": plan_rendered_ego,
        "plan_svf_path": plan_svf_path,
        "plan_svf_goal": plan_svf_goal,
        "plan_waypts_expert": plan_waypts_expert,
        "plan_grid_idcs": plan_grid_idcs,
        "plan_bc_targets": plan_bc_targets,
        "plan_ego_motion": plan_ego_motion,
        "lcf_timestamp": 123,
        "date_token": "123",
    }

    batch = []
    for _ in range(batchsize):
        batch.append(data)

    batch = collate_p3c_plan(batch)

    assert "road_map" in batch
    assert "rendered_obs" in batch
    assert "rendered_obs_fut" in batch
    assert "plan_fut_state" in batch
    assert "plan_his_state" in batch
    assert "plan_rendered_ego" in batch
    assert "plan_svf_path" in batch
    assert "plan_svf_goal" in batch
    assert "plan_waypts_expert" in batch
    assert "plan_grid_idcs" in batch
    assert "plan_bc_targets" in batch
    assert "plan_ego_motion" in batch
    assert "lcf_timestamp" in batch
    assert "date_token" in batch

    assert batch["road_map"].shape == (batchsize, 1, img_h, img_w)
    assert batch["rendered_obs"].shape == (
        batchsize,
        context_fram,
        img_h,
        img_w,
    )
    assert batch["rendered_obs_fut"].shape == (
        batchsize,
        traj_len,
        img_h,
        img_w,
    )
    assert batch["plan_fut_state"].shape == (batchsize, traj_len, state_dim)
    assert batch["plan_his_state"].shape == (
        batchsize,
        state_dim,
        1,
        context_fram,
    )
    assert batch["plan_rendered_ego"].shape == (
        batchsize,
        traj_len,
        img_h,
        img_w,
    )
    assert batch["plan_svf_path"].shape == (batchsize, 1, grid_dim, grid_dim)
    assert batch["plan_svf_goal"].shape == (batchsize, 1, grid_dim, grid_dim)
    assert batch["plan_waypts_expert"].shape == (batchsize, horizon, 2)
    assert batch["plan_grid_idcs"].shape == (batchsize, horizon, 2)
    assert batch["plan_bc_targets"].shape == (
        batchsize,
        horizon,
        action,
        grid_dim,
        grid_dim,
    )
    assert batch["plan_ego_motion"].shape == (batchsize, 2, grid_dim, grid_dim)
