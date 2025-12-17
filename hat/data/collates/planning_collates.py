# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Dict, List

import numpy as np
import torch

__all__ = [
    "collate_p3c_plan",
]


def collate_p3c_plan(samples: List[Dict]) -> Dict:
    """Collate function for imitation plan model.

    Args:
        batch: a list of dataloader output.
    Returns:
        batch_data: the dictionary that contains necessary
            information for imitation planning.
    """

    required_key = [
        "road_map",
        "rendered_obs",
        "rendered_obs_fut",
        "plan_fut_state",
        "plan_his_state",
        "plan_rendered_ego",
        "plan_svf_path",
        "plan_svf_goal",
        "plan_waypts_expert",
        "plan_grid_idcs",
        "plan_bc_targets",
        "plan_ego_motion",
    ]

    optional_key = ["lcf_timestamp", "date_token"]

    batch_data = {i: [] for i in required_key + optional_key}

    # 1. Unpack the batch data.
    for item in samples:
        for key in required_key:
            if key in item:
                batch_data[key].append(item[key])
            else:
                raise ValueError(
                    f"The batch data do not have the required key {key}."
                )
        for key in optional_key:
            if key in item:
                batch_data[key].append(item[key])

    # 2. Tensor packing and type conversion.
    # 2.1. Road map:
    #   [[H, W]] -> [batch_size, 1, H, W]
    #   [[H, W, C]] -> [batch_size, C, H, W]
    batch_road_maps = torch.FloatTensor(np.stack(batch_data["road_map"], 0))
    if len(batch_road_maps.shape) == 3:
        batch_data["road_map"] = torch.unsqueeze(batch_road_maps, 1)
    elif len(batch_road_maps.shape) == 4:
        batch_data["road_map"] = batch_road_maps.permute(
            (0, 3, 1, 2)
        ).contiguous()
    else:
        raise ValueError(
            "The shape of road maps should be either [H, W] or [H, W, C]."
        )

    # 2.2. Rendered obstacle OG.
    #   [[H, W, C]] -> [batch_size, C, H, W]
    batch_rendered_obs = torch.FloatTensor(
        np.stack(batch_data["rendered_obs"], 0)
    )
    batch_data["rendered_obs"] = batch_rendered_obs.permute(
        (0, 3, 1, 2)
    ).contiguous()

    # 2.3. Ground truth future state: [batch_size, traj_len, 6]
    batch_fut_states = [
        torch.unsqueeze(torch.FloatTensor(gt), 0)
        for gt in batch_data["plan_fut_state"]
    ]
    batch_data["plan_fut_state"] = torch.cat(batch_fut_states, dim=0)

    # 2.4. Ground truth history state: [batch_size, 4, 5]
    batch_his_states = [
        torch.unsqueeze(torch.FloatTensor(gt), 0)
        for gt in batch_data["plan_his_state"]
    ]
    batch_his_states = torch.cat(batch_his_states, dim=0)
    batch_data["plan_his_state"] = batch_his_states.permute(0, 2, 1).unsqueeze(
        2
    )

    # 2.5. Rendered ego.
    #   [[H, W, C]] -> [batch_size, C, H, W]
    batch_rendered_ego = torch.FloatTensor(
        np.stack(batch_data["plan_rendered_ego"], 0)
    )
    batch_data["plan_rendered_ego"] = batch_rendered_ego.permute(
        (0, 3, 1, 2)
    ).contiguous()

    # 2.6. svf_expert: [batch_size, 1, grid_dim,grid_dim]   path
    batch_svf_path = [
        torch.unsqueeze(torch.FloatTensor(gt), 0)
        for gt in batch_data["plan_svf_path"]
    ]
    batch_data["plan_svf_path"] = torch.cat(batch_svf_path, dim=0)

    # 2.7. svf_expert: [batch_size, 1, grid_dim,grid_dim]   goal
    batch_svf_goal = [
        torch.unsqueeze(torch.FloatTensor(gt), 0)
        for gt in batch_data["plan_svf_goal"]
    ]
    batch_data["plan_svf_goal"] = torch.cat(batch_svf_goal, dim=0)

    # 2.9. waypts_expert: [batch_size, horizon, 2]
    batch_waypts_expert = [
        torch.unsqueeze(torch.FloatTensor(gt), 0)
        for gt in batch_data["plan_waypts_expert"]
    ]
    batch_data["plan_waypts_expert"] = torch.cat(batch_waypts_expert, dim=0)

    # 2.10. grid_idcs: [batch_size, horizon, 2]
    batch_grid_idcs = [
        torch.unsqueeze(torch.FloatTensor(gt), 0)
        for gt in batch_data["plan_grid_idcs"]
    ]
    batch_data["plan_grid_idcs"] = torch.cat(batch_grid_idcs, dim=0)

    # 2.11. bc_targets: [batch_size,horizon,action_num,grid_dim,grid_dim]
    batch_bc_targets = [
        torch.unsqueeze(torch.FloatTensor(gt).bool(), 0)
        for gt in batch_data["plan_bc_targets"]
    ]
    batch_data["plan_bc_targets"] = torch.cat(batch_bc_targets, dim=0)

    # 2.12. Rendered obstacle OG in future.
    #   [[H, W, C]] -> [batch_size, C, H, W]
    batch_rendered_obs_fut = torch.FloatTensor(
        np.stack(batch_data["rendered_obs_fut"], 0)
    )
    batch_data["rendered_obs_fut"] = batch_rendered_obs_fut.permute(
        (0, 3, 1, 2)
    ).contiguous()

    # 2.13. Rendered ego motion.
    #   [[C, H, W]] -> [batch_size, C, H, W]
    batch_plan_ego_motion = [
        torch.unsqueeze(torch.FloatTensor(gt), 0)
        for gt in batch_data["plan_ego_motion"]
    ]
    batch_data["plan_ego_motion"] = torch.cat(batch_plan_ego_motion, dim=0)

    return batch_data
