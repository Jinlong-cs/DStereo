# Copyright (c) Horizon Robotics. All rights reserved.

import dataclasses
import gzip
import os
import pickle
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from hat.registry import OBJECT_REGISTRY

__all__ = ["OccflowWaymoDataset"]


# Holds num_waypoints occupancy and flow tensors for one agent class.
@dataclasses.dataclass
class _WaypointGridsOneType:
    """Sequence of occupancy and flow tensors for one agent type."""

    # num_waypoints tensors shaped [batch_size, height, width, 1]
    observed_occupancy: List[torch.Tensor] = dataclasses.field(
        default_factory=list
    )
    # num_waypoints tensors shaped [batch_size, height, width, 1]
    occluded_occupancy: List[torch.Tensor] = dataclasses.field(
        default_factory=list
    )
    # num_waypoints tensors shaped [batch_size, height, width, 2]
    flow: List[torch.Tensor] = dataclasses.field(default_factory=list)
    # The origin occupancy for each flow waypoint.  Notice that a flow field
    # transforms some origin occupancy into some destination occupancy.
    # Flow-origin occupancies are the base occupancies for each flow field.
    # num_waypoints tensors shaped [batch_size, height, width, 1]
    flow_origin_occupancy: List[torch.Tensor] = dataclasses.field(
        default_factory=list
    )


@dataclasses.dataclass
class AgentGrids:
    """Contains any topdown render for vehicles and pedestrians."""

    vehicles: Optional[torch.Tensor] = None
    pedestrians: Optional[torch.Tensor] = None
    cyclists: Optional[torch.Tensor] = None

    def view(self, agent_type: str) -> torch.Tensor:
        """Retrieve topdown tensor for given agent type."""
        if agent_type == "vehicles":
            return self.vehicles
        elif agent_type == "pedestrians":
            return self.pedestrians
        elif agent_type == "cyclists":
            return self.cyclists
        else:
            raise ValueError(f"Unknown agent type:{agent_type}.")


# Holds num_waypoints occupancy and flow tensors for all agent clases. This is
# used to store both ground-truth and predicted topdowns.
@dataclasses.dataclass
class WaypointGrids:
    """Occupancy and flow sequences for vehicles, pedestrians, cyclists."""

    vehicles: _WaypointGridsOneType = dataclasses.field(
        default_factory=_WaypointGridsOneType
    )
    pedestrians: _WaypointGridsOneType = dataclasses.field(
        default_factory=_WaypointGridsOneType
    )
    cyclists: _WaypointGridsOneType = dataclasses.field(
        default_factory=_WaypointGridsOneType
    )

    def view(self, agent_type: str) -> _WaypointGridsOneType:
        """Retrieve occupancy and flow sequences for given agent type."""
        if agent_type == "vehicles":
            return self.vehicles
        elif agent_type == "pedestrians":
            return self.pedestrians
        elif agent_type == "cyclists":
            return self.cyclists
        else:
            raise ValueError(f"Unknown agent type:{agent_type}.")

    def get_observed_occupancy_at_waypoint(self, k: int) -> AgentGrids:
        """Return occupancies of currently-observed agents at waypoint k."""
        agent_grids = AgentGrids()
        if self.vehicles.observed_occupancy:
            agent_grids.vehicles = self.vehicles.observed_occupancy[k]
        if self.pedestrians.observed_occupancy:
            agent_grids.pedestrians = self.pedestrians.observed_occupancy[k]
        if self.cyclists.observed_occupancy:
            agent_grids.cyclists = self.cyclists.observed_occupancy[k]
        return agent_grids

    def get_occluded_occupancy_at_waypoint(self, k: int) -> AgentGrids:
        """Return occupancies of currently-occluded agents at waypoint k."""
        agent_grids = AgentGrids()
        if self.vehicles.occluded_occupancy:
            agent_grids.vehicles = self.vehicles.occluded_occupancy[k]
        if self.pedestrians.occluded_occupancy:
            agent_grids.pedestrians = self.pedestrians.occluded_occupancy[k]
        if self.cyclists.occluded_occupancy:
            agent_grids.cyclists = self.cyclists.occluded_occupancy[k]
        return agent_grids

    def get_observed_occluded_veh_occupancy_at_waypoint(
        self, k: int
    ) -> AgentGrids:
        """Return occupancies of currently-observed agents at waypoint k."""
        agent_grids = AgentGrids()
        if self.vehicles.observed_occupancy:
            agent_grids.vehicles = self.vehicles.observed_occupancy[k]
        if self.vehicles.occluded_occupancy:
            agent_grids.pedestrians = self.vehicles.occluded_occupancy[k]
        return agent_grids

    def get_flow_at_waypoint(self, k: int) -> AgentGrids:
        """Return flow fields of all agents at waypoint k."""
        agent_grids = AgentGrids()
        if self.vehicles.flow:
            agent_grids.vehicles = self.vehicles.flow[k]
        if self.pedestrians.flow:
            agent_grids.pedestrians = self.pedestrians.flow[k]
        if self.cyclists.flow:
            agent_grids.cyclists = self.cyclists.flow[k]
        return agent_grids


def convert_to_waypoints(
    occupancy: torch.Tensor,
    flow: torch.Tensor,
    flow_origin_occupancy: Optional[torch.Tensor] = None,
) -> WaypointGrids:
    """Generate waypoint grids for prediction or ground truth.

    Only ground truth has flow_origin_occupancy.

    """
    batch, H, W, _ = occupancy.shape
    occupancy = occupancy.view((batch, H, W, 2, -1, 8))
    flow = flow.view((batch, H, W, 2, -1, 8))
    if flow_origin_occupancy is not None:
        flow_origin_occupancy = flow_origin_occupancy.view(
            (batch, H, W, 1, -1, 8)
        )
    waypoint_logits = WaypointGrids()
    # Slice channels into output predictions.
    num_class = occupancy.shape[4]
    class_names = ["vehicles", "pedestrians", "cyclists"]
    for i in range(num_class):
        class_name = class_names[i]
        for k in range(8):
            observed_occupancy = occupancy[:, :, :, 0:1, i, k]
            occluded_occupancy = occupancy[:, :, :, 1:2, i, k]
            gt_flow = flow[:, :, :, :, i, k]
            getattr(waypoint_logits, class_name).observed_occupancy.append(
                observed_occupancy
            )
            getattr(waypoint_logits, class_name).occluded_occupancy.append(
                occluded_occupancy
            )
            getattr(waypoint_logits, class_name).flow.append(gt_flow)
            if flow_origin_occupancy is not None:
                gt_flow_origin_occupancy = flow_origin_occupancy[
                    :, :, :, :, i, k
                ]
                getattr(
                    waypoint_logits, class_name
                ).flow_origin_occupancy.append(gt_flow_origin_occupancy)
    return waypoint_logits


@OBJECT_REGISTRY.register_module
class OccflowWaymoDataset(Dataset):
    """Dataset for Occupancy and Flow Prediction.

    Currently only support reading offline(pre-generated) data.

    Args:
        dataroot: The path of offline data.
        token_list: The path of token list file.
        subsample: Subsample rate of data.
        is_train: If in training stage, vis_grids which is for
            visualization purpose is not loaded.
        enable_sparse: Use occupancy sparse input.
        enable_sparse_with_road: Use roadmap sparse input.
        sparse_dataroot: The path of offline sparse data.

    """

    def __init__(
        self,
        dataroot: str,
        token_list: str,
        subsample: int = 1,
        is_train: bool = True,
        enable_sparse: bool = False,
        enable_sparse_with_road: bool = False,
        sparse_dataroot: str = None,
    ):
        self.dataroot = dataroot
        self.token_filename = os.path.join(dataroot, token_list)
        self.token_list = []
        with open(self.token_filename) as f:
            for line in f:
                self.token_list.append(line.rstrip())
        self.token_list = self.token_list[::subsample]
        self.is_train = is_train

        self.enable_sparse = enable_sparse
        self.enable_sparse_with_road = enable_sparse_with_road
        self.sparse_dataroot = sparse_dataroot

    def gload(self, filename):
        file = gzip.GzipFile(filename, "rb")
        res = pickle.load(file)
        file.close()
        return res

    def __len__(self):
        return len(self.token_list)

    def __getitem__(self, index):
        filename = os.path.join(self.dataroot, self.token_list[index])
        data = self.gload(filename)

        if self.is_train and "output_vis_grids.agent_trails" in data:
            del data["output_vis_grids.agent_trails"]

        if self.enable_sparse:
            sparse_filename = os.path.join(
                self.sparse_dataroot, self.token_list[index]
            )
            sparse_data = self.gload(sparse_filename)
            if self.enable_sparse_with_road:
                sparse_feature, coords = sparse_data["inputs"]["sparse_inputs"]
            else:
                sparse_feature = sparse_data["inputs"]["model_inputs"][0]
                coords = sparse_data["inputs"]["model_inputs"][2]

            data["inputs"]["sparse_inputs"] = (sparse_feature, coords)

        # remove 34 - 56 channel (unbounded angle input)
        data["inputs"]["model_inputs"] = np.concatenate(
            [
                data["inputs"]["model_inputs"][:34],
                data["inputs"]["model_inputs"][56:],
            ],
            axis=0,
        )
        return_data = {
            "raster_inputs": data["inputs"]["model_inputs"].astype(np.float32),
            "scenario_id": data["scenario_id"],
            "ground_truth": data["ground_truth"],
        }
        return return_data
