# Copyright (c) Horizon Robotics. All rights reserved.

from collections import OrderedDict
from typing import Optional

import torch

from hat.data.datasets.traj_pred_dataset import BaseTrajDataset
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "UniformpathPostProcessor",
]


@OBJECT_REGISTRY.register
class UniformpathPostProcessor(torch.nn.Module):
    """The post processing of the Uniformpath traj prediction model.

    This processor should be used in the `Uniformpath` structure.
    """

    def __init__(
        self,
        ego_track_id: int = -BaseTrajDataset.ANSWER,
    ):
        """Initialize method."""
        super(UniformpathPostProcessor, self).__init__()
        self.ego_track_id = ego_track_id

    @torch.no_grad()
    def forward(self, model_output, anchor_num: Optional[int] = None):
        """Post processing forward.

        The post processing contains the following steps, the
        "switches" are defined in the initialization method:
        1. Process for metric calculation.

        Args:
            model_output (Dict): the model output dictionary.
            anchor_num (int): the number of anchor.

        Return:
            results (Dict): the dictionary of model trajectory
                for visualization and metric calculation. It contains
                the following keys:
                ['cur_head_mask', 'trajectories', 'means',
                'log_anchors_probs', 'gts', 'masks']
        """
        valid_ids = model_output["filtered_obs_ids"]
        means = model_output["filtered_obs_trajs"]
        anchor_probs = model_output["log_anchors_probs"]
        gts = model_output["filtered_obs_gt"]
        masks = model_output["filtered_obs_gt_masks"]

        valid_ids = [
            track_id for track_ids in valid_ids for track_id in track_ids
        ]
        means = torch.stack([means] * anchor_num, dim=1)
        gts = torch.nan_to_num(gts)
        head_mask = torch.ones([masks.shape[0]], dtype=masks.dtype)
        for i, obs_id in enumerate(valid_ids):
            if obs_id == self.ego_track_id:
                head_mask[i] = False
        trajectories = means[:, :, None, :, :]

        metric_info = {
            "pred_trajs": means[head_mask].cpu().detach(),
            "traj_probs": torch.exp(anchor_probs[head_mask]).cpu().detach(),
            "gt_trajs": torch.stack([gts[head_mask]] * means.shape[1], dim=1)
            .cpu()
            .detach(),
            "timestep_masks": masks[head_mask].cpu().detach(),
        }

        results = OrderedDict()
        results["cur_head_mask"] = head_mask
        results["trajectories"] = trajectories
        results["means"] = means
        results["log_anchors_probs"] = anchor_probs
        results["gts"] = gts
        results["masks"] = masks
        results.update(metric_info)
        return results
