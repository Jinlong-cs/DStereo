# Copyright (c) Horizon Robotics. All rights reserved.

import numpy as np
import torch

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list
from .metric import EvalMetric


@OBJECT_REGISTRY.register
class Eye3dPoseAngleAndDist(EvalMetric):
    """Compute average angle and depth difference.

    Args:
        name: name of this metric instance for display
        num_mode: number of modes/groups training in network
        std: std of depth label, for scaling
        is_3d_error_normed: is 3d position error calculated in
            normed coordinate
    """

    def __init__(
        self, is_seperate=False, num_mode=1, std=1, is_3d_error_normed=True
    ):
        self.num_mode = num_mode
        self.std = std
        self.is_seperate = is_seperate
        self.is_3d_error_normed = is_3d_error_normed
        names = []
        if is_seperate:
            for name_list in [[*"rpy"], "lx ly lz rx ry rz".split()]:
                for i in range(num_mode):
                    for subname in name_list:
                        names.append(f"{subname}_mode_{i}")
        else:
            name_list = ["pose", "depth"]
            for subname in name_list:
                for i in range(num_mode):
                    names.append(f"{subname}_mode_{i}")
        super().__init__(names)

    def update(self, labels, preds):
        labels = _as_list(labels)
        preds = _as_list(preds)
        for label, pred in zip(labels, preds):
            batch = label["gt_pose"].shape[0]
            if self.is_seperate:
                diff_pose = (
                    torch.abs(label["gt_pose"] - pred["pred_pose"])
                    .sum(0)
                    .reshape(-1)
                )
                diff_depth = torch.abs(
                    label["gt_eye3d"] - pred["pred_eye3d"]
                ).reshape(batch, self.num_mode, 3)[..., :2]
                xyz = label["gt_eye3d_xyz"].reshape(
                    batch, self.num_mode * 2, 3
                )
                vec = xyz / xyz[..., -1:]
                diff_norm = (
                    vec.reshape(batch, self.num_mode, 2, 3)
                    * diff_depth[..., None]
                )
                if not self.is_3d_error_normed:
                    rot = label["gt_norm_rot"].reshape(
                        batch, self.num_mode, 3, 3
                    )
                    diff_xyz = (
                        torch.einsum(
                            "inab,inea->ineb", rot, diff_norm.to(torch.float32)
                        )
                        .abs()
                        .sum(0)
                        .reshape(-1)
                    )
                    diff_sum = torch.concat([diff_pose, diff_xyz])
                else:
                    diff_sum = torch.concat(
                        [diff_pose, diff_norm.abs().sum(0).reshape(-1)]
                    )
            else:
                diff = torch.concat(
                    [
                        label["gt_pose"] - pred["pred_pose"],
                        label["gt_eye3d"] - pred["pred_eye3d"],
                    ],
                    dim=1,
                )
                diff = torch.abs(diff)
                diff_sum = diff.sum(0).reshape(-1, 3).mean(-1)
                diff_sum[self.num_mode :] *= 1.5
            self.sum_metric = self.sum_metric + diff_sum
            self.num_inst += batch

    def compute(self):
        val = super().compute()
        if self.is_seperate:
            val = np.array(val)
            val[: 3 * self.num_mode] *= 90
            val[3 * self.num_mode :] *= self.std
        else:
            val = np.array(val).reshape(2, -1)
            val = val * np.array([90, self.std]).reshape(2, 1)
        return val.reshape(-1).tolist()
