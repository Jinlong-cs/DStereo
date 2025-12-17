# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from hat.core.face3d.lbs import batch_rodrigues, rot_mat_to_euler
from hat.registry import OBJECT_REGISTRY
from .metric import EvalMetric

__all__ = ["PoseMAE", "Eye3dMAE"]


def transfer_head_pose_axes(pose_euler: Union[torch.Tensor, np.ndarray]):
    if isinstance(pose_euler, torch.Tensor):
        transfer = torch
    elif isinstance(pose_euler, np.ndarray):
        transfer = np
    else:
        raise ValueError("Not supported input format.")
    pose_euler[:, 0] = transfer.where(
        pose_euler[:, 0] < 0,
        (pose_euler[:, 0] + 180) * -1,
        (pose_euler[:, 0] - 180) * -1,
    )

    pose_euler[:, 2] = transfer.where(
        pose_euler[:, 2] < 0,
        pose_euler[:, 2] + 180,
        pose_euler[:, 2] - 180,
    )
    return pose_euler


@OBJECT_REGISTRY.register
class PoseMAE(EvalMetric):
    """Computes MAE score for pose rpy.

    Args:
        name: Task name of this metric instance for display.
        modeltype: set as wpp or pp. When set as pp, output pose vector will be
        converted to euler by rule. Defalut to wpp.
    """

    def __init__(
        self, model_type: str = "wpp", keys: Optional[List[str]] = None
    ):
        assert model_type in ["wpp", "pp"]
        self.model_type = model_type
        name = ["roll", "pitch", "yaw", "MAE"]
        self.keys = keys
        if keys is not None:
            name = name + [f"{key}_{n}" for key in keys for n in name]
        super(PoseMAE, self).__init__(name)

    def _init_states(self):
        """Initialize state variables.

        It is generally recommended to create state variables with
        add_state method, since in that case synchronization and reset
        can be handled automatically.

        State variables manually added with `self.xxx = yyy` cannot be
        synchronized and thus cannot be used in distributed case. Besides,
        they need to be manually reset by extending the reset method.
        """
        self.add_state(
            "sum_metric",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "num_inst",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        if self.keys is not None:
            for key in self.keys:
                self.add_state(
                    f"sum_metric_{key}",
                    default=torch.tensor(0.0),
                    dist_reduce_fx="sum",
                )
                self.add_state(
                    f"num_inst_{key}",
                    default=torch.tensor(0.0),
                    dist_reduce_fx="sum",
                )

    def update(self, labels: Dict, preds: Dict):
        preds_rvec = preds["global_pose"].reshape(-1, 3)
        preds_mat = batch_rodrigues(preds_rvec)
        if self.model_type == "pp":
            vir2real_rotmat = labels.get(
                "vir2real_rotmat", torch.eye(3, device=preds_mat.device)
            )
            # convert preds rot mat from virtual camera to real camera.
            preds_mat = vir2real_rotmat @ preds_mat
        preds_euler = rot_mat_to_euler(preds_mat, "zxy")
        if self.model_type == "pp":
            preds_euler = transfer_head_pose_axes(preds_euler)
        if self.model_type == "wpp":
            preds_euler = -preds_euler

        gt_pose = labels["gt_pose"]
        error = torch.sum(abs(preds_euler - gt_pose), dim=0)
        self.num_inst = self.num_inst + gt_pose.shape[0]
        self.sum_metric = self.sum_metric + error

        if self.keys is not None:
            img_keys = labels["img_path"]
            error = abs(preds_euler - gt_pose)
            for idx, img_key in enumerate(img_keys):
                for key in self.keys:
                    if key in img_key:
                        setattr(
                            self,
                            f"sum_metric_{key}",
                            getattr(self, f"sum_metric_{key}") + error[idx],
                        )  # noqa
                        setattr(
                            self,
                            f"num_inst_{key}",
                            getattr(self, f"num_inst_{key}") + 1,
                        )  # noqa

    def compute(self):
        val = self.sum_metric / self.num_inst
        val = val.cpu().numpy().tolist()
        val = val + [np.mean(val)]
        if self.keys is not None:
            for key in self.keys:
                v = getattr(self, f"sum_metric_{key}")
                n = getattr(self, f"num_inst_{key}")
                v = (v / n).cpu().numpy()
                v = v.tolist() + [np.mean(v).round(4)]
                val.extend(v)
        return val


@OBJECT_REGISTRY.register
class Eye3dMAE(EvalMetric):
    """Computes eye 3d location MAE."""

    def __init__(self, keys=None):
        name = ["left_x", "left_y", "left_z", "right_x", "right_y", "rigth_z"]
        self.keys = keys
        if keys is not None:
            name = name + [f"{key}_{n}" for key in keys for n in name]
        super(Eye3dMAE, self).__init__(name)

    def _init_states(self):
        """Initialize state variables.

        It is generally recommended to create state variables with
        add_state method, since in that case synchronization and reset
        can be handled automatically.

        State variables manually added with `self.xxx = yyy` cannot be
        synchronized and thus cannot be used in distributed case. Besides,
        they need to be manually reset by extending the reset method.
        """
        self.add_state(
            "sum_metric",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "num_inst",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        if self.keys is not None:
            for key in self.keys:
                self.add_state(
                    f"sum_metric_{key}",
                    default=torch.tensor(0.0),
                    dist_reduce_fx="sum",
                )
                self.add_state(
                    f"num_inst_{key}",
                    default=torch.tensor(0.0),
                    dist_reduce_fx="sum",
                )

    def update(self, labels: Dict, preds: Dict):
        pred_left = preds["eye3d_left"].reshape(-1, 3)
        pred_right = preds["eye3d_right"].reshape(-1, 3)
        vir2real_rotmat = labels.get(
            "vir2real_rotmat", torch.eye(3, device=pred_left.device)
        )
        pred_left = (vir2real_rotmat @ pred_left[..., None]).squeeze(-1)
        pred_right = (vir2real_rotmat @ pred_right[..., None]).squeeze(-1)

        gt_left = labels["eye3d_left"]
        gt_right = labels["eye3d_right"]

        pred_eye3d = torch.cat([pred_left, pred_right], -1)
        gt_eye3d = torch.cat([gt_left, gt_right], -1)
        error = abs(pred_eye3d - gt_eye3d).sum(0)
        self.num_inst = self.num_inst + gt_eye3d.shape[0]
        self.sum_metric = self.sum_metric + error
        if self.keys is not None:
            img_keys = labels["img_path"]
            error = abs(pred_eye3d - gt_eye3d)
            for idx, img_key in enumerate(img_keys):
                for key in self.keys:
                    if key in img_key:
                        setattr(
                            self,
                            f"sum_metric_{key}",
                            getattr(self, f"sum_metric_{key}") + error[idx],
                        )  # noqa
                        setattr(
                            self,
                            f"num_inst_{key}",
                            getattr(self, f"num_inst_{key}") + 1,
                        )  # noqa

    def compute(self):
        val = self.sum_metric / self.num_inst
        val = val.cpu().numpy().tolist()
        if self.keys is not None:
            for key in self.keys:
                v = getattr(self, f"sum_metric_{key}")
                n = getattr(self, f"num_inst_{key}")
                v = (v / n).cpu().numpy().tolist()
                val.extend(v)
        return val


@OBJECT_REGISTRY.register
class Face3dSTD(EvalMetric):
    """Evaluate face3d stability with std."""

    def __init__(self, model_type="pp"):
        self.model_type = model_type
        if model_type == "wpp":
            name = ["roll", "pitch", "yaw"]
        elif model_type == "pp":
            name = [
                "roll",
                "pitch",
                "yaw",
                "left_x",
                "left_y",
                "left_z",
                "right_x",
                "right_y",
                "right_z",
            ]
        else:
            raise ValueError(f"Not supported model_type: {model_type}")
        super().__init__(name)

    def _init_states(self):
        self.add_state("pred_pose", default=[])
        self.add_state("pred_eye3d_left", default=[])
        self.add_state("pred_eye3d_right", default=[])

    def update(self, labels: Dict, preds: Dict):
        rvec = preds["global_pose"].reshape(-1, 3)
        rot_mat = batch_rodrigues(rvec)
        if self.model_type == "pp":
            vir2real_rotmat = labels.get(
                "vir2real_rotmat", torch.eye(3, device=rvec.device)
            )
            # convert preds rot mat from virtual camera to real camera.
            rot_mat = vir2real_rotmat @ rot_mat
            # eye 3d position
            eye3d_left = preds["eye3d_left"].reshape(-1, 3)
            eye3d_right = preds["eye3d_right"].reshape(-1, 3)
            eye3d_left = vir2real_rotmat @ eye3d_left[..., None]
            eye3d_right = vir2real_rotmat @ eye3d_right[..., None]
            self.pred_eye3d_left.append(eye3d_left)
            self.pred_eye3d_right.append(eye3d_right)
        preds_euler = rot_mat_to_euler(rot_mat, "zxy")
        if self.model_type == "pp":
            preds_euler = transfer_head_pose_axes(preds_euler)
        if self.model_type == "wpp":
            preds_euler = -preds_euler
        self.pred_pose.append(preds_euler)

    def compute(self):
        preds = torch.cat(self.pred_pose, 0)
        if len(self.pred_eye3d_left) > 0:
            pred_eye3d_left = torch.cat(self.pred_eye3d_left).reshape(-1, 3)
            preds = torch.cat([preds, pred_eye3d_left], -1)
        if len(self.pred_eye3d_right) > 0:
            pred_eye3d_right = torch.cat(self.pred_eye3d_right).reshape(-1, 3)
            preds = torch.cat([preds, pred_eye3d_right], -1)
        std = np.std(preds.cpu().numpy(), axis=0).tolist()
        return std
