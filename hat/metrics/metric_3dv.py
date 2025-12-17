# Copyright (c) Horizon Robotics. All rights reserved.

import logging
import os
from typing import Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list
from hat.utils.logger import rank_zero_info
from hat.utils.package_helper import require_packages
from .metric import EvalMetric

__all__ = ["AbsRel", "PoseRTE", "RMSE", "ConfRMSE"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class AbsRel(EvalMetric):
    """calculation multi range absrel for depth task.

    math.abs(pred-gt)/gt.

    Args:
        range_list (list) : A list contains multi range to calculation absrel.
        name (str): metric name.
    """

    def __init__(
        self, range_list=((0, 150)), name="absrel", inter_model="nearest"
    ):
        names = []
        for (low, high) in range_list:
            assert low < high, "range setting not valid"
            names.append("(%s,%s)" % (str(low), str(high)))
        self.range_list = range_list
        self.summary_name = name
        super(AbsRel, self).__init__(names)

    def _init_states(self):
        self.add_state(
            "sum_metric",
            torch.zeros((len(self.range_list),)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "num_inst",
            torch.zeros((len(self.range_list),)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "sum_std",
            torch.zeros((len(self.range_list),)),
            dist_reduce_fx="sum",
        )

    def get_absrel(self, pred, gt, low, high):
        valid_mask = (gt > low) * (gt < high)
        mask_zeros = (gt == 0) * (1e-6)
        diff = (pred - gt).abs()
        abs_rel = diff / (gt + mask_zeros)  # Not divide zero.
        abs_rel_valid = abs_rel * valid_mask
        abs_rel_valid_mean = abs_rel_valid.sum() / valid_mask.sum()
        return abs_rel_valid_mean

    def update(
        self,
        gt_depth: torch.Tensor,
        pred_depth: Union[Sequence[torch.Tensor], torch.Tensor],
    ):

        pred = _as_list(pred_depth)[0][:, 0:1]

        pred = F.interpolate(
            pred, [gt_depth.size(2), gt_depth.size(3)], mode="nearest"
        )

        for i, (low, high) in enumerate(self.range_list):
            abs_rel = self.get_absrel(pred, gt_depth, low, high)
            if not torch.any(torch.isnan(abs_rel)):
                self.sum_metric[i] += abs_rel
                self.sum_std[i] += abs_rel ** 2
                self.num_inst[i] += 1

    def compute(self):
        val_mean = self.sum_metric / self.num_inst
        val_std = torch.sqrt(self.sum_std / self.num_inst - val_mean ** 2)
        summary_str = f"~~~~ {self.summary_name} Summary metrics ~~~~\n"
        line_format = "{:<15} {:>9} {:>9}\n"
        summary_str += line_format.format("Range", "Mean", "Std")
        for i in range(len(self.name)):
            line_format = "{:<15}{:>10}{:>10}\n"
            summary_str += line_format.format(
                self.name[i],
                "{:.3f}".format(val_mean[i]),
                "{:.3f}".format(val_std[i]),
            )
        logger.info(summary_str)
        return val_mean.cpu().numpy().tolist(), val_std.cpu().numpy().tolist()


@OBJECT_REGISTRY.register
class RMSE(EvalMetric):
    """Calculation multi range rmse for depth task.

    rmse = math.sqrt(math.mean((gt - pred) ** 2)).

    Args:
        range_list: A sequence contains multi range to calculation absrel.
    """

    def __init__(
        self, range_list: Tuple[Tuple] = ((0, 150),), name: str = "RMSE"
    ):
        self.summary_name = name
        names = []
        range_list = _as_list(range_list)
        for (low, high) in range_list:
            assert low < high, "range setting not valid"
            names.append("(%s,%s] " % (str(low), str(high)))
        self.range_list = range_list
        super(RMSE, self).__init__(names)

    def _init_states(self):
        self.add_state(
            "sum_metric",
            torch.zeros((len(self.range_list),)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "num_inst",
            torch.zeros((len(self.range_list),)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "sum_std",
            torch.zeros((len(self.range_list),)),
            dist_reduce_fx="sum",
        )

    def get_rmse(self, pred, gt, low, high):
        valid_mask = (gt > low) * (gt <= high)
        rmse = (pred - gt) ** 2
        rmse_valid = rmse * valid_mask
        rmse_mean = rmse_valid.sum() / valid_mask.sum()
        return rmse_mean

    def update(
        self,
        gt_depth: torch.Tensor,
        pred_depth: Union[Sequence[torch.Tensor], torch.Tensor],
    ):

        pred = _as_list(pred_depth)[0]
        if pred.shape[2:] != gt_depth.shape[2:]:
            pred = F.interpolate(
                pred, size=gt_depth.size()[-2:], mode="bilinear"
            )

        for i, depth_range in enumerate(self.range_list):
            low, high = depth_range
            rmse = self.get_rmse(pred, gt_depth, low, high)
            if not torch.any(torch.isnan(rmse)):
                self.sum_metric[i] += torch.sqrt(rmse)
                self.sum_std[i] += rmse
                self.num_inst[i] += 1

    def compute(self):
        val_mean = self.sum_metric / self.num_inst
        val_std = torch.sqrt(self.sum_std / self.num_inst - val_mean ** 2)
        summary_str = f"~~~~ {self.summary_name} Summary metrics ~~~~\n"
        line_format = "{:<15} {:>9} {:>9}\n"
        summary_str += line_format.format("Range", "Mean", "Std")
        for i in range(len(self.name)):
            line_format = "{:<15}{:>10}{:>10}\n"
            summary_str += line_format.format(
                self.name[i],
                "{:.3f}".format(val_mean[i]),
                "{:.3f}".format(val_std[i]),
            )
        logger.info(summary_str)
        return val_mean, val_std


@OBJECT_REGISTRY.register
class ConfRMSE(RMSE):
    """Calculation multi range confidence rmse for depth task."""

    def get_rmse(self, pred, gt, gt_depth, low, high):
        valid_mask = (gt_depth > low) * (gt_depth <= high)
        rmse = (pred - gt) ** 2
        rmse_valid = rmse * valid_mask
        rmse_mean = rmse_valid.sum() / valid_mask.sum()
        return rmse_mean

    def update(
        self,
        gt_depth: torch.Tensor,
        gt_conf: torch.Tensor,
        pred_conf: Union[Sequence[torch.Tensor], torch.Tensor],
    ):
        pred = _as_list(pred_conf)[0]
        if pred.shape[2:] != gt_conf.shape[2:]:
            pred = F.interpolate(
                pred, size=gt_conf.size()[-2:], mode="bilinear"
            )

        for i, depth_range in enumerate(self.range_list):
            low, high = depth_range
            rmse = self.get_rmse(pred, gt_conf, gt_depth, low, high)
            if not torch.any(torch.isnan(rmse)):
                self.sum_metric[i] += torch.sqrt(rmse)
                self.sum_std[i] += rmse
                self.num_inst[i] += 1


@OBJECT_REGISTRY.register
class PoseRTE(EvalMetric):
    """RTE metric for pose.

    Args:
        gt_pose_path (str): gt pose path.
        name (str): Name of this metric instance for display.
        save_traj_path (str): path to save trajectory result,
            None means not save.
        scale_coe (float): scale coefficient for pose. default is 0.01.
            donot recommend changing it.
    """

    @require_packages("horizon_driving_dataset")
    def __init__(
        self,
        gt_pose_path,
        name="pose_rte",
        save_traj_path=None,
        scale_coe=0.01,
    ):

        self.gt_pose_path = gt_pose_path
        self.gt_pose = np.loadtxt(gt_pose_path)

        from horizon_driving_dataset import PoseEvaluator, PoseTransformer

        self.pt = PoseTransformer()
        self.pe = PoseEvaluator(alignment="7dof")

        self.save_dir = None
        self.epoch = 1
        self.scale_coe = scale_coe

        if save_traj_path is not None:
            self.save_dir = os.path.join(
                save_traj_path, self.gt_pose_path.split("/")[-2]
            )
            if not os.path.exists(self.save_dir):
                try:
                    os.makedirs(self.save_dir)
                except Exception:
                    pass

        super(PoseRTE, self).__init__(name)

    def _init_states(self):
        self.add_state("timestamp", [], dist_reduce_fx="cat")
        self.add_state("pred_trans", [], dist_reduce_fx="cat")
        self.add_state("pred_axis", [], dist_reduce_fx="cat")

    def reset(self) -> None:
        self.pt.reset()
        super().reset()

    def update(
        self,
        timestamp: Union[Sequence[torch.Tensor], torch.Tensor],
        axisangle: Union[Sequence[torch.Tensor], torch.Tensor],
        translation: Union[Sequence[torch.Tensor], torch.Tensor],
    ) -> None:

        self.timestamp.append(timestamp)
        translation = _as_list(translation)[-1]
        self.pred_trans.append(translation.view(-1, 3))
        axisangle = _as_list(axisangle)[-1]
        self.pred_axis.append(axisangle.view(-1, 3))

    def compute(self):

        trans = (self.pred_trans * self.scale_coe).cpu().numpy()
        axis = (self.pred_axis * self.scale_coe).cpu().numpy()
        timestamp = self.timestamp.cpu().numpy()

        self.pt.from_relative_axis_angle(axis)
        self.pt.from_relative_translation(trans)
        self.pt.load_timestamp(timestamp)
        self.pt.sort_by_timestamps()
        tum_array = self.pt.dumparray(style="tum")

        if self.save_dir:
            np.savetxt(
                os.path.join(self.save_dir, "pred_pose_%d.txt" % self.epoch),
                tum_array,
            )
            self.epoch += 1

        if os.getenv("HAT_PIPELINE_TEST", "0") == "0":
            result_dict = self.pe.eval(self.gt_pose, tum_array)
            key = [
                "scale",
                "RTE",
                "RRE",
                "EulerRoll",
                "EulerPitch",
                "EulerYaw",
                "ATE",
                "RRE_m",
                "RRE_deg",
                "ITE",
                "IRE",
                "instant_roll",
                "instant_pitch",
                "instant_yaw",
            ]
            for k in key:
                rank_zero_info(k + "  :%.2f" % result_dict[k])
        else:
            rank_zero_info("skip PoseRTE metric in pipeline_test")
            result_dict = {"RTE": 0.0}
        return (self.name, result_dict["RTE"])
