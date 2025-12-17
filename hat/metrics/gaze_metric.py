# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from typing import List, Union

import numpy as np
import torch
from prettytable import PrettyTable

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list
from .metric import EvalMetric

logger = logging.getLogger(__name__)

__all__ = ["AngleDifferenceMetric", "EyeLdmksDist", "GazeModelTestMetric"]


R2A = 180 / np.pi
A2R = np.pi / 180


@OBJECT_REGISTRY.register
class AngleDifferenceMetric(EvalMetric):
    """Compute Average Angle Difference with possible missing values.

    Args:
        name: Name of this metric instance for display.
        use_glass: Use in two head branch(glass、no glass).
        thresh: Gaze that exceeds the specified thred is ignored
            when calculating the metric.
    """

    def __init__(
        self,
        name="angle_diff",
        use_glass: bool = False,
        thresh: int = 90,
    ):
        super(AngleDifferenceMetric, self).__init__(name)
        self.eval_type = name
        self.use_glass = use_glass
        self.thresh = thresh

    def _compute_angle_error(self, label, pred):
        """Compute angle errors of gaze prediction(degree).

        Args:
            label: ground truth of gaze(pitch, yaw)
            pred: prediction of gaze(pitch, yaw)
        """
        pred_x, pred_y, pred_z = self._convert_to_unit_vector(pred * A2R)
        label_x, label_y, label_z = self._convert_to_unit_vector(label * A2R)
        angles = pred_x * label_x + pred_y * label_y + pred_z * label_z
        angles = torch.arccos(torch.clip(angles, min=-1.0, max=1.0)) * R2A
        return angles

    def _convert_to_unit_vector(self, angles):
        x = torch.cos(angles[:, 0]) * torch.sin(angles[:, 1])
        y = torch.sin(angles[:, 0])
        z = -(torch.cos(angles[:, 0]) * torch.cos(angles[:, 1]))
        return x, y, z

    def preprocess(self, gt_gaze, pred_gaze):
        pred_gaze = pred_gaze.squeeze()
        if self.eval_type == "left_eye":
            val_pred = pred_gaze[:, :2]
            val_gt = gt_gaze[:, :2]
        elif self.eval_type == "right_eye":
            val_pred = pred_gaze[:, 2:]
            val_gt = gt_gaze[:, 2:]
        return val_gt, val_pred

    def update(self, labels, preds):
        labels_ = labels.copy()
        preds_ = preds.copy()
        if self.use_glass:
            gt_gazes = _as_list(labels_["gt_gaze"])
            gt_glasses = _as_list(labels_["gt_glass"])
            pred_gazes = _as_list(preds_["gaze"])
            pred_glass_gazes = _as_list(preds_["glass_gaze"])
            for (gt_gaze, gt_glass, pred_gaze, pred_glass_gaze,) in zip(
                gt_gazes,
                gt_glasses,
                pred_gazes,
                pred_glass_gazes,
            ):
                # calc error for non-glass cases
                _gaze_label, _gaze_wo_glass_pred = self.preprocess(
                    gt_gaze, pred_gaze
                )
                gaze_thresh_mask = _gaze_label.abs() < self.thresh
                _mask = (_gaze_label > -1000) * torch.bitwise_and(
                    gaze_thresh_mask[:, 0], gaze_thresh_mask[:, 1]
                ).view(-1, 1)
                _gaze_label = _gaze_label * _mask
                _gaze_wo_glass_pred = _gaze_wo_glass_pred * _mask
                _diff_wo_glass = self._compute_angle_error(
                    _gaze_label, _gaze_wo_glass_pred
                )
                _diff_wo_glass = _diff_wo_glass * (1 - gt_glass.squeeze())
                _diff_wo_glass_sum = _diff_wo_glass.sum()

                # calc error for glass cases
                _, _gaze_wt_glass_pred = self.preprocess(
                    gt_gaze, pred_glass_gaze
                )
                _gaze_wt_glass_pred = _gaze_wt_glass_pred * _mask
                _diff_wt_glass = self._compute_angle_error(
                    _gaze_label, _gaze_wt_glass_pred
                )
                _diff_wt_glass = _diff_wt_glass * gt_glass.squeeze()
                _diff_wt_glass_sum = _diff_wt_glass.sum()

                num_inst = _mask.int().min(axis=-1)[0].sum()
                self.sum_metric += _diff_wo_glass_sum + _diff_wt_glass_sum
                self.num_inst += num_inst
        else:
            gt_gazes = _as_list(labels_["gt_gaze"])
            pred_gazes = _as_list(preds_["gaze"])
            for gt_gaze, pred_gaze in zip(gt_gazes, pred_gazes):
                pred_gaze = pred_gaze.squeeze()
                assert (
                    pred_gaze.shape == gt_gaze.shape
                ), "pred_label shape != label shape"
                gt_gaze, pred_gaze = self.preprocess(gt_gaze, pred_gaze)
                gaze_thresh_mask = gt_gaze.abs() < self.thresh
                mask = (gt_gaze > -1000) * torch.bitwise_and(
                    gaze_thresh_mask[:, 0], gaze_thresh_mask[:, 1]
                ).view(-1, 1)
                # mask = gt_gaze > -1000
                gt_gaze = gt_gaze * mask
                pred_gaze = pred_gaze * mask
                diff = self._compute_angle_error(gt_gaze, pred_gaze)
                total = diff.sum()
                num_inst = (diff > 0).sum()
                self.sum_metric += total
                self.num_inst += num_inst


@OBJECT_REGISTRY.register
class EyeLdmksDist(EvalMetric):
    """Customized metric for eye ldmks.

    Args:
        name: Name of this metric instance for display.
        size: Input size of data.
    """

    def __init__(self, name: str, size: tuple = (320, 192)):
        super(EyeLdmksDist, self).__init__(name)
        self._size = size

    def preprocess(self, gt_eye_ldmks, pred_eye_ldmks):
        val_pred = pred_eye_ldmks
        val_gt = gt_eye_ldmks
        return val_gt, val_pred

    def update(self, labels, preds):
        labels = _as_list(labels)
        preds = _as_list(
            torch.concat(preds["eye_ldmk"], dim=1).reshape(-1, 42, 2)
        )
        for label, pred in zip(labels, preds):
            label, pred = self.preprocess(label, pred)
            mask = label > -1000
            label = label * mask
            pred = pred * mask
            diff = pred - label
            diff[:, :, 0] *= self._size[0]
            diff[:, :, 1] *= self._size[1]
            diff = torch.norm(diff, p=2, dim=-1)
            num_diff = (diff > 0).sum()
            self.sum_metric += diff.sum()
            self.num_inst += num_diff


COND_TYPE = [
    "YAW_30",
    "YAW_50",
    "YAW_75",
    "ALL",
    "YAW_30_Ex",
    "YAW_50_Ex",
    "YAW_75_Ex",
    "ALL_Ex",
]


@OBJECT_REGISTRY.register
class GazeModelTestMetric(AngleDifferenceMetric):
    """Compute Average Angle Difference with possible missing values.

    Only use in test model.
    Args:
        name: Name of this metric instance for display.
    """

    def __init__(
        self,
        name: Union[List[str], str] = None,
        use_glass: bool = False,
        save_txt: str = None,
    ):
        super(GazeModelTestMetric, self).__init__(name, use_glass)
        if save_txt is not None:
            self.fp = open(save_txt, "a")
        else:
            self.fp = None

    def _init_states(self):
        for cond in COND_TYPE:
            for dire in ["left", "right"]:
                for error_type in ["pitch", "yaw", "angle"]:
                    self.add_state(
                        "{}_{}_error_{}".format(dire, error_type, cond), []
                    )

    def reset(self):
        # TODO
        pass

    def _compute_cond_mask(self, gt_gaze, cond_type):
        gt_gaze = torch.tensor(gt_gaze)
        if cond_type == "YAW_30":
            mask = torch.where(
                (abs(gt_gaze[:, 1]) <= 30) & (abs(gt_gaze[:, 3]) <= 30)
            )
        elif cond_type == "YAW_50":
            mask = torch.where(
                (abs(gt_gaze[:, 1]) <= 50) & (abs(gt_gaze[:, 3]) <= 50)
            )
        elif cond_type == "YAW_75":
            mask = torch.where(
                (abs(gt_gaze[:, 1]) <= 75) & (abs(gt_gaze[:, 3]) <= 75)
            )
        elif cond_type == "ALL":
            mask = torch.where(
                (abs(gt_gaze[:, 1]) <= 180) & (abs(gt_gaze[:, 3]) <= 180)
            )
        elif cond_type == "YAW_30_Ex":
            mask = torch.where(
                (abs(gt_gaze[:, 1]) <= 30) & (abs(gt_gaze[:, 3]) <= 30)
            )
        elif cond_type == "YAW_50_Ex":
            mask = torch.where(
                (abs(gt_gaze[:, 1]) <= 50)
                & (abs(gt_gaze[:, 3]) <= 50)
                & (abs(gt_gaze[:, 1]) > 30)
                & (abs(gt_gaze[:, 3]) > 30)
            )
        elif cond_type == "YAW_75_Ex":
            mask = torch.where(
                (abs(gt_gaze[:, 1]) <= 75)
                & (abs(gt_gaze[:, 3]) <= 75)
                & (abs(gt_gaze[:, 1]) > 50)
                & (abs(gt_gaze[:, 3]) > 50)
            )
        elif cond_type == "ALL_Ex":
            mask = torch.where(
                (abs(gt_gaze[:, 1]) <= 180)
                & (abs(gt_gaze[:, 3]) <= 180)
                & (abs(gt_gaze[:, 1]) > 75)
                & (abs(gt_gaze[:, 3]) > 75)
            )
        return mask[0]

    def preprocess(self, gt_gaze, pred_gaze):
        pred_gaze = pred_gaze.squeeze()
        left_gt, left_pred, right_gt, right_pred = (
            gt_gaze[:, :2],
            pred_gaze[:, :2],
            gt_gaze[:, 2:],
            pred_gaze[:, 2:],
        )

        for cond in COND_TYPE:
            cond_mask = self._compute_cond_mask(gt_gaze, cond)
            for dire in ["left", "right"]:
                for type_error in ["pitch", "yaw", "angle"]:
                    error = self._compute_error(
                        left_gt,
                        left_pred,
                        right_gt,
                        right_pred,
                        dire,
                        type_error,
                        cond_mask,
                    )
                    if error is not None:
                        getattr(
                            self,
                            "{}_{}_error_{}".format(dire, type_error, cond),
                        ).extend(error.cpu().numpy().tolist())

    def update(self, labels, preds):
        labels = _as_list(labels["gt_gaze"])
        if self.use_glass:
            gaze_no_glass = preds["gaze"].squeeze()
            gaze_glass = preds["glass_gaze"].squeeze()
            glass_cls = torch.argmax(preds["glass_cls"].squeeze(), dim=1)
            pred_gaze = gaze_glass * glass_cls.reshape(
                -1, 1
            ) + gaze_no_glass * (1 - glass_cls).reshape(-1, 1)
            pred_gaze = _as_list(pred_gaze)
        else:
            pred_gaze = _as_list(preds["gaze"])
        for label, gaze in zip(labels, pred_gaze):
            gaze = gaze.squeeze()
            assert gaze.shape == label.shape, "pred_label shape != label shape"
            self.preprocess(label, gaze)

    def _compute_error(
        self,
        left_gt,
        left_pred,
        right_gt,
        right_pred,
        dire,
        type_error,
        cond_mask,
    ):
        if dire == "left":
            gt, pred = left_gt, left_pred
        else:
            gt, pred = right_gt, right_pred
        # TODO
        if cond_mask.shape[0] == 0:
            return None
        gt = gt[cond_mask]
        pred = pred[cond_mask]
        if type_error == "pitch":
            gt, pred = gt[:, 0], pred[:, 0]
            return torch.abs(gt - pred)
        elif type_error == "yaw":
            gt, pred = gt[:, 1], pred[:, 1]
            return torch.abs(gt - pred)
        return self._compute_angle_error(gt, pred)

    def compute(
        self,
    ):
        # Yaw_30、Yaw_50、Yaw_75、ALL
        # Yaw_30_Ex、Yaw_50_Ex、Yaw_75_Ex、ALL_Ex
        # pitch、yaw、angle、P90
        for type_error in ["pitch", "yaw", "angle", "P90"]:
            for cond in COND_TYPE:
                if type_error == "P90":
                    left_error = getattr(
                        self, "left_{}_error_{}".format("angle", cond)
                    )
                    right_error = getattr(
                        self, "right_{}_error_{}".format("angle", cond)
                    )
                    num_inst = len(left_error)
                    setattr(
                        self,
                        "{}_{}".format(type_error, cond),
                        round(
                            0.5
                            * (
                                sorted(left_error)[int(num_inst * 0.9)]
                                + sorted(right_error)[int(num_inst * 0.9)]
                            )
                            if num_inst
                            else -1000,
                            2,
                        ),
                    )
                else:
                    left_error = getattr(
                        self, "left_{}_error_{}".format(type_error, cond)
                    )
                    right_error = getattr(
                        self, "right_{}_error_{}".format(type_error, cond)
                    )
                    setattr(
                        self,
                        "{}_{}".format(type_error, cond),
                        round(
                            (np.mean(left_error) + np.mean(right_error)) / 2, 2
                        ),
                    )

    def get(self):
        self.compute()
        table = PrettyTable()
        table.field_names = ["YAW_30", "YAW_50", "YAW_75", "ALL"]
        table.add_row(
            [
                "{}/{}/{}/{}".format(
                    getattr(self, f"{'pitch'}_YAW_30"),
                    getattr(self, f"{'yaw'}_YAW_30"),
                    getattr(self, f"{'angle'}_YAW_30"),
                    getattr(self, f"{'P90'}_YAW_30"),
                ),
                "{}/{}/{}/{}".format(
                    getattr(self, f"{'pitch'}_YAW_50"),
                    getattr(self, f"{'yaw'}_YAW_50"),
                    getattr(self, f"{'angle'}_YAW_50"),
                    getattr(self, f"{'P90'}_YAW_50"),
                ),
                "{}/{}/{}/{}".format(
                    getattr(self, f"{'pitch'}_YAW_75"),
                    getattr(self, f"{'yaw'}_YAW_75"),
                    getattr(self, f"{'angle'}_YAW_75"),
                    getattr(self, f"{'P90'}_YAW_75"),
                ),
                "{}/{}/{}/{}".format(
                    getattr(self, f"{'pitch'}_ALL"),
                    getattr(self, f"{'yaw'}_ALL"),
                    getattr(self, f"{'angle'}_ALL"),
                    getattr(self, f"{'P90'}_ALL"),
                ),
            ]
        )
        table_ex = PrettyTable()
        table_ex.field_names = [
            "YAW_30_Ex",
            "YAW_50_Ex",
            "YAW_75_Ex",
            "ALL_Ex",
        ]
        table_ex.add_row(
            [
                "{}/{}/{}/{}".format(
                    getattr(self, f"{'pitch'}_YAW_30_Ex"),
                    getattr(self, f"{'yaw'}_YAW_30_Ex"),
                    getattr(self, f"{'angle'}_YAW_30_Ex"),
                    getattr(self, f"{'P90'}_YAW_30_Ex"),
                ),
                "{}/{}/{}/{}".format(
                    getattr(self, f"{'pitch'}_YAW_50_Ex"),
                    getattr(self, f"{'yaw'}_YAW_50_Ex"),
                    getattr(self, f"{'angle'}_YAW_50_Ex"),
                    getattr(self, f"{'P90'}_YAW_50_Ex"),
                ),
                "{}/{}/{}/{}".format(
                    getattr(self, f"{'pitch'}_YAW_75_Ex"),
                    getattr(self, f"{'yaw'}_YAW_75_Ex"),
                    getattr(self, f"{'angle'}_YAW_75_Ex"),
                    getattr(self, f"{'P90'}_YAW_75_Ex"),
                ),
                "{}/{}/{}/{}".format(
                    getattr(self, f"{'pitch'}_ALL_Ex"),
                    getattr(self, f"{'yaw'}_ALL_Ex"),
                    getattr(self, f"{'angle'}_ALL_Ex"),
                    getattr(self, f"{'P90'}_ALL_Ex"),
                ),
            ]
        )
        logger.info(table)
        logger.info(table_ex)
        # save evaluate result
        if self.fp is not None:
            self.fp.write(str(table) + "\n")
            self.fp.write(str(table_ex) + "\n")
        return [], []
