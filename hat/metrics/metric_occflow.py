# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from typing import Dict

import torch
import torch.nn.functional as F
import torchmetrics
import torchmetrics.functional as MF
import yaml
from easydict import EasyDict as edict

from hat.data.datasets.occflow_dataset import convert_to_waypoints
from hat.metrics.metric import EvalMetric
from hat.registry import OBJECT_REGISTRY
from hat.utils.tensor_func import divide_no_nan, tensor_mean

__all__ = ["OccFlowMetrics"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class OccFlowMetrics(EvalMetric):
    """Metric for occflow prediction task.

    There are 7 metrics included: auc for observed and occluded occupancy,
    soft-iou for observed and occluded occupancuy, flow end point error,
    auc for flow-grounded occupancy(main metric) and soft-iou for
    flow-grounded occupancy. Please refer to
    https://horizonrobotics.feishu.cn/file/boxcnvc2zh3rLIaPVMD6ZfIXzdg for
    metric details.

    Args:
        name(str): Name of this metric instance for display, also used as
            monitor params for Checkpoint.
        quantize(bool): Whether to quantize occupancy and flow prediction.
        cls_index(list(int)): Class index for metrics computation.
    """

    def __init__(
        self,
        name: str = "OccFlowMetrics",
        quantize: bool = False,
        cls_index: tuple = (0,),
    ):

        super(OccFlowMetrics, self).__init__(name=name)
        self.name = name
        self.quantize = quantize
        self.cls_index = cls_index
        self.pr_curve = torchmetrics.BinnedPrecisionRecallCurve(
            num_classes=1, thresholds=100
        )
        config_text = """
        num_past_steps: 10
        num_future_steps: 80
        num_waypoints: 8
        cumulative_waypoints: false
        normalize_sdc_yaw: true
        grid_height_cells: 256
        grid_width_cells: 256
        sdc_y_in_grid: 192
        sdc_x_in_grid: 128
        pixels_per_meter: 3.2
        agent_points_per_side_length: 48
        agent_points_per_side_width: 16
        """
        self.config = edict(yaml.safe_load(config_text))

    def _init_states(self):
        self.add_state(
            "vehicles_observed_auc",
            default=torch.zeros(1),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "vehicles_occluded_auc",
            default=torch.zeros(1),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "vehicles_observed_iou",
            default=torch.zeros(1),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "vehicles_occluded_iou",
            default=torch.zeros(1),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "vehicles_flow_epe", default=torch.zeros(1), dist_reduce_fx="sum"
        )
        self.add_state(
            "vehicles_flow_warped_occupancy_auc",
            default=torch.zeros(1),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "vehicles_flow_warped_occupancy_iou",
            default=torch.zeros(1),
            dist_reduce_fx="sum",
        )
        self.add_state("total", default=torch.zeros(1), dist_reduce_fx="sum")

    def update(
        self,
        predictions: Dict[str, torch.Tensor],
        ground_truth: Dict[str, torch.Tensor],
    ):
        """
        Update internal buffer with latest predictions.

        Args:
            predictions: model output.
            ground_truth: gt.

        """
        # N, C, H, W to N, H, W, C
        pred_waypoints, true_waypoints = self.convert_to_waypoints(
            predictions, ground_truth
        )
        # evaluate
        # Warp flow-origin occupancies according to predicted flow fields.
        warped_flow_origins = self._flow_warp(
            true_waypoints=true_waypoints,
            pred_waypoints=pred_waypoints,
        )

        # Flow for waypoint 0 can be constructed if there are any occupanciest
        # at waypoint 0.  -1 maps to the current time.
        has_true_observed_occupancy = {-1: True}
        has_true_occluded_occupancy = {-1: True}

        metrics_dict = {
            "vehicles_observed_auc": [],
            "vehicles_occluded_auc": [],
            "vehicles_observed_iou": [],
            "vehicles_occluded_iou": [],
            "vehicles_flow_epe": [],
            "vehicles_flow_warped_occupancy_auc": [],
            "vehicles_flow_warped_occupancy_iou": [],
        }

        for k in range(self.config.num_waypoints):
            true_observed_occupancy = (
                true_waypoints.vehicles.observed_occupancy[k]
            )
            pred_observed_occupancy = (
                pred_waypoints.vehicles.observed_occupancy[k]
            )
            true_occluded_occupancy = (
                true_waypoints.vehicles.occluded_occupancy[k]
            )
            pred_occluded_occupancy = (
                pred_waypoints.vehicles.occluded_occupancy[k]
            )
            true_flow = true_waypoints.vehicles.flow[k]
            pred_flow = pred_waypoints.vehicles.flow[k]

            has_true_observed_occupancy[k] = (
                torch.max(true_observed_occupancy) > 0
            )
            has_true_occluded_occupancy[k] = (
                torch.max(true_occluded_occupancy) > 0
            )
            has_true_flow = (
                has_true_observed_occupancy[k]
                and has_true_observed_occupancy[k - 1]
            ) or (
                has_true_occluded_occupancy[k]
                and has_true_occluded_occupancy[k - 1]
            )

            if has_true_observed_occupancy[k]:
                metrics_dict["vehicles_observed_auc"].append(
                    self._compute_occupancy_auc(
                        true_observed_occupancy, pred_observed_occupancy
                    )
                )
                metrics_dict["vehicles_observed_iou"].append(
                    self._compute_occupancy_soft_iou(
                        true_observed_occupancy, pred_observed_occupancy
                    )
                )
            if has_true_occluded_occupancy[k]:
                metrics_dict["vehicles_occluded_auc"].append(
                    self._compute_occupancy_auc(
                        true_occluded_occupancy, pred_occluded_occupancy
                    )
                )
                metrics_dict["vehicles_occluded_iou"].append(
                    self._compute_occupancy_soft_iou(
                        true_occluded_occupancy, pred_occluded_occupancy
                    )
                )

            if has_true_flow:
                metrics_dict["vehicles_flow_epe"].append(
                    self._compute_flow_epe(true_flow, pred_flow)
                )

                # Compute flow-warped occupancy metrics.
                # First, construct ground-truth occupancy of all observed and
                # occluded vehicles.
                true_all_occupancy = torch.clamp(
                    true_observed_occupancy + true_occluded_occupancy, 0, 1
                )
                # Construct predicted version of same value.
                pred_all_occupancy = torch.clamp(
                    pred_observed_occupancy + pred_occluded_occupancy, 0, 1
                )
                # Expect to see the same results by warping the flow-origin
                # occupancies.
                flow_warped_origin_occupancy = warped_flow_origins[k]
                # Construct quantity that requires both prediction paths to be
                # correct.
                flow_grounded_pred_all_occupancy = (
                    pred_all_occupancy * flow_warped_origin_occupancy
                )
                # Now compute occupancy metrics between this quantity and
                # ground-truth.
                metrics_dict["vehicles_flow_warped_occupancy_auc"].append(
                    self._compute_occupancy_auc(
                        true_all_occupancy, flow_grounded_pred_all_occupancy
                    )
                )
                metrics_dict["vehicles_flow_warped_occupancy_iou"].append(
                    self._compute_occupancy_soft_iou(
                        true_all_occupancy, flow_grounded_pred_all_occupancy
                    )
                )

        # Compute means and return as proto message.
        if len(metrics_dict["vehicles_observed_auc"]) > 0:
            self.vehicles_observed_auc += tensor_mean(
                metrics_dict["vehicles_observed_auc"]
            )
        if len(metrics_dict["vehicles_occluded_auc"]) > 0:
            self.vehicles_occluded_auc += tensor_mean(
                metrics_dict["vehicles_occluded_auc"]
            )
        if len(metrics_dict["vehicles_observed_iou"]) > 0:
            self.vehicles_observed_iou += tensor_mean(
                metrics_dict["vehicles_observed_iou"]
            )
        if len(metrics_dict["vehicles_occluded_iou"]) > 0:
            self.vehicles_occluded_iou += tensor_mean(
                metrics_dict["vehicles_occluded_iou"]
            )
        if len(metrics_dict["vehicles_flow_epe"]) > 0:
            self.vehicles_flow_epe += tensor_mean(
                metrics_dict["vehicles_flow_epe"]
            )
        if len(metrics_dict["vehicles_flow_warped_occupancy_auc"]) > 0:
            self.vehicles_flow_warped_occupancy_auc += tensor_mean(
                metrics_dict["vehicles_flow_warped_occupancy_auc"]
            )
        if len(metrics_dict["vehicles_flow_warped_occupancy_iou"]) > 0:
            self.vehicles_flow_warped_occupancy_iou += tensor_mean(
                metrics_dict["vehicles_flow_warped_occupancy_iou"]
            )
        self.total += 1

    def compute(self):
        metrics_dict = {
            "vehicles_observed_auc": round(
                (self.vehicles_observed_auc / self.total).item(), 3
            ),
            "vehicles_occluded_auc": round(
                (self.vehicles_occluded_auc / self.total).item(), 3
            ),
            "vehicles_observed_iou": round(
                (self.vehicles_observed_iou / self.total).item(), 3
            ),
            "vehicles_occluded_iou": round(
                (self.vehicles_occluded_iou / self.total).item(), 3
            ),
            "vehicles_flow_epe": round(
                (self.vehicles_flow_epe / self.total).item(), 3
            ),
            "vehicles_flow_warped_occupancy_auc": round(
                (self.vehicles_flow_warped_occupancy_auc / self.total).item(),
                3,
            ),
            "vehicles_flow_warped_occupancy_iou": round(
                (self.vehicles_flow_warped_occupancy_iou / self.total).item(),
                3,
            ),
        }
        summary_str = "~~~~~~~ Vehicle metrics ~~~~~~~\n"
        for k, v in metrics_dict.items():
            summary_str += "{:s}: {:.3f}\n".format(k, v)
        logger.info(summary_str)
        return metrics_dict

    def _compute_occupancy_auc(
        self,
        true_occupancy: torch.Tensor,
        pred_occupancy: torch.Tensor,
    ):
        """Compute the AUC between the predicted and true occupancy grids.

        Args:
            true_occupancy: float32 [batch_size, height, width, 1] tensor
                in [0, 1].
            pred_occupancy: float32 [batch_size, height, width, 1] tensor
                in [0, 1].
        Returns:
            AUC: float32 scalar.
        """
        true_occupancy = torch.flatten(true_occupancy)
        pred_occupancy = torch.flatten(pred_occupancy)
        self.pr_curve.reset()
        precision, recall, _ = self.pr_curve(pred_occupancy, true_occupancy)
        auc_result = MF.auc(recall, precision, reorder=True)
        return auc_result

    def _compute_occupancy_soft_iou(
        self,
        true_occupancy: torch.Tensor,
        pred_occupancy: torch.Tensor,
    ):
        """Compute the soft IoU between predicted and true occupancy grids.

        Args:
            true_occupancy: float32 [batch_size, height, width, 1] tensor
                in [0, 1].
            pred_occupancy: float32 [batch_size, height, width, 1] tensor
                in [0, 1].
        Returns:
            Soft IoU score: float32 scalar.
        """
        true_occupancy = torch.flatten(true_occupancy)
        pred_occupancy = torch.flatten(pred_occupancy)

        intersection = torch.mean(pred_occupancy * true_occupancy)
        true_sum = torch.mean(true_occupancy)
        pred_sum = torch.mean(pred_occupancy)
        # Scenes with empty ground-truth will have a score of 0.
        return divide_no_nan(intersection, pred_sum + true_sum - intersection)

    def _compute_flow_epe(
        self,
        true_flow: torch.Tensor,
        pred_flow: torch.Tensor,
    ):
        """Compute average end-point-error.

        Compute average end-point-error between predicted and true flow
        fields. Flow end-point-error measures the Euclidean distance between
        the predicted and ground-truth flow vector endpoints.

        Args:
            true_flow: float32 Tensor shaped [batch_size, height, width, 2].
            pred_flow: float32 Tensor shaped [batch_size, height, width, 2].
        Returns:
            EPE averaged over all grid cells: float32 scalar.
        """
        # [batch_size, height, width, 2]
        diff = true_flow - pred_flow
        # [batch_size, height, width, 1]
        flow_exists = (
            torch.any(true_flow != 0, dim=-1).to(torch.float32).unsqueeze(-1)
        )
        diff = diff * flow_exists
        # [batch_size, height, width, 1]
        epe = torch.linalg.norm(diff, ord=2, dim=-1, keepdims=True)
        # Scalar.
        sum_epe = torch.sum(epe)
        # Scalar.
        sum_flow_exists = torch.sum(flow_exists)
        # Scalar.
        return divide_no_nan(sum_epe, sum_flow_exists)

    def _flow_warp(
        self,
        true_waypoints,
        pred_waypoints,
    ):
        """Warp the gt occupancy origin with flow prediction.

        Args:
            true_waypoints: ground-truth waypoints.
            pred_flow: prediction waypoints.
        Returns:
            Warped occupancy.
        """
        h = torch.arange(self.config.grid_height_cells, dtype=torch.float32)
        w = torch.arange(self.config.grid_width_cells, dtype=torch.float32)
        h_idx, w_idx = torch.meshgrid(h, w)
        # These indices map each (x, y) location to (x, y).
        # [height, width, 2] but storing x, y coordinates.
        identity_indices = torch.stack(
            (
                torch.t(h_idx),
                torch.t(w_idx),
            ),
            dim=-1,
        )

        warped_flow_origins = []
        for k in range(self.config.num_waypoints):
            # [batch_size, height, width, 1]
            flow_origin_occupancy = (
                true_waypoints.vehicles.flow_origin_occupancy[k]
            )
            # [batch_size, height, width, 2]
            pred_flow = pred_waypoints.vehicles.flow[k]
            # Shifting the identity grid indices according to predicted flow
            # tells us # the source (origin) grid cell for each flow vector.
            # We simply sample # occupancy values from these locations.
            # [batch_size, height, width, 2]
            identity_indices = identity_indices.to(pred_flow.device)
            warped_indices = identity_indices + pred_flow
            # convert flow_origin from [N, H, W, C] to [N, C, H, W]
            flow_origin_occupancy = flow_origin_occupancy.permute([0, 3, 1, 2])
            # Normalize flow grid
            warped_indices[..., 0] = (
                warped_indices[..., 0] / (self.config.grid_width_cells - 1) * 2
                - 1
            )
            warped_indices[..., 1] = (
                warped_indices[..., 1]
                / (self.config.grid_height_cells - 1)
                * 2
                - 1
            )
            warped_origin = F.grid_sample(
                input=flow_origin_occupancy,
                grid=warped_indices,
                align_corners=True,
            )
            warped_flow_origins.append(warped_origin.permute(0, 2, 3, 1))

        return warped_flow_origins

    def convert_to_waypoints(
        self,
        predictions: Dict[str, torch.Tensor],
        ground_truth: Dict[str, torch.Tensor],
    ):
        """Convert prediction and gt dict to waypoints format.

        Args:
            predictions: model output.
            ground_truth: gt.
        """
        # N, C, H, W to N, H, W, C
        occupancy_preds = (
            predictions["occ_preds"].sigmoid().permute([0, 2, 3, 1])
        )
        flow_preds = predictions["flow_preds"].permute([0, 2, 3, 1])
        if self.quantize:
            occupancy_preds = torch.round(occupancy_preds * 255) / 255
            flow_preds = torch.clamp(torch.round(flow_preds), -128, 127)
        occupancy_target = ground_truth["occupancy_waypoints"][
            :, :, :, :, self.cls_index, :
        ].flatten(3, 5)
        flow_target = ground_truth["flow_waypoints"][
            :, :, :, :, self.cls_index, :
        ].flatten(3, 5)
        flow_origin_occupancy_target = ground_truth[
            "flow_origin_occupancy_waypoints"
        ][:, :, :, :, self.cls_index, :].flatten(3, 5)

        pred_waypoints = convert_to_waypoints(occupancy_preds, flow_preds)
        target_waypoints = convert_to_waypoints(
            occupancy_target, flow_target, flow_origin_occupancy_target
        )
        return pred_waypoints, target_waypoints
