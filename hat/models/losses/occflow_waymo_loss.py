# Copyright (c) Horizon Robotics. All rights reserved.
from collections import OrderedDict
from typing import Dict, List, Union

import torch
import torch.nn.functional as F

from hat.registry import OBJECT_REGISTRY

__all__ = [
    "BaseOccflowLossWaymo",
    "FocalLossWaymo",
    "FlowLoss",
    "FlowTracedLoss",
    "ProbabilisticLossWaymo",
]


@OBJECT_REGISTRY.register
class BaseOccflowLossWaymo(torch.nn.Module):
    """Loss computation for occupancy and flow prediction task.

    There are 4 losses included: focal loss for occupancy, smooth l1 loss for
    flow, traced BCE loss for grounded occupancy and traced focal loss for
    flow-grounded occupancy. Please refer to
    https://horizonrobotics.feishu.cn/file/boxcnvc2zh3rLIaPVMD6ZfIXzdg for
    loss details.

    Args:
        compute_loss_cls_index: list of class index to compute loss
        loss_weights: loss weights for different losses
        flow_loss_args: arguments for flow loss
        traced_bce_loss_args: arguments for traced bce loss
        traced_focal_loss_args: arguments for traced focal loss
    """

    def __init__(
        self,
        compute_loss_cls_index: List[int],
        loss_weights: List[int] = None,
        flow_loss_args: Dict = None,
        traced_bce_loss_args: Dict = None,
        traced_focal_loss_args: Dict = None,
    ):
        super(BaseOccflowLossWaymo, self).__init__()

        self.compute_loss_cls_index = compute_loss_cls_index
        self.losses_fn = torch.nn.ModuleDict()

        # occupancy focal loss
        self.losses_fn["occupancy_loss"] = FocalLossWaymo(
            compute_loss_cls_index=self.compute_loss_cls_index,
        )

        # flow loss
        if not flow_loss_args:
            flow_loss_args = {}
        self.losses_fn["flow_loss"] = FlowLoss(
            compute_loss_cls_index=self.compute_loss_cls_index,
            **flow_loss_args
        )

        # traced bce loss
        if not traced_bce_loss_args:
            traced_bce_loss_args = {}
        self.losses_fn["traced_bce_loss"] = FlowTracedLoss(
            compute_loss_cls_index=self.compute_loss_cls_index,
            loss_type="bce",
            **traced_bce_loss_args
        )
        # traced focal loss
        if not traced_focal_loss_args:
            traced_focal_loss_args = {}
        self.losses_fn["traced_focal_loss"] = FlowTracedLoss(
            compute_loss_cls_index=self.compute_loss_cls_index,
            loss_type="focal_loss",
            **traced_focal_loss_args
        )
        if loss_weights is None:
            self.loss_weights = [1] * len(self.losses_fn)
        else:
            assert len(loss_weights) == len(self.losses_fn)
            self.loss_weights = loss_weights

        self.qconfig = None
        print("OCCFLOW LOSS INITIALIZED")

    def forward(self, pred, target):
        loss = OrderedDict()
        for loss_key in self.losses_fn.keys():
            loss[loss_key] = self.losses_fn[loss_key](pred, target)

        combined_loss = torch.tensor(0).float().to(pred["occ_preds"].device)
        for i, value in enumerate(loss.values()):
            combined_loss += self.loss_weights[i] * value
        loss["combined_loss"] = combined_loss
        return loss


@OBJECT_REGISTRY.register
class FocalLossWaymo(torch.nn.Module):
    """Focal loss for observed and occluded occupancy."""

    def __init__(
        self,
        compute_loss_cls_index: List[int],
        n_future: int = 8,
    ):
        super(FocalLossWaymo, self).__init__()
        self.compute_loss_cls_index = compute_loss_cls_index
        self.n_future = n_future

    def forward(
        self, predictions: Dict, ground_truth: Union[Dict, torch.Tensor]
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            predictions: dict with occupancy prediction with shape
                [N, n_class*2*n_future, H, W]
            ground_truth: ground_truth dict that has 'occupancy_waypoints'
                with shape [N, H, W, 2, n_class, n_future]
        """
        preds = (
            predictions["occ_preds"]
            .permute([0, 2, 3, 1])
            .unflatten(3, [2, len(self.compute_loss_cls_index), self.n_future])
            .sigmoid()
        )
        # (batch, H, W, 2(observed/occluded), cls, num_waypoints)
        target = ground_truth["occupancy_waypoints"][
            :, :, :, :, self.compute_loss_cls_index, :
        ]
        preds = torch.clamp(preds, 0.001, 1 - 0.001)
        pos_inds = target.ge(1).float()
        neg_inds = target.lt(1).float()
        pos_loss = torch.log(preds) * torch.pow(1 - preds, 2) * pos_inds
        neg_loss = torch.log(1 - preds) * torch.pow(preds, 2) * neg_inds
        return -(pos_loss + neg_loss).mean()


@OBJECT_REGISTRY.register
class FlowLoss(torch.nn.Module):
    """Smooth L1 loss for flow prediction.

    Args:
        flow_dim: flow prediction dimension, default is 2.
        n_future: flow prediction frames, default is 8.
        enable_time_discount: whether to weight flow loss over time.
        discount_factor: the exponetial time weight factor for flow loss.
    """

    def __init__(
        self,
        compute_loss_cls_index: List[int],
        flow_dim: int = 2,
        n_future: int = 8,
        enable_time_discount: bool = False,
        discount_factor: float = 1.0,
    ):
        super(FlowLoss, self).__init__()
        self.smooth_l1_loss = torch.nn.SmoothL1Loss(reduction="none", beta=1.0)
        self.compute_loss_cls_index = compute_loss_cls_index
        self.flow_dim = flow_dim
        self.n_future = n_future

        # exponetial weight decay/ascend on time
        self.enable_time_discount = enable_time_discount
        self.discount_factor = discount_factor
        self.time_weights = [
            self.discount_factor ** (self.n_future - 1 - i)
            for i in range(self.n_future)
        ]

    def forward(
        self, predictions: Dict, ground_truth: Union[Dict, torch.Tensor]
    ) -> torch.Tensor:
        """Compute flow loss.

        Args:
            predictions: dict with flow prediction with shape
                [N, n_class*2*n_future, H, W]
            ground_truth: ground_truth dict that has 'flow_waypoints'
                with shape [N, H, W, 2, n_class, n_future] and
                'occupancy_waypoints' with shape
                [N, H, W, 2, n_class, n_future]

        """
        # N, H, W, 2, cls, n_future
        preds = (
            predictions["flow_preds"]
            .permute([0, 2, 3, 1])
            .unflatten(3, [2, len(self.compute_loss_cls_index), self.n_future])
        )
        target_flow_waypoints = ground_truth["flow_waypoints"][
            :, :, :, :, self.compute_loss_cls_index, :
        ]
        # only calculate loss for obj area
        target_occupancy_waypoints = ground_truth["occupancy_waypoints"][
            :, :, :, :, self.compute_loss_cls_index, :
        ]
        positive_mask = target_occupancy_waypoints.sum(3, keepdim=True) > 0
        positive_mask = positive_mask.repeat((1, 1, 1, 2, 1, 1))
        positive_mask = positive_mask.float()
        target = target_flow_waypoints
        flow_loss = self.smooth_l1_loss(preds, target)
        total_flow_loss = 0
        if self.enable_time_discount:
            for i in range(self.n_future):
                positive_mask[..., i] = (
                    positive_mask[..., i] * self.time_weights[i]
                )
        total_flow_loss = (flow_loss * positive_mask).sum() / (
            positive_mask.sum() / 2 + 1e-5
        )
        return total_flow_loss


@OBJECT_REGISTRY.register
class FlowTracedLoss(torch.nn.Module):
    """Traced loss for flow-grounded occupancy.

    First use predicted flow to warp GT occupancy from last frame and then
    combine it with predicted occupancy, the result is called flow-grounded
    occupancy. Use flow-grounded occupancy to compute occupancy loss.

    Args:
        flow_dim: flow prediction dimension, default is 2.
        n_future: flow prediction frames, default is 8.
        enable_time_discount: whether to weight flow loss over time.
        discount_factor: the exponetial time weight factor for flow loss.
        loss_type: support "bce" for BCE loss and "focal" for focal loss.
        eps_min: lower-bound of the range for prediction to be clamped to.
        eps_max: upper-bound of the range for prediction to be clamped to.
        gt_weighted_args: has related arguments to weight loss computation
            based on GT distribution. Currently not recommend to use, will
            add more support in the future.
        skip_empty: skip loss computation on grids that have no occupancy.

    """

    def __init__(
        self,
        compute_loss_cls_index,
        flow_dim: int = 2,
        n_future: int = 8,
        enable_time_discount: bool = False,
        discount_factor: float = 1.0,
        loss_type: str = "bce",
        eps_min: float = 1e-4,
        eps_max: float = 1e-4,
        gt_weighted_args: Dict = None,
        skip_empty: bool = False,
    ):
        super(FlowTracedLoss, self).__init__()
        self.loss_type = loss_type
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.compute_loss_cls_index = compute_loss_cls_index
        self.flow_dim = flow_dim
        self.n_future = n_future
        if not gt_weighted_args:
            self.gt_weighted_args = {"enable": False}
        else:
            self.gt_weighted_args = gt_weighted_args

        # exponetial weight decay/ascend on time
        self.enable_time_discount = enable_time_discount
        self.discount_factor = discount_factor
        self.time_weights = [
            self.discount_factor ** (self.n_future - 1 - i)
            for i in range(self.n_future)
        ]
        self.skip_empty = skip_empty

    def forward(
        self, predictions: Dict, ground_truth: Union[Dict, torch.Tensor]
    ) -> torch.Tensor:
        """Compute flow loss.

        Args:
            predictions: dict with occpancy and flow prediction both with shape
                [N, n_class*2*n_future, H, W]
            ground_truth: ground_truth dict that has 'flow_waypoints'
                with shape [N, H, W, 2, n_class, n_future] and
                'occupancy_waypoints' with shape
                [N, H, W, 2, n_class, n_future] and
                'flow_origin_occupancy_waypoints' with shape
                [N, H, W, 1, n_class, n_future]

        """
        num_cls = len(self.compute_loss_cls_index)
        # N, 2*n_future, H, W
        pred_flow = predictions["flow_preds"]
        N, C, H, W = pred_flow.shape
        pred_flow = pred_flow.permute([0, 2, 3, 1]).view(
            (N, H, W, self.flow_dim, num_cls, self.n_future)
        )

        # N, 2*n_future, H, W
        pred_occupancy = predictions["occ_preds"].sigmoid()
        pred_occupancy = pred_occupancy.permute([0, 2, 3, 1]).view(
            (N, H, W, self.flow_dim, num_cls, self.n_future)
        )
        # merge occluded and observed occupancy
        pred_occupancy_all = torch.clip(
            pred_occupancy.sum(dim=3, keepdim=True), min=0.0, max=1.0
        )
        # N, H, W, 2, cls, n_future
        # only calculate loss for obj area
        target_occupancy_waypoints = ground_truth["occupancy_waypoints"][
            :, :, :, :, self.compute_loss_cls_index, :
        ]
        target_occupancy_waypoints_all = torch.clip(
            target_occupancy_waypoints.sum(dim=3, keepdim=True),
            min=0.0,
            max=1.0,
        )
        # N, H, W, cls
        current_occupancy = ground_truth["flow_origin_occupancy_waypoints"][
            :, :, :, 0, self.compute_loss_cls_index, :
        ]
        current_occupancy = current_occupancy.to(pred_flow.dtype)
        warped_flow_origins = flow_warp(current_occupancy, pred_flow)
        flow_joint_occpancy = warped_flow_origins * pred_occupancy_all
        mask = torch.ones_like(pred_occupancy_all)
        if self.skip_empty:
            has_occ_mask = target_occupancy_waypoints_all > 0
            for dimension in [1, 2, 3, 4]:
                has_occ_mask = torch.any(
                    has_occ_mask, dim=dimension, keepdim=True
                )
            for k in range(1, self.n_future):
                flow_mask = torch.logical_or(
                    has_occ_mask[..., k], has_occ_mask[..., k - 1]
                )
                mask[..., k] = mask[..., k] * flow_mask

        if self.enable_time_discount:
            time_weight_tensor = mask.new_tensor(self.time_weights)[
                None, None, None, None, None, :
            ]
            mask = mask * time_weight_tensor

        if self.gt_weighted_args["enable"]:
            gt_dist_weight = ground_truth["gt_dist"]
            gt_dist_weight = torch.min(
                gt_dist_weight, dim=3, keepdim=True
            ).values
            mask = mask * gt_dist_weight

        if self.gt_weighted_args.get("tracked", False):
            loss = 0
            tracked_weight = torch.zeros_like(
                target_occupancy_waypoints_all[..., 0]
            )
            for i in range(self.n_future):
                flow_joint_occpancy_cur = flow_joint_occpancy[..., i]
                target_occupancy_cur = target_occupancy_waypoints_all[..., i]
                mask_cur = mask[..., i] + tracked_weight
                loss_cur, bce_loss_cur = self.calculate_loss(
                    flow_joint_occpancy_cur, target_occupancy_cur, mask_cur
                )
                loss = loss + loss_cur / self.n_future
                tracked_weight = bce_loss_cur.sigmoid()
        else:
            loss, _ = self.calculate_loss(
                flow_joint_occpancy, target_occupancy_waypoints_all, mask
            )
        return loss

    def calculate_loss(self, preds, target, mask):
        if self.loss_type == "bce":
            final_loss = 0
            preds = torch.clamp(preds, self.eps_min, 1 - self.eps_max)
            bce_loss = self.bce_loss(preds.logit(), target)
            final_loss = (bce_loss * mask).sum() / (mask.sum() + 1e-5)
            return final_loss, bce_loss
        elif self.loss_type == "focal_loss":
            loss = self.focal_loss(preds, target)
            return loss, None
        else:
            raise NotImplementedError

    def soft_iou_loss(self, preds, target, mask):
        intersection = preds * target
        union = preds + target - preds * target
        loss = 1 - (intersection * mask).sum() / ((union * mask).sum() + 1e-5)
        return loss, intersection, union

    def focal_loss(self, preds, target):
        preds = torch.clamp(preds, self.eps_min, 1 - self.eps_max)
        pos_inds = target.ge(1).float()
        neg_inds = target.lt(1).float()
        pos_loss = torch.log(preds) * torch.pow(1 - preds, 2) * pos_inds
        neg_loss = torch.log(1 - preds) * torch.pow(preds, 2) * neg_inds
        return -(pos_loss + neg_loss).mean()


@OBJECT_REGISTRY.register
class ProbabilisticLossWaymo(torch.nn.Module):
    """KL-divergence loss for Gaussian distribution.

    Implement based on paper: https://arxiv.org/abs/2104.10490.

    """

    def __init__(
        self,
        scale_weights: List[float] = (1.0,),
        step: int = 0,
        warmup_step: int = 100,
    ):
        super(ProbabilisticLossWaymo, self).__init__()
        self.scale_weights = scale_weights
        self.step = step
        self.warmup_step = warmup_step

    def forward(
        self, predictions: Dict, ground_truth: Union[Dict, torch.Tensor]
    ) -> torch.Tensor:
        output = predictions["output_distributions"]
        present_mu = output["present_mu"]
        present_log_sigma = output["present_log_sigma"]
        future_mu = output["future_mu"]
        future_log_sigma = output["future_log_sigma"]

        final_loss = 0
        for i, sw in enumerate(self.scale_weights):
            var_future = torch.exp(2 * future_log_sigma[i])
            var_present = torch.exp(2 * present_log_sigma[i])
            kl_div = (
                present_log_sigma[i]
                - future_log_sigma[i]
                - 0.5
                + (var_future + (future_mu[i] - present_mu[i]) ** 2)
                / (2 * var_present + 1e-5)
            )
            kl_div = torch.clamp(kl_div, min=-1e4, max=1e4)
            if (
                self.step < self.warmup_step
            ):  # first several iteration is too large
                kl_div = kl_div / 1000
            if len(kl_div.shape) == 4:
                kl_loss = torch.mean(torch.sum(kl_div, dim=1))
            else:
                kl_loss = torch.mean(torch.sum(kl_div, dim=-1))
            final_loss += kl_loss * sw

        self.step += 1
        return final_loss


def flow_warp(
    current_occupancy: torch.Tensor,
    pred_flows: torch.Tensor,
    n_future: int = 8,
) -> torch.Tensor:
    """Warps ground-truth flow-origin occupancies according to predicted flows.

    Performs bilinear interpolation and samples from 4 pixels for each flow
    vector.

    Args:
        current_occupancy: prev occupancy to be warped with shape
            (N, H, W, cls, num_waypoints)
        pred_flows: predicted flows with shape
            (N, H, W, flow_dim, num_cls, num_waypoints)
    Returns:
        List of `num_waypoints` occupancy grids for vehicles as float32
            [batch_size, height, width, 1] tensors.
    """
    device, dtype = pred_flows.device, pred_flows.dtype
    _, grid_height, grid_width, num_cls, _ = current_occupancy.shape
    h = torch.arange(0, grid_height, dtype=dtype, device=device)
    w = torch.arange(0, grid_width, dtype=dtype, device=device)
    h_idx, w_idx = torch.meshgrid(h, w)
    # These indices map each (x, y) location to (x, y).
    # [height, width, 2] but storing x, y coordinates.
    identity_indices = torch.stack(
        (
            torch.transpose(h_idx, 0, 1),
            torch.transpose(w_idx, 0, 1),
        ),
        dim=-1,
    )[
        None, ...
    ]  # (1, W, H, 2)
    warped_flow_origins = []
    for cls_dix in range(num_cls):
        warped_flow_origins_cls = []
        # [batch_size, height, width, 1]
        flow_origin_occupancy_cls = current_occupancy[
            ..., cls_dix : cls_dix + 1, :
        ]
        flow_origin_occupancy_cls = flow_origin_occupancy_cls.permute(
            [0, 3, 1, 2, 4]
        )
        for k in range(n_future):
            flow_origin_occupancy = flow_origin_occupancy_cls[..., k]
            # [batch_size, height, width, cls, 2]
            pred_flow = pred_flows[..., cls_dix, k]
            # Shifting the identity grid indices according to predicted flow
            # tells us the source (origin) grid cell for each flow vector.
            # We simply sample occupancy values from these locations.
            # [batch_size, height, width, 2]
            warped_indices = identity_indices + pred_flow

            # torch grid sampler takes range from -1 to 1, normalize indices
            warped_indices[..., 0] = (
                (warped_indices[..., 0] - grid_height / 2) / (grid_height) * 2
            )
            warped_indices[..., 1] = (
                (warped_indices[..., 1] - grid_width / 2) / (grid_width) * 2
            )
            # [batch_size, height, width, 2]
            warped_origin = F.grid_sample(
                input=flow_origin_occupancy,
                grid=warped_indices,
                padding_mode="border",
            )  # default bilinear
            # warped_origin = warped_origin.permute((0, 2, 3, 1)) # to N,H,W,1
            warped_flow_origins_cls.append(
                warped_origin.permute((0, 2, 3, 1))[..., None, None]
            )
        warped_flow_origins.append(torch.cat(warped_flow_origins_cls, dim=-1))
    warped_flow_origins = torch.cat(warped_flow_origins, dim=-2)
    return warped_flow_origins
