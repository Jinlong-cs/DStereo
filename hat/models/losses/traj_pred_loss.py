# Copyright (c) Horizon Robotics. All rights reserved.

import copy
import math
import sys
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from hat.models.losses.utils import weight_reduce_loss
from hat.registry import OBJECT_REGISTRY
from hat.utils.tensor_func import mean_with_mask, take_row

try:
    sys.path.append("/home/users/shengzhe.dai/cnn1/DenseTNT-argoverse2/src/")
    sys.path.append("/running_package/HAT/DenseTNT-argoverse2/src/")
    sys.path.append("/running_package/code_package/DenseTNT-argoverse2/src/")
    import utils_cython
except Exception:
    warnings.warn(
        "Fail to load the cython optimization lib, if you do not use "
        "DenseTNT, do not care this warning."
    )

__all__ = [  # noqa
    "BasicMultipathLoss",
    "SoftMultipathLoss",
    "TrackValidHeadLoss",
    "BehavHeadLoss",
    "HTTPHeatmapLoss",
    "HTTPOffsetLoss",
    "TrajConfidenceLoss",
    "TrajOffsetRegLoss",
    "SGNetLoss",
    "DenseTNTloss",
    "DenseTNTScoreLoss",
    "DenseTNTStageTwoLoss",
    "SoftDenseTNTloss",
    "DenseTNTPlusloss",
]


def get_min_l2_distance_index(
    trajectory: torch.Tensor,
    anchors: torch.Tensor,
    mask: Optional[List] = None,
    ret_l2_dis: bool = False,
):
    """Find the nearest anchor index for trajectory.

    Args:
        trajectory: the gt trajectories, shape [num_obj, traj_len, 2].
        anchors: the anchors, shape [num_obj, num_anchor, traj_len, 2].
        mask: indices of valid coordinates. Defaults to None.
        ret_l2_dis: whether to return l2 distance tensor.

    Returns:
        index (int): the index of the min l2 distance anchor.
        loss (torch.Tensor): (if enabled) reture the distance loss
            between gt trajectories and anchors.
    """
    # Vectorized implementation.
    n_obj, trj_len = trajectory.shape[:2]
    if mask is None:
        mask = torch.ones([n_obj, trj_len])
    anchors = anchors.to(trajectory.device)
    dist = torch.norm(trajectory[:, None, :, :] - anchors, dim=-1)
    dist[torch.isnan(dist)] = 0
    loss = torch.sum(dist * mask[:, None, :], dim=-1)
    loss /= torch.sum(mask, dim=-1)[:, None] + 1e-6
    if ret_l2_dis:
        return torch.argmin(loss, dim=-1), loss
    else:
        return torch.argmin(loss, dim=-1)


@OBJECT_REGISTRY.register
class BasicMultipathLoss(torch.nn.Module):
    """Calculate the loss of Multipath trajectory prediction.

    The classification loss is the negative log likelihood of the most
    probable anchor. The regression loss is the likelihood of the
    ground truth trajectory in the multivirate Gaussian distribution of
    the most probable anchor (the prediction outputs the mean and
    convariance matrix).

    Args:
        head_weight (List): the weights of the heads.
        reg_loss_scale (int, optional): regression loss scale.
            Default to 1.
        reg_loss_clip_min (float, optional): the min value to
            clip the regression loss.
        reg_loss_clip_max (float, optional): the max value to
            clip the regression loss.
        reg_loss_type (str, optional): the regression loss type,
            must be one of the ['GMM', 'Huber']. Default to 'GMM'.

    Returns:
        losses dict with 'cls_loss' and 'reg_loss'.

    """

    _REG_LOSS_TYPE = ["GMM", "Huber"]

    def __init__(
        self,
        head_weight=1,
        reg_loss_scale=1,
        reg_loss_clip_min=0,
        reg_loss_clip_max=30,
        reg_loss_type="GMM",
    ):
        super(BasicMultipathLoss, self).__init__()
        assert (
            reg_loss_type in self._REG_LOSS_TYPE
        ), f"not allowed reg loss type: {reg_loss_type}"
        self.weight = head_weight
        self.reg_loss_scale = reg_loss_scale
        self.reg_loss_clip_min = reg_loss_clip_min
        self.reg_loss_clip_max = reg_loss_clip_max
        self.reg_loss_type = reg_loss_type
        if reg_loss_type == "Huber":
            self.huber_loss = torch.nn.HuberLoss(reduction="none")
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, model_output, target=None):
        means = model_output["means"]
        gts = model_output["gts"]
        if "anchors" in model_output:
            anchors = model_output["anchors"]
        probabilities = model_output["log_anchors_probs"]
        masks = model_output["masks"]
        scale_trils = model_output["scale_trils"]

        num_obs, num_anchors, traj_len = means.shape[:3]
        if "anchors" in model_output:
            anchors = model_output["anchors"]
        else:
            anchors = torch.zeros([num_obs, num_anchors, traj_len, 2])
        if "log_anchors_probs" in model_output:
            probabilities = model_output["log_anchors_probs"]
        else:
            probabilities = torch.zeros([num_obs, num_anchors])
        if "cur_head_mask" in model_output:
            head_mask = model_output["cur_head_mask"]
        else:
            head_mask = torch.ones([num_obs], dtype=masks.dtype)

        gts = gts.to(means.device)
        masks = masks.to(means.device)
        anchors = anchors.to(means.device)
        probabilities = probabilities.to(means.device)
        head_mask = head_mask.to(means.device)

        # Get min anchor indices.
        if "anchors" in model_output:
            min_anchor_indices = get_min_l2_distance_index(gts, anchors, masks)
        else:
            min_anchor_indices = get_min_l2_distance_index(gts, means, masks)

        # Classifciation loss: logsoftmax + nll.
        cls_loss_fn = torch.nn.NLLLoss(reduction="none")
        cls_loss = cls_loss_fn(probabilities, min_anchor_indices)
        cls_loss = torch.mean(cls_loss * head_mask)

        # Regression loss.
        gts = torch.cat([gt.reshape(-1, 2) for gt in gts], dim=0)
        means = torch.cat(
            [
                mean[min_anchor, :, :].reshape(-1, 2)
                for min_anchor, mean in zip(min_anchor_indices, means)
            ],
            dim=0,
        )
        if self.reg_loss_type == "Huber":
            reg_loss = torch.mean(self.huber_loss(means, gts), dim=1)
            reg_loss = (
                reg_loss.reshape(masks.size()) * masks * head_mask.unsqueeze(1)
            )
        if self.reg_loss_type == "GMM":
            scale_trils = torch.cat(
                [
                    st[min_anchor, :, :, :].reshape(-1, 2, 2)
                    for min_anchor, st, in zip(min_anchor_indices, scale_trils)
                ],
                dim=0,
            )
            reg_loss = -torch.distributions.MultivariateNormal(
                loc=means, scale_tril=scale_trils
            ).log_prob(gts)
            reg_loss = reg_loss.reshape(masks.size())
            reg_loss = torch.clip(
                reg_loss * masks * head_mask.unsqueeze(1),
                min=self.reg_loss_clip_min,
                max=self.reg_loss_clip_max,
            )

        reg_loss = torch.mean(reg_loss) * self.reg_loss_scale

        cls_loss = weight_reduce_loss(cls_loss, self.weight)
        reg_loss = weight_reduce_loss(reg_loss, self.weight)

        return {"cls_loss": cls_loss, "reg_loss": reg_loss}


@OBJECT_REGISTRY.register
class SoftMultipathLoss(BasicMultipathLoss):
    """Calculate the loss of Multipath trajectory prediction.

    In the class `BasicMultipathLoss`, we only choose the top 1
    anchor (by l2 distance to GT) to generate gradients. This
    may cause unbalanced training across different anchors.
    An intuitive phenomenon is although we set more than one
    hundred anchors (and imagine they can all handle some
    situations), the model always output some certain anchors
    with high probabilities.

    In this class, we try to add more samples during training.
    We name this method "soft". If we enable neither `soft_cls`
    nor `soft_reg`. This class will degradates to
    `BasicMultipathLoss`.
    """

    def __init__(
        self,
        head_weight=None,
        reg_loss_scale=1,
        reg_loss_clip_min=0,
        reg_loss_clip_max=25,
        is_soft_cls=False,
        is_soft_reg=False,
        soft_mode="by_l2",
        # Below are the parameters for mode "by_l2"
        num_soft_trajs=3,
        soft_trajs_threshold=4,
    ):
        """Initialize method.

        Args:
            head_weight (List): the weights of the heads.
            reg_loss_scale (int, optional): regression loss scale.
                Default to 1.
            reg_loss_clip_min (float, optional): the min value to
                clip the regression loss.
            reg_loss_clip_max (float, optional): the max value to
                clip the regression loss.
            is_soft_cls (bool, optional): whether to use soft
                classfication.
            is_soft_reg (bool, optional): whether to use soft
                regression.
            soft_mode (str, optional): how to select the anchors
                to use in the soft mode. only support ["by_l2"] now.
            num_soft_trajs (int): the number of the used soft
                trajectories.
            soft_trajs_threshold (float): the threshold of the
                soft trajectories. If the `soft_mode` is "by_l2", this
                paramter is the max mean L2 distance between valid
                soft trajectories and GT. The unit is [m].
        """
        super(SoftMultipathLoss, self).__init__(
            head_weight,
            reg_loss_scale,
            reg_loss_clip_min,
            reg_loss_clip_max,
        )
        self.is_soft_cls = is_soft_cls
        self.is_soft_reg = is_soft_reg
        self.soft_mode = soft_mode
        self.num_soft_trajs = num_soft_trajs
        self.soft_trajs_threshold = soft_trajs_threshold

    def forward(self, model_output, target=None):
        means = model_output["means"]
        gts = model_output["gts"]
        anchors = model_output["anchors"]
        probabilities = model_output["log_anchors_probs"]
        masks = model_output["masks"]
        scale_trils = model_output["scale_trils"]
        if "cur_head_mask" in model_output:
            head_mask = model_output["cur_head_mask"]
        else:
            head_mask = torch.ones([masks.shape[0]], dtype=masks.dtype)
        gts = gts.to(means.device)
        masks = masks.to(means.device)
        head_mask = head_mask.to(means.device)
        n_obj, n_anchors, traj_len, _ = means.shape

        # Get min anchor indices.
        min_anchor_indices, l2_dis = get_min_l2_distance_index(
            gts, anchors, mask=masks, ret_l2_dis=True
        )
        # Regression loss. For computation efficiency, we flatten every
        #   gaussian and ground truth in time and space into 1 dimension.
        soft_loss_mean = []
        soft_loss_scale = []
        soft_loss_gt = []
        batch_l2_probs = []
        batch_anchor_probs = []
        batch_soft_mask = []
        batch_cls_l2_probs = []
        batch_cls_log_anc_probs = []
        num_all_traj_pts = 0
        for min_anc_idx, l2, mean, mask, st, gt, log_anc_prob in zip(
            min_anchor_indices,
            l2_dis,
            means,
            masks,
            scale_trils,
            gts,
            probabilities,
        ):
            if self.is_soft_reg:
                sort_indices = torch.argsort(l2)
                if self.soft_mode == "by_l2":
                    sorted_l2 = l2[sort_indices]
                    valid_indices = sort_indices[
                        sorted_l2 < self.soft_trajs_threshold
                    ]
                    n_trajs = min(len(valid_indices), self.num_soft_trajs)
                    if not n_trajs:
                        continue
                    anc_indices = sort_indices[:n_trajs]
                else:
                    raise ValueError(
                        f"Unsupported soft mode {self.soft_mode}."
                    )
            else:
                n_trajs = 1
                anc_indices = min_anc_idx
            # Calculate l2 probabilities.
            anc_l2 = l2[anc_indices].reshape(-1)
            l2_softmax = torch.exp(-anc_l2)
            l2_prob = l2_softmax / float(torch.sum(l2_softmax))
            # Calculate predicted anchor probabilities.
            log_anc_prob = log_anc_prob[anc_indices].reshape(-1)
            anc_prob = torch.exp(log_anc_prob)
            anc_prob /= float(torch.sum(anc_prob))

            # Save necessary information.
            anc_means = mean[anc_indices, :, :].reshape(-1, traj_len, 2)
            anc_st = st[anc_indices, :, :, :].reshape(-1, traj_len, 2, 2)
            soft_loss_mean.append(anc_means[:, mask, :].reshape(-1, 2))
            soft_loss_scale.append(anc_st[:, mask, :, :].reshape(-1, 2, 2))
            soft_loss_gt.append(gt[mask, :].reshape(-1, 2).repeat(n_trajs, 1))
            # -- Probabilities for classification loss.
            batch_cls_l2_probs.append(l2_prob.reshape(-1))
            batch_cls_log_anc_probs.append(log_anc_prob.reshape(-1))
            # -- Anchor probabilities.
            anc_prob = anc_prob.detach()[:, None].repeat(1, traj_len)
            batch_anchor_probs.append(anc_prob[:, mask].reshape(-1))
            # -- L2 probabilities.
            l2_prob = l2_prob[:, None].repeat(1, traj_len)
            batch_l2_probs.append(l2_prob[:, mask].reshape(-1))
            batch_soft_mask.append(mask)
            num_all_traj_pts += torch.sum(mask)

        num_obj = len(soft_loss_mean)
        means = torch.cat(soft_loss_mean, dim=0)
        scale_trils = torch.cat(soft_loss_scale, dim=0)
        gts = torch.cat(soft_loss_gt, dim=0)
        l2_probs = torch.cat(batch_l2_probs, dim=0)
        # anc_probs = torch.cat(batch_anchor_probs, dim=0)
        cls_l2_probs = torch.cat(batch_cls_l2_probs, dim=0)
        cls_log_anc_probs = torch.cat(batch_cls_log_anc_probs, dim=0)

        # Calculate regression loss.
        loss_tensor = -torch.distributions.MultivariateNormal(
            loc=means, scale_tril=scale_trils
        ).log_prob(gts)
        l2_probs = torch.where(
            torch.isnan(l2_probs), torch.zeros_like(l2_probs), l2_probs
        )
        num_notnan_traj_pts = num_all_traj_pts - torch.sum(
            torch.isnan(l2_probs)
        )
        reg_loss = loss_tensor * l2_probs
        reg_loss = torch.clip(
            reg_loss, min=self.reg_loss_clip_min, max=self.reg_loss_clip_max
        )
        reg_loss = torch.where(
            torch.isnan(reg_loss), torch.zeros_like(reg_loss), reg_loss
        )
        reg_loss = torch.sum(reg_loss) / num_notnan_traj_pts

        # Calculate classification loss.
        if self.is_soft_cls:
            # Use cross entropy: -sum (y_i * log P_i)
            #   y_i: L2 probabilities; P_i: anchor probabilities.
            cls_loss = -torch.sum(cls_l2_probs * cls_log_anc_probs)
            cls_loss /= num_obj
        else:
            cls_loss_fn = torch.nn.NLLLoss(reduction="none")
            cls_loss = cls_loss_fn(probabilities, min_anchor_indices)
            cls_loss = torch.mean(cls_loss * head_mask)

        all_head_cls_loss = weight_reduce_loss(cls_loss, self.weight)
        all_head_reg_loss = weight_reduce_loss(reg_loss, self.weight)

        return {
            "cls_loss": all_head_cls_loss,
            "reg_loss": all_head_reg_loss,
        }


@OBJECT_REGISTRY.register
class TrackValidHeadLoss(torch.nn.Module):
    """Calculate the loss of Track Valid head."""

    def __init__(
        self,
        head_weight: float = 1,
        positive_loss_weight_veh: float = 0.8,
        negative_loss_weight_veh: float = 0.2,
        positive_loss_weight_ped: float = 0.3,
        negative_loss_weight_ped: float = 0.7,
        positive_loss_weight_cyc: float = 0.55,
        negative_loss_weight_cyc: float = 0.45,
    ):
        """Initialize method.

        Args:
            head_weight: the weights of the heads.
            positive_loss_weight: the loss weight of positive samples.
            negative_loss_weight: the loss weight of negative samples.
        """
        super(TrackValidHeadLoss, self).__init__()
        self.weight = head_weight
        # filted obs (0) is positive
        self.positive_loss_weight_veh = positive_loss_weight_veh
        self.negative_loss_weight_veh = negative_loss_weight_veh
        self.positive_loss_weight_ped = positive_loss_weight_ped
        self.negative_loss_weight_ped = negative_loss_weight_ped
        self.positive_loss_weight_cyc = positive_loss_weight_cyc
        self.negative_loss_weight_cyc = negative_loss_weight_cyc

    def forward(self, model_output: Dict, target=None):
        """Forward.

        Args:
            model_output: should contains the keys:
                "track_valid_cls" and "track_id_gt".
        """
        track_valid_cls = model_output["track_valid_cls"]
        track_id_gt = model_output["track_id_gt"]
        track_id_cls = model_output["batch_history_valid_class"]
        device = track_valid_cls.device
        track_id_gt = track_id_gt.to(device)
        track_id_cls = track_id_cls.to(device).long()

        track_index = torch.arange(track_valid_cls.shape[0])
        track_valid_cls = track_valid_cls[track_index, track_id_cls]
        # Calculate track valid classification loss.
        # Set different weights to different class.
        weight = torch.zeros_like(track_id_gt).float().to(device)
        weight[
            (track_id_gt > 0.5) & (track_id_cls == 0)
        ] = self.negative_loss_weight_veh
        weight[
            (track_id_gt <= 0.5) & (track_id_cls == 0)
        ] = self.positive_loss_weight_veh
        weight[
            (track_id_gt > 0.5) & (track_id_cls == 1)
        ] = self.negative_loss_weight_ped
        weight[
            (track_id_gt <= 0.5) & (track_id_cls == 1)
        ] = self.positive_loss_weight_ped
        weight[
            (track_id_gt > 0.5) & (track_id_cls == 2)
        ] = self.negative_loss_weight_cyc
        weight[
            (track_id_gt <= 0.5) & (track_id_cls == 2)
        ] = self.positive_loss_weight_cyc

        track_valid_cls_loss_fn = torch.nn.BCELoss(
            weight=weight, reduction="mean"
        )
        track_valid_cls = track_valid_cls.reshape(track_id_gt.shape)
        track_valid_loss = track_valid_cls_loss_fn(
            track_valid_cls, track_id_gt
        )
        all_head_track_valid_loss = weight_reduce_loss(
            track_valid_loss, self.weight
        )
        losses = {"track_valid_loss": all_head_track_valid_loss}

        return losses


@OBJECT_REGISTRY.register
class BehavHeadLoss(torch.nn.Module):
    """Calculate the loss of Behav head."""

    def __init__(
        self,
        lat_head_weight: float = 1.0,
        lon_head_weight: float = 1.0,
        lat_class_weight: List = (1.0, 1.0, 1.0),
        lon_class_weight: List = (1.0, 1.0, 1.0),
    ):
        """Initialize method.

        Args:
            lat_head_weight: the weight of the lateral behavior predict head.
            lon_head_weight: the weight of the longitudinal behavior predict
                head.
            lat_class_weight: label weights of three lateral behaviors.
            lon_class_weight: label weights of three longitudinal behaviors.
        """
        super(BehavHeadLoss, self).__init__()
        self.lat_head_weight = lat_head_weight
        self.lon_head_weight = lon_head_weight
        self.lat_class_weight = torch.FloatTensor(lat_class_weight)
        self.lon_class_weight = torch.FloatTensor(lon_class_weight)
        self.loss_fn = F.cross_entropy

    def forward(self, model_output: Dict):
        """Forward.

        Args:
            model_output: should contains the keys: "lat_behav_probs",
                "lat_behav_gts", "lon_behav_probs", "lon_behav_gts".
        """
        lat_behav_probs = model_output["lat_behav_probs"].view(-1, 3)
        lon_behav_probs = model_output["lon_behav_probs"].view(-1, 3)
        device = lat_behav_probs.device

        lat_behav_gts = model_output["lat_behav_gts"].to(device)
        lon_behav_gts = model_output["lon_behav_gts"].to(device)
        lat_class_weight = self.lat_class_weight.to(device)
        lon_class_weight = self.lon_class_weight.to(device)

        lat_valid_mask = lat_behav_gts >= 0
        if lat_valid_mask.sum() == 0:
            lat_behav_loss = 0
        else:
            lat_behav_loss = self.loss_fn(
                lat_behav_probs[lat_valid_mask],
                lat_behav_gts[lat_valid_mask],
                lat_class_weight,
            )

        lon_valid_mask = lon_behav_gts >= 0
        if lon_valid_mask.sum() == 0:
            lon_behav_loss = 0
        else:
            lon_behav_loss = self.loss_fn(
                lon_behav_probs[lon_valid_mask],
                lon_behav_gts[lon_valid_mask],
                lon_class_weight,
            )
        losses = {
            "lat_behav_pred_loss": lat_behav_loss * self.lat_head_weight,
            "lon_behav_pred_loss": lon_behav_loss * self.lon_head_weight,
        }

        return losses


class FocalLoss(torch.nn.Module):
    """
    Focal loss for imbalanced behavior classification.

    Args:
        num_labels: number of classes.
        cls_weights: loss weight for different classes.
        activation_type: the way to normalize the predicted vector, only
            sigmoid and softmax.
        gamma: hard examples mining hyperparameter for focal loss.
        alpha: class balanced hyperparameter for focal loss.
        epsilon: a value added to the antilogarithm for numerical stability.
    """

    def __init__(
        self,
        num_labels: float,
        cls_weights: List,
        activation_type: str = "softmax",
        gamma: float = 2.0,
        epsilon: float = 1e-9,
    ):
        super(FocalLoss, self).__init__()
        self.num_labels = num_labels
        self.cls_weights = torch.FloatTensor(cls_weights)
        self.gamma = gamma
        self.epsilon = epsilon
        self.activation_type = activation_type

    def forward(self, input, target, weight):
        """Forward.

        Args:
            input (torch.FloatTensor, [num_obj, num_labels]): model output
                pred vector.
            target (torch.LongTensor, [num_obj]): ground truth labels.
            weight (torch.FloatTensor, [num_obj]): loss mask for valid
                behavior object.

        Returns:
            loss (float): mean loss of all valid obstacles.
        """
        assert self.activation_type in [
            "softmax",
            "sigmoid",
        ], f"Unsupported activation type: {self.activation_type}!"
        alpha = self.cls_weights.to(input.device)
        valid_num = (weight != 0).sum()
        if self.activation_type == "softmax":
            idx = target.view(-1, 1)
            onehot_key = torch.zeros(
                idx.size(0),
                self.num_labels,
                dtype=torch.float32,
                device=idx.device,
            )
            onehot_key = onehot_key.scatter_(1, idx, 1)
            logits = torch.softmax(input, dim=-1)
            loss = (
                -alpha
                * onehot_key
                * torch.pow((1 - logits), self.gamma)
                * (logits + self.epsilon).log()
            )
            loss = loss.sum(1)
        elif self.activation_type == "sigmoid":
            multi_hot_key = target
            logits = torch.sigmoid(input)
            zero_hot_key = 1 - multi_hot_key
            loss = (
                -alpha
                * multi_hot_key
                * torch.pow((1 - logits), self.gamma)
                * (logits + self.epsilon).log()
            )
            loss += (
                -(1 - alpha)
                * zero_hot_key
                * torch.pow(logits, self.gamma)
                * (1 - logits + self.epsilon).log()
            )

        return (loss * weight).sum() / (valid_num + self.epsilon)


@OBJECT_REGISTRY.register
class HTTPHeatmapLoss(torch.nn.Module):
    """
    Heatmap loss for HeaTmap-based Trajectory Predictor.

    Args:
        alpha: Hyperparameter for pos loss and focal loss coefficient.
        beta: Hyperparameter for focal loss coefficient.
        eps: A value added to the antilogarithm for numerical stability.
        pos_threshold: The lower threshold on the heatmap gt of the positive
            sample.
        pos_loss_weight: Loss weight for positive samples.
        neg_loss_weight: Loss weight for negative samples.
        impassable_loss_weight: Weight for the negative samples in impassable
            area.
        ignore_neg_num: The number of neg samples with the highest confidence
            to be ignored.
        hard_neg_threshold: The neg samples, with pred confidence below this
            threshold, will be ignored.
        pred_norm: Whether to normalize the input prediction by sigmoid.
        focal_loss: Whether to multiply focal loss coefficient.
        loss_weight: Weight before output.
        reduction: The way to reduce loss along batch dimension.
    """

    def __init__(
        self,
        alpha: float = 2,
        beta: float = 4,
        eps: float = 1e-12,
        pos_threshold: float = 0.7,
        pos_loss_weight: float = 1,
        neg_loss_weight: float = 1,
        impassable_loss_weight: float = 10.0,
        ignore_neg_num: int = 10,
        hard_neg_threshold: float = 1e-3,
        pred_norm: bool = False,
        focal_loss: bool = False,
        loss_weight: float = 1.0,
        reduction: str = "mean",
    ):
        super(HTTPHeatmapLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.pos_threshold = pos_threshold
        self.pos_loss_weight = pos_loss_weight
        self.neg_loss_weight = neg_loss_weight
        self.impassable_loss_weight = impassable_loss_weight
        self.ignore_neg_num = ignore_neg_num
        self.hard_neg_threshold = hard_neg_threshold
        self.pred_norm = pred_norm
        self.focal_loss = focal_loss
        self.loss_weight = loss_weight
        self.reduction = reduction

    @autocast(enabled=False)
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        passable_masks: Optional[torch.Tensor] = None,
    ):
        if len(pred.shape) == 4:
            pred = pred[:, 0]
        if self.pred_norm:
            pred = torch.nn.functional.sigmoid(pred)

        N, H, W = pred.shape
        pos_mask = target >= self.pos_threshold

        loss_map = torch.where(
            pos_mask,
            -torch.log(pred + self.eps)
            * torch.pow(target, self.beta)
            * self.pos_loss_weight,
            -torch.log(1 - pred + self.eps) * self.neg_loss_weight,
        )
        if self.focal_loss:
            coefficient = torch.where(
                pos_mask,
                torch.pow(1 - pred, self.alpha),
                torch.pow(pred, self.alpha) * torch.pow(1 - target, self.beta),
            )
            loss_map = loss_map * coefficient

        pos_loss = torch.where(pos_mask, loss_map, pred.new_tensor(0.0))
        pos_loss = pos_loss.sum(dim=[1, 2]) / torch.maximum(
            target.new_tensor(1.0), pos_mask.sum(dim=[1, 2])
        )

        neg_loss = torch.where(
            pos_mask, -torch.log(pred.new_tensor(self.eps / 10)), loss_map
        )
        neg_loss, indices = torch.sort(neg_loss.reshape(N, H * W), dim=-1)
        pos_mask = self._sort_by(pos_mask.reshape(N, H * W), indices)
        neg_mask = torch.logical_not(pos_mask)

        if passable_masks is not None and self.impassable_loss_weight != 1:
            passable_masks = self._sort_by(
                passable_masks.reshape(N, H * W), indices
            )
            neg_loss = torch.where(
                torch.logical_and(passable_masks < 1e-3, neg_mask),
                neg_loss * self.impassable_loss_weight,
                neg_loss,
            )
        neg_loss = neg_loss * neg_mask

        # simple mining
        neg_num = torch.sum(neg_mask, dim=-1).min() - self.ignore_neg_num
        neg_loss = neg_loss[:, :neg_num]
        # hard mining
        neg_loss = torch.where(
            neg_loss < self.hard_neg_threshold,
            neg_loss.new_tensor(0),
            neg_loss,
        ).sum(dim=1) / torch.maximum(
            (neg_loss >= self.hard_neg_threshold).sum(dim=1),
            neg_loss.new_tensor(1),
        )

        loss = pos_loss + neg_loss

        loss = (
            weight_reduce_loss(loss, reduction=self.reduction)
            * self.loss_weight
        )
        return loss

    def _sort_by(self, value, index):
        shape = value.shape
        for i, s in enumerate(shape[:-1]):
            temp = torch.arange(s, device=index.device)
            for t in shape[i + 1 :]:
                temp *= t
                temp = torch.unsqueeze(temp, dim=-1)
            index = index + temp
        index = index.reshape(-1)
        value = torch.index_select(value.reshape(-1), index=index, dim=0)
        value = value.reshape(shape)
        return value


@OBJECT_REGISTRY.register
class HTTPOffsetLoss(torch.nn.Module):
    """
    Offset loss for heatmap_head of HeaTmap-based Trajectory Predictor. \
    loss = mean(abs(pred - offset_gt))_{heatmap_gt > pos_threshold}.

    Args:
        loss_weight: Weight before output.
        pos_threshold: The lower threshold on the heatmap gt of the positive
            sample.
        reduction: The way to reduce loss along batch dimension.
    """

    def __init__(
        self,
        loss_weight: float = 1,
        pos_threshold: float = 0.95,
        reduction: str = "mean",
    ):
        super(HTTPOffsetLoss, self).__init__()
        self.loss_weight = loss_weight
        self.pos_threshold = pos_threshold
        self.reduction = reduction

    @autocast(enabled=False)
    def forward(
        self,
        offset: torch.Tensor,
        offset_gt: torch.Tensor,
        heatmap_gt: torch.Tensor,
    ):
        pos_mask = heatmap_gt > self.pos_threshold
        loss = torch.where(
            pos_mask,
            torch.abs(offset_gt - offset).sum(dim=1),
            offset.new_tensor(0),
        )
        loss = mean_with_mask(
            loss.flatten(start_dim=1), pos_mask.flatten(start_dim=1)
        )
        loss = weight_reduce_loss(loss, reduction=self.reduction)
        return loss


@OBJECT_REGISTRY.register
class TrajConfidenceLoss(torch.nn.Module):
    """
    Confidence loss for traj_reg_head of HeaTmap-based Trajectory Predictor. \
    ade = mean((traj_pred - traj_gt) ^ 2) \
    loss = - ade x log(pred) - (1 - ade) x log(1 - pred).

    Args:
        loss_weight: Weight before output.
        pred_norm: Whether to normalize the input prediction by sigmoid.
        reduction: The way to reduce loss along batch dimension.
    """

    def __init__(
        self,
        loss_weight: float = 1,
        pred_norm: bool = False,
        eps: float = 1e-12,
        reduction: str = "mean",
    ):
        super(TrajConfidenceLoss, self).__init__()
        self.loss_weight = loss_weight
        self.pred_norm = pred_norm
        self.eps = eps
        self.reduction = reduction

    @autocast(enabled=False)
    def forward(
        self,
        confidence: torch.Tensor,
        traj_pred_decoded: torch.Tensor,
        traj_gt: torch.Tensor,
        future_mask: Optional[torch.Tensor] = None,
    ):
        if self.pred_norm:
            confidence = torch.nn.functional.sigmoid(confidence)
        N, M, T, _ = traj_pred_decoded.shape
        ade = torch.norm(traj_pred_decoded - traj_gt[:, None], p=2, dim=-1)
        if future_mask:
            ade = mean_with_mask(ade, future_mask[:, None])
        else:
            ade = ade.mean(dim=-1)
        gt = torch.exp(-ade)

        loss = -gt * torch.log(confidence + self.eps) - (1 - gt) * torch.log(
            1 - confidence + self.eps
        )
        loss = weight_reduce_loss(loss, reduction=self.reduction)
        return loss * self.loss_weight


@OBJECT_REGISTRY.register
class TrajOffsetRegLoss(torch.nn.Module):
    """
    Offset loss for heatmap_head of HeaTmap-based Trajectory Predictor. \
    error = abs(pred - (traj_gt - anchor)) \
    loss = clip(error, min=sigma) ^ 2.

    Args:
        sigma: Threshold of insensitive error function.
        pos_threshold: The lower threshold on the heatmap gt of the positive
            sample.
        impossable_loss_weight: Weight for the predction which out of passable
            area.
        loss_weight: Weight before output.
        reduction: The way to reduce loss along batch dimension.
    """

    def __init__(
        self,
        sigma: float = 1,
        pos_threshold: float = 2,
        impassable_loss_weight: float = 1,
        loss_weight: float = 1,
        reduction: str = "mean",
    ):
        super(TrajOffsetRegLoss, self).__init__()
        self.sigma = sigma
        self.pos_threshold = pos_threshold
        self.impassable_loss_weight = impassable_loss_weight
        self.loss_weight = loss_weight
        self.reduction = reduction

    @autocast(enabled=False)
    def forward(
        self,
        traj_pred: torch.Tensor,
        traj_gt: torch.Tensor,
        endpoint: torch.Tensor,
        future_mask: torch.Tensor,
        passable_mask: Optional[torch.Tensor] = None,
    ):
        N, M, T, _ = traj_pred.shape

        # label assign
        endpoint_gt = traj_gt[:, -1:]
        delta = torch.norm(endpoint - endpoint_gt, p=2, dim=-1)
        match_mask = torch.logical_or(
            delta < self.pos_threshold,
            delta == delta.min(dim=-1, keepdim=True).values,
        )

        anchor = (
            endpoint[:, :, None]
            / endpoint.new_tensor(T)
            * torch.arange(1, T + 1, device=endpoint.device).reshape(
                1, 1, T, 1
            )
        )
        offset_gt = traj_gt[:, None] - anchor
        diff = traj_pred - offset_gt
        distance_error = torch.norm(diff, dim=-1, p=2)
        loss = self._loss_func(distance_error)

        if passable_mask is not None and self.impassable_loss_weight != 1:
            H, W = passable_mask.shape[1:]
            traj_decoded = torch.clamp(traj_pred + anchor, min=0)
            traj_decoded = torch.stack(
                [
                    torch.clamp(traj_decoded[..., 0], max=H),
                    torch.clamp(traj_decoded[..., 1], max=W),
                ],
                axis=-1,
            )
            traj_decoded = torch.round(
                torch.reshape(traj_decoded, [N, M * T, 2])
            )
            index = traj_decoded[..., 0] * W + traj_decoded[..., 1]
            passable_flag = take_row(passable_mask.reshape(N, -1), index)
            passable_flag = passable_flag.reshape(N, M, T)
            loss = torch.where(
                passable_flag.to(dtype=torch.bool),
                loss,
                loss * self.impassable_loss_weight,
            )

        if future_mask is not None:
            loss = mean_with_mask(loss, future_mask[:, None])
        else:
            loss = loss.mean(dim=-1)
        loss = mean_with_mask(loss, match_mask)
        loss = weight_reduce_loss(loss, reduction=self.reduction)
        return loss * self.loss_weight

    def _loss_func(
        self, error: torch.Tensor, coefficient: Optional[torch.Tensor] = None
    ):
        error = torch.pow(torch.clamp(error - self.sigma, min=0), 2)
        if coefficient is not None:
            coefficient = torch.clamp(torch.abs(coefficient), max=10)
            coefficient = torch.where(
                coefficient < self.sigma,
                coefficient.new_tensor(0.1),
                coefficient,
            )
            error = error * coefficient
        return error


@OBJECT_REGISTRY.register
class MTPLoss(torch.nn.Module):
    """Multiple-Trajectory Prediction loss.

    Essentially it is a sum of classification and regression loss.
    The classification loss tries to penalize modes not matching gt.
    The regression loss measures how close the matching mode is to the gt.

    For details see <https://arxiv.org/pdf/1809.10732.pdf>

    Args:
        use_variance: whether the prediction contains variance. Defaults to
            False.
        alpha: weight for the classification loss. Defaults to 1.0.
        beta: weight for the regression loss. Defaults to 1.0
    """

    def __init__(
        self, use_variance: bool = False, alpha: float = 1.0, beta: float = 1.0
    ):
        """Initialize MTP loss."""
        super(MTPLoss, self).__init__()
        self.use_variance = use_variance
        self.alpha = alpha
        self.beta = beta

    def forward(
        self, predictions: Dict, ground_truth: Union[Dict, torch.Tensor]
    ) -> torch.Tensor:
        """Compute MTP loss.

        Args:
            predictions: Dictionary with
                'traj': predicted trajectories
                'probs': mode (log) probabilities
            ground_truth: Either a tensor with ground truth trajectories
                or a dictionary
        """

        # Unpack arguments
        traj = predictions["traj"]
        log_probs = predictions["probs"]
        traj_gt = (
            ground_truth["traj"]
            if isinstance(ground_truth, dict)
            else ground_truth
        )

        # Useful variables
        batch_size = traj.shape[0]
        sequence_length = traj.shape[2]
        pred_params = 5 if self.use_variance else 2

        # Masks for variable length ground truth trajectories
        if isinstance(ground_truth, dict) and "masks" in ground_truth:
            masks = ground_truth["masks"]
        else:
            masks = torch.zeros(batch_size, sequence_length).to(traj.device)

        # Obtain mode with minimum ADE with respect to ground truth:
        errs, inds = self.min_ade(traj, traj_gt, masks)
        inds_rep = inds.repeat(sequence_length, pred_params, 1, 1).permute(
            3, 2, 0, 1
        )

        # Calculate MSE or NLL loss for trajectories corresponding to selected
        # outputs:
        traj_best = traj.gather(1, inds_rep).squeeze(dim=1)

        if self.use_variance:
            l_reg = self.traj_nll(traj_best, traj_gt, masks)
        else:
            l_reg = errs

        # Compute classification loss
        l_class = -torch.squeeze(log_probs.gather(1, inds.unsqueeze(1)))

        loss = self.beta * l_reg + self.alpha * l_class
        loss = torch.mean(loss)

        return loss

    @staticmethod
    def min_ade(
        traj: torch.Tensor, traj_gt: torch.Tensor, masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute average displacement error and find the best trajectory.

        Args:
            traj: predictions, shape
                [batch_size, num_modes, sequence_length, 2]
            traj_gt: ground truth trajectory, shape
                [batch_size, sequence_length, 2]
            masks: masks for varying length ground truth, shape
                [batch_size, sequence_length]

        Returns:
            errs, inds: errors and indices for modes with min error, shape
                [batch_size]
        """
        num_modes = traj.shape[1]

        traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1)
        masks_rpt = masks.unsqueeze(1).repeat(1, num_modes, 1)
        err = traj_gt_rpt - traj[:, :, :, 0:2]
        err = torch.pow(err, exponent=2)
        err = torch.sum(err, dim=3)
        err = torch.pow(err, exponent=0.5)
        err = torch.sum(err * (1 - masks_rpt), dim=2) / torch.clip(
            torch.sum((1 - masks_rpt), dim=2), min=1
        )
        err, inds = torch.min(err, dim=1)

        return err, inds

    @staticmethod
    def traj_nll(
        pred_dist: torch.Tensor, traj_gt: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        """Compute negative log likelihood of a single-mode trajectory.

        The trajectory consists of a bivariate Gaussian distribution predicted
        at each time in the prediction horizon.

        Args:
            pred_dist: parameters of a bivariate Gaussian distribution,
                shape [batch_size, sequence_length, 5]
            traj_gt: ground truth trajectory,
                shape [batch_size, sequence_length, 2]
            masks: masks for varying length ground truth,
                shape [batch_size, sequence_length]

        Returns:
            Negative log likehood loss.
        """
        mu_x = pred_dist[:, :, 0]
        mu_y = pred_dist[:, :, 1]
        x = traj_gt[:, :, 0]
        y = traj_gt[:, :, 1]

        sig_x = pred_dist[:, :, 2]
        sig_y = pred_dist[:, :, 3]
        rho = pred_dist[:, :, 4]
        ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)

        nll = (
            0.5
            * torch.pow(ohr, 2)
            * (
                torch.pow(sig_x, 2) * torch.pow(x - mu_x, 2)
                + torch.pow(sig_y, 2) * torch.pow(y - mu_y, 2)
                - 2
                * rho
                * torch.pow(sig_x, 1)
                * torch.pow(sig_y, 1)
                * (x - mu_x)
                * (y - mu_y)
            )
            - torch.log(sig_x * sig_y * ohr)
            + 1.8379
        )

        nll[nll.isnan()] = 0
        nll[nll.isinf()] = 0

        nll = torch.sum(nll * (1 - masks), dim=1) / torch.sum(
            (1 - masks), dim=1
        )
        # Note: Normalizing with torch.sum((1 - masks), dim=1) makes values
        # somewhat comparable for trajectories of
        # different lengths

        return nll


@OBJECT_REGISTRY.register
class SGNetLoss(torch.nn.Module):
    """SGNet pedestrian prediction loss.

    The SGNet loss consists of four parts: KLD loss, goal loss,
    pred loss and probability loss. Futhermore, goal loss and pred loss
    consist of three parts: bbox loss, iou loss and mpb loss. For details,
    please visit/read/refer to:
    https://horizonrobotics.feishu.cn/docs/doccngGQjuzhDycL5EAWt9v2fpg
    """

    def __init__(
        self,
        KLD_weight: float = 1000.0,
        goal_weight: float = 0.015,
        pred_weight: float = 0.01,
        prob_weight: float = 1.0,
        bbox_loss_weight: float = 1.0,
        iou_loss_weight: float = 60.0,
        mpb_loss_weight: float = 1.0,
        iou_loss_type: str = "iou_loss",
        eps: float = 1e-6,
    ):
        """Initialize method.

        Args:
            KLD_weight: the weight of KLD loss.
            goal_weight: the weight of goal loss.
            pred_weight: the weight of pred loss.
            prob_weight: the weight of prob loss.
            bbox_loss_weight: the weight of bbox loss.
            iou_loss_weight: the weight of iou loss.
            mpb_loss_weight: the weight of mpb loss.
            eps: a small positive value that prevents division by zero.
        """
        super(SGNetLoss, self).__init__()
        self.KLD_weight = KLD_weight
        self.goal_weight = goal_weight
        self.pred_weight = pred_weight
        self.prob_weight = prob_weight
        self.bbox_weight = bbox_loss_weight
        self.iou_weight = iou_loss_weight
        self.mpb_weight = mpb_loss_weight
        self.eps = eps
        self.iou_loss_type = iou_loss_type

    def min_L2_loss(self, x_pred: torch.Tensor, x_true: torch.Tensor):
        """L2 loss calculate function.

        Args:
            x_pred (batch_size, K, step, dim): the model pred output.
            x_true (batch_size, K, step, dim): the groundtruth.

        Returns:
            min_L2_loss: the mini-batch mean min_L2 loss over K trajs.
        """
        batch_size = x_pred.shape[0]
        # [batch_size, K]
        square_L2_loss = torch.square(x_pred - x_true).sum((2, 3))
        L2_loss = torch.sqrt(square_L2_loss + self.eps)
        min_L2_loss_id = torch.argmin(L2_loss, dim=-1)
        min_L2_loss = L2_loss[range(batch_size), min_L2_loss_id].mean()

        return min_L2_loss

    def min_iou_loss(self, x_pred: torch.Tensor, x_true: torch.Tensor):
        """Iou loss calculate function.

        NOTE[zhanbo01.li]: the input bbox is x1y1x2y2, the last dimension
        of x_pred and x_true is the coords of bbox:
        x[:, :, :, 0]: x coord of the top left point of bbox.
        x[:, :, :, 1]: y coord of the top left point of bbox.
        x[:, :, :, 2]: x coord of the bottom right point of bbox.
        x[:, :, :, 3]: y coord of the bottom right point of bbox.

        Args:
            x_pred (batch_size, K, step, dim): the model pred output.
            x_true (batch_size, K, step, dim): the groundtruth.

        Returns:
            min_iou_loss: the mini-batch min_iou loss over K trajs.
        """
        batch_size = x_pred.shape[0]
        # [batch_size, K, dec_step]
        p_x1, p_y1 = x_pred[:, :, :, 0], x_pred[:, :, :, 1]
        p_x2, p_y2 = x_pred[:, :, :, 2], x_pred[:, :, :, 3]
        t_x1, t_y1 = x_true[:, :, :, 0], x_true[:, :, :, 1]
        t_x2, t_y2 = x_true[:, :, :, 2], x_true[:, :, :, 3]

        x1 = torch.max(p_x1, t_x1)
        y1 = torch.max(p_y1, t_y1)
        x2 = torch.min(p_x2, t_x2)
        y2 = torch.min(p_y2, t_y2)

        w = (x2 - x1).clamp(0.0)
        h = (y2 - y1).clamp(0.0)

        inter_aera = w * h
        p_aera = (p_x2 - p_x1) * (p_y2 - p_y1)
        t_aera = (t_x2 - t_x1) * (t_y2 - t_y1)

        union_aera = p_aera + t_aera - inter_aera + self.eps

        # [batch_size, K]
        iou = (inter_aera / union_aera).clamp(self.eps).mean(dim=-1)

        # Each iou should be less than 1.
        bool_iou = iou <= 1
        assert bool_iou.all(), f"iou must <= 1, {iou}"

        iou_loss = -iou.log()
        min_iou_loss_id = torch.argmin(iou_loss, dim=-1)
        min_iou_loss = iou_loss[range(batch_size), min_iou_loss_id].mean()

        return min_iou_loss

    def min_giou_loss(self, x_pred: torch.Tensor, x_true: torch.Tensor):
        """Giou loss calculate function.

        NOTE[zhanbo01.li]: the input bbox is x1y1x2y2, the last dimension
        of x_pred and x_true is the coords of bbox:
        x[:, :, :, 0]: x coord of the top left point of bbox.
        x[:, :, :, 1]: y coord of the top left point of bbox.
        x[:, :, :, 2]: x coord of the bottom right point of bbox.
        x[:, :, :, 3]: y coord of the bottom right point of bbox.

        Args:
            x_pred (batch_size, K, step, dim): the model pred output.
            x_true (batch_size, K, step, dim): the groundtruth.

        Returns:
            min_giou_loss: the mini-batch min_giou loss over K trajs.
        """
        batch_size = x_pred.shape[0]
        # [batch_size, K, dec_step]
        p_x1, p_y1 = x_pred[:, :, :, 0], x_pred[:, :, :, 1]
        p_x2, p_y2 = x_pred[:, :, :, 2], x_pred[:, :, :, 3]
        t_x1, t_y1 = x_true[:, :, :, 0], x_true[:, :, :, 1]
        t_x2, t_y2 = x_true[:, :, :, 2], x_true[:, :, :, 3]

        x1 = torch.max(p_x1, t_x1)
        y1 = torch.max(p_y1, t_y1)
        x2 = torch.min(p_x2, t_x2)
        y2 = torch.min(p_y2, t_y2)

        w = (x2 - x1).clamp(0.0)
        h = (y2 - y1).clamp(0.0)

        inter_aera = w * h
        p_aera = (p_x2 - p_x1) * (p_y2 - p_y1)
        t_aera = (t_x2 - t_x1) * (t_y2 - t_y1)

        union_aera = p_aera + t_aera - inter_aera + self.eps

        # [batch_size, K]
        iou = inter_aera / union_aera

        # area_C
        x1 = torch.min(p_x1, t_x1)
        y1 = torch.min(p_y1, t_y1)
        x2 = torch.max(p_x2, t_x2)
        y2 = torch.max(p_y2, t_y2)
        area_C = (x2 - x1) * (y2 - y1)

        add_area = union_aera
        end_area = (area_C - add_area) / (area_C + self.eps)

        giou = (iou - end_area).clamp(-1).mean(dim=-1)

        # Each iou should be less than 1.
        bool_giou = giou <= 1
        assert bool_giou.all(), f"iou must <= 1, {giou}"

        giou_loss = 1 - giou
        min_giou_loss_id = torch.argmin(giou_loss, dim=-1)
        min_giou_loss = giou_loss[range(batch_size), min_giou_loss_id].mean()

        return min_giou_loss

    def min_diou_loss(self, x_pred: torch.Tensor, x_true: torch.Tensor):
        """Diou loss calculate function.

        NOTE[zhanbo01.li]: the input bbox is x1y1x2y2, the last dimension
        of x_pred and x_true is the coords of bbox:
        x[:, :, :, 0]: x coord of the top left point of bbox.
        x[:, :, :, 1]: y coord of the top left point of bbox.
        x[:, :, :, 2]: x coord of the bottom right point of bbox.
        x[:, :, :, 3]: y coord of the bottom right point of bbox.

        Args:
            x_pred (batch_size, K, step, dim): the model pred output.
            x_true (batch_size, K, step, dim): the groundtruth.

        Returns:
            min_diou_loss: the mini-batch min_diou loss over K trajs.
        """
        batch_size = x_pred.shape[0]
        # [batch_size, K, dec_step]
        p_x1, p_y1 = x_pred[:, :, :, 0], x_pred[:, :, :, 1]
        p_x2, p_y2 = x_pred[:, :, :, 2], x_pred[:, :, :, 3]
        t_x1, t_y1 = x_true[:, :, :, 0], x_true[:, :, :, 1]
        t_x2, t_y2 = x_true[:, :, :, 2], x_true[:, :, :, 3]

        x1 = torch.max(p_x1, t_x1)
        y1 = torch.max(p_y1, t_y1)
        x2 = torch.min(p_x2, t_x2)
        y2 = torch.min(p_y2, t_y2)

        w = (x2 - x1).clamp(0.0)
        h = (y2 - y1).clamp(0.0)

        inter_aera = w * h
        p_aera = (p_x2 - p_x1) * (p_y2 - p_y1)
        t_aera = (t_x2 - t_x1) * (t_y2 - t_y1)

        union_aera = p_aera + t_aera - inter_aera + self.eps

        # [batch_size, K]
        iou = inter_aera / union_aera
        # diou area
        x1 = torch.min(p_x1, t_x1)
        y1 = torch.min(p_y1, t_y1)
        x2 = torch.max(p_x2, t_x2)
        y2 = torch.max(p_y2, t_y2)

        cc = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
        p_xc = (p_x1 + p_x2) / 2
        p_yc = (p_y1 + p_y2) / 2
        t_xc = (t_x1 + t_x2) / 2
        t_yc = (t_y1 + t_y2) / 2
        rou = (p_xc - t_xc) * (p_xc - t_xc) + (t_yc - p_yc) * (t_yc - p_yc)
        diou = (iou - rou / (cc + self.eps)).clamp(-1).mean(dim=-1)

        # Each iou should be less than 1.
        bool_diou = diou <= 1
        assert bool_diou.all(), f"iou must <= 1, {diou}"

        diou_loss = 1 - diou
        min_diou_loss_id = torch.argmin(diou_loss, dim=-1)
        min_diou_loss = diou_loss[range(batch_size), min_diou_loss_id].mean()

        return min_diou_loss

    def min_ciou_loss(self, x_pred: torch.Tensor, x_true: torch.Tensor):
        """Ciou loss calculate function.

        NOTE[zhanbo01.li]: the input bbox is x1y1x2y2, the last dimension
        of x_pred and x_true is the coords of bbox:
        x[:, :, :, 0]: x coord of the top left point of bbox.
        x[:, :, :, 1]: y coord of the top left point of bbox.
        x[:, :, :, 2]: x coord of the bottom right point of bbox.
        x[:, :, :, 3]: y coord of the bottom right point of bbox.

        Args:
            x_pred (batch_size, K, step, dim): the model pred output.
            x_true (batch_size, K, step, dim): the groundtruth.

        Returns:
            min_ciou_loss: the mini-batch min_ciou loss over K trajs.
        """
        batch_size = x_pred.shape[0]
        # [batch_size, K, dec_step]
        p_x1, p_y1 = x_pred[:, :, :, 0], x_pred[:, :, :, 1]
        p_x2, p_y2 = x_pred[:, :, :, 2], x_pred[:, :, :, 3]
        t_x1, t_y1 = x_true[:, :, :, 0], x_true[:, :, :, 1]
        t_x2, t_y2 = x_true[:, :, :, 2], x_true[:, :, :, 3]

        x1 = torch.max(p_x1, t_x1)
        y1 = torch.max(p_y1, t_y1)
        x2 = torch.min(p_x2, t_x2)
        y2 = torch.min(p_y2, t_y2)

        w = (x2 - x1).clamp(0.0)
        h = (y2 - y1).clamp(0.0)

        inter_aera = w * h
        p_aera = (p_x2 - p_x1) * (p_y2 - p_y1)
        t_aera = (t_x2 - t_x1) * (t_y2 - t_y1)

        union_aera = p_aera + t_aera - inter_aera + self.eps

        # [batch_size, K]
        iou = inter_aera / union_aera
        # diou area
        x1 = torch.min(p_x1, t_x1)
        y1 = torch.min(p_y1, t_y1)
        x2 = torch.max(p_x2, t_x2)
        y2 = torch.max(p_y2, t_y2)

        cc = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
        p_xc = (p_x1 + p_x2) / 2
        p_yc = (p_y1 + p_y2) / 2
        t_xc = (t_x1 + t_x2) / 2
        t_yc = (t_y1 + t_y2) / 2
        rou = (p_xc - t_xc) * (p_xc - t_xc) + (t_yc - p_yc) * (t_yc - p_yc)

        # ciou
        w1 = p_x2 - p_x1
        h1 = p_y2 - p_y1
        w2 = t_x2 - t_x1
        h2 = t_y2 - t_y1
        with torch.no_grad():
            arctan = torch.atan(w2 / (h2 + self.eps)) - torch.atan(
                w1 / (h1 + self.eps)
            )
            v = (4 / (math.pi ** 2)) * torch.pow(
                (
                    torch.atan(w2 / (h2 + self.eps))
                    - torch.atan(w1 / (h1 + self.eps))
                ),
                2,
            )
            S = 1 - iou
            alpha = v / (S + v)
            w_temp = 2 * w1
        ar = (8 / (math.pi ** 2)) * arctan * ((w1 - w_temp) * h1)
        ciou = (
            (iou - rou / (cc + self.eps) - alpha * ar)
            .clamp(min=-1.0, max=1.0)
            .mean(dim=-1)
        )

        # Each iou should be less than 1.
        bool_ciou = ciou <= 1
        assert bool_ciou.all(), f"iou must <= 1, {ciou}"

        ciou_loss = 1 - ciou
        min_ciou_loss_id = torch.argmin(ciou_loss, dim=-1)
        min_ciou_loss = ciou_loss[range(batch_size), min_ciou_loss_id].mean()

        return min_ciou_loss

    def prob_loss(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        prob: torch.Tensor,
        loss_type: str = "l2",
    ):
        """Probability loss calculate function.

        Args:
            x_pred ([batch_size, K, dec_steps, dim]): the model pred output.
            x_true ([batch_size, K, dec_steps, dim]): the groundtruth.
            prob ([batch_size, 1, 1, K]): the predicted probabilities of
                tarjectories.
            loss_type: the prob loss type, it can be "l2" or "ce". If it is
                "l2", the prob loss is the l2 loss between the predicted
                probabilities and the generated probabilities, if it is "ce",
                the prob loss is the cross entropy loss between the predicted
                probabilities and the generated probabilities.

        Returns:
            prob_loss: the L2 loss between the generated GT probabilities
                and the predicted probabilities if loss_type is "l2". If
                loss_type is "ce", return the cross entropy loss between them.
        """
        assert loss_type in ["l2", "ce"], f"Not implemented type: {loss_type}"

        # 1. get the GT probabilities.
        # [batch_size, K]
        square_l2_loss = torch.square(x_pred - x_true).sum((2, 3))
        l2_loss = torch.sqrt(square_l2_loss + self.eps)
        exp_l2_loss = torch.exp(-l2_loss)
        sum_exp_l2_loss = torch.sum(exp_l2_loss, dim=-1).unsqueeze(-1)
        softmax_prob = exp_l2_loss / sum_exp_l2_loss

        # 2. get the predicted probabilities.
        # [batch_size, K]
        prob = prob.squeeze(1).squeeze(1)
        exp_prob = torch.exp(prob)
        sum_exp_prob = torch.sum(exp_prob, dim=-1).unsqueeze(-1)
        pred_softmax_prob = exp_prob / sum_exp_prob

        # 3. calculate the loss.
        if loss_type == "l2":
            square_prob_l2_loss = torch.square(
                softmax_prob - pred_softmax_prob
            ).sum(1)
            prob_loss = torch.sqrt(square_prob_l2_loss + self.eps).mean()
        else:
            batch_size = x_pred.shape[0]
            prob_loss = -torch.sum(softmax_prob * torch.log(pred_softmax_prob))
            prob_loss = prob_loss / batch_size

        return prob_loss

    def forward(self, output: dict):
        """Update loss with the latest prediction results.

        Args:
            output: A dict of model output which includes the predicted and
                ground-truth trajectories.The following keys of output will
                be used in calculating metrics:
                1. 'pred_traj' (Tensor, [batch_size, K, dec_step, dim]):
                   the predicted trajectories in x1x2y1y2 style.
                2. 'pred_cxcympb' (Tensor, [batch_size, K, dec_step, dim]):
                   the predicted trajectories in cxcympb style, mpb: middle
                   point of the bottom edge.
                3. 'target_traj' (Tensor, [batch_size, 1, dec_step, dim]):
                   the ground-truth trajectories in x1x2y1y2 style.
                4. 'target_cxcympb' (Tensor, [batch_size, 1, dec_step, dim]):
                   the ground-truth trajectories in cxcympb style.
                5. 'all_goal_trajs' (Tensor, [batch_size, 1, dec_step, dim]):
                    the goal trajectories in x1x2y1y2 style of SGE module
                    in SGNet.
                6. 'goal_cxcympb' (Tensor, [batch_size, 1, dec_step, dim]):
                    the goal trajectories in cxcympb style of SGE module
                    in SGNet.
                7. 'KLD' (Tensor, [1]):
                    the KLD divergence in CVAE module.
                8. 'probabilities' (FloatTensor, [batch_size, 1, 1, K])
                    the predicted probs of the K trajectories, this item
                    only exists when the K > 1.

        Returns:
            all_loss: a dict contains all the loss.
        """
        # 1. extract the model postprocess output.
        # mpb is the middle point of the bottom of a bbox.
        # [batch_size, K, dec_step, dim]
        pred_traj = output["pred_traj"]
        pred_cxcympb = output["pred_cxcympb"]
        k_value = pred_traj.shape[1]
        # [batch_size, 1, dec_step, dim]
        target_traj = output["target_traj"]
        target_cxcympb = output["target_cxcympb"]
        # [batch_size, K, dec_step, dim]
        rep_tar_traj = target_traj.repeat(1, k_value, 1, 1)
        rep_tar_cxcympb = target_cxcympb.repeat(1, k_value, 1, 1)
        # [batch_size, 1, dec_step, dim]
        all_goal_trajs = output["all_goal_trajs"]
        goal_cxcympb = output["goal_cxcympb"]
        KLD = output["KLD"]

        # 2. calculate the loss.
        # 2.1 KLD loss.
        KLD_loss = self.KLD_weight * KLD
        # 2.2 goal_traj loss.
        # 2.2.1 goal bbox loss
        goal_bbox_loss = self.min_L2_loss(all_goal_trajs, target_traj)
        goal_bbox_loss = weight_reduce_loss(self.bbox_weight, goal_bbox_loss)
        # 2.2.2 goal iou loss
        if self.iou_loss_type == "iou_loss":
            goal_iou_loss = self.min_iou_loss(all_goal_trajs, target_traj)
        elif self.iou_loss_type == "giou_loss":
            goal_iou_loss = self.min_giou_loss(all_goal_trajs, target_traj)
        elif self.iou_loss_type == "diou_loss":
            goal_iou_loss = self.min_diou_loss(all_goal_trajs, target_traj)
        elif self.iou_loss_type == "ciou_loss":
            goal_iou_loss = self.min_ciou_loss(all_goal_trajs, target_traj)
        goal_iou_loss = weight_reduce_loss(self.iou_weight, goal_iou_loss)
        # 2.2.3 goal mpb loss
        # goal_cxcympb[:, :, :, 2:4]: x and y coords of the mpb of pred goal.
        # target_cxcympb[:, :, :, 2:4]: x and y coords of the mpb of GT.
        goal_mpb = goal_cxcympb[:, :, :, 2:4]
        target_mpb = target_cxcympb[:, :, :, 2:4]
        goal_mpb_loss = self.min_L2_loss(goal_mpb, target_mpb)
        goal_mpb_loss = weight_reduce_loss(self.mpb_weight, goal_mpb_loss)
        # 2.2.4 goal weighted loss
        goal_loss = goal_bbox_loss + goal_iou_loss + goal_mpb_loss
        goal_loss = weight_reduce_loss(self.goal_weight, goal_loss)
        # 2.3 pred_traj loss.
        # 2.3.1 pred bbox loss.
        pred_bbox_loss = self.min_L2_loss(pred_traj, rep_tar_traj)
        pred_bbox_loss = weight_reduce_loss(self.bbox_weight, pred_bbox_loss)
        # 2.3.2 pred iou loss.
        if self.iou_loss_type == "iou_loss":
            pred_iou_loss = self.min_iou_loss(pred_traj, rep_tar_traj)
        elif self.iou_loss_type == "giou_loss":
            pred_iou_loss = self.min_giou_loss(pred_traj, rep_tar_traj)
        elif self.iou_loss_type == "diou_loss":
            pred_iou_loss = self.min_diou_loss(pred_traj, rep_tar_traj)
        elif self.iou_loss_type == "ciou_loss":
            pred_iou_loss = self.min_ciou_loss(pred_traj, rep_tar_traj)
        pred_iou_loss = weight_reduce_loss(self.iou_weight, pred_iou_loss)
        # 2.3.3 pred mpb loss.
        # pred_cxcympb[:, :, :, 2:4]: x and y coords of the mpb of pred traj.
        # rep_tar_cxcympb[:, :, :, 2:4]: x and y coords of the mpb of GT.
        rep_pred_mpb = pred_cxcympb[:, :, :, 2:4]
        rep_target_mpb = rep_tar_cxcympb[:, :, :, 2:4]
        pred_mpb_loss = self.min_L2_loss(rep_pred_mpb, rep_target_mpb)
        pred_mpb_loss = weight_reduce_loss(self.mpb_weight, pred_mpb_loss)
        # 2.3.4 pred weighted loss.
        pred_loss = pred_bbox_loss + pred_iou_loss + pred_mpb_loss
        pred_loss = weight_reduce_loss(self.pred_weight, pred_loss)
        # 2.4 total loss.
        total_loss = KLD_loss + goal_loss + pred_loss

        # 3.update the loss result dict.
        all_loss = {
            "total_loss": total_loss,
            "KLD_loss": KLD_loss,
            "goal_loss": goal_loss,
            "pred_loss": pred_loss,
            "goal_bbox_loss": goal_bbox_loss,
            "goal_iou_loss": goal_iou_loss,
            "goal_mpb_loss": goal_mpb_loss,
            "pred_bbox_loss": pred_bbox_loss,
            "pred_iou_loss": pred_iou_loss,
            "pred_mpb_loss": pred_mpb_loss,
        }

        # 4. add the prob_loss if k_value > 1.
        if "probabilities" in output:
            prob = output["probabilities"]
            prob_loss = self.prob_loss(pred_traj, rep_tar_traj, prob)
            prob_loss = weight_reduce_loss(self.prob_weight, prob_loss)

            total_loss = total_loss + prob_loss
            all_loss["total_loss"] = total_loss
            all_loss["prob_loss"] = total_loss

        return all_loss


@OBJECT_REGISTRY.register
class DenseTNTloss(torch.nn.Module):
    """DenseTNT loss.

    The DenseTNT loss consists of three parts: traj l1 loss,
    scores nll loss (for goal scores and road scores), road nll loss
    is None if stage one block is not used. For details,
    please visit/read/refer to:
    https://arxiv.org/abs/2108.09640
    """

    def __init__(
        self,
        l1_weight=1,
        goal_nll_weight=1,
        road_nll_weight=1,
    ):
        """Initialize method.

        Args:
            l1_weight: the weight of traj l1 loss.
            goal_nll_weight: the weight of goal nll loss.
            road_nll_weight: the weight of road nll loss.
        """
        super(DenseTNTloss, self).__init__()
        self.l1_weight = l1_weight
        self.goal_nll_weight = goal_nll_weight
        self.road_nll_weight = road_nll_weight

    def traj_l1_loss(self, pred_traj: torch.Tensor, target: torch.Tensor):
        """Smooth L1 loss calculate function.

        Args:
            pred_traj: (num_obs, 1, num_steps, 2)
            target: (num_obs, 1, num_steps, 2)
            Returns:
                l1_loss: the distance loss between
                    gt trajectories and pred trajectories.
        """
        k_value = pred_traj.shape[1]
        target = target.repeat(1, k_value, 1, 1)
        l1_loss = F.smooth_l1_loss(pred_traj, target)

        return l1_loss

    def scores_nll_loss(self, scores: torch.Tensor, labels: torch.Tensor):
        """NLL loss calculate function.

        Args:
            scores: (num_obs, 1, 1, num_goals)
            labels: (num_obs, 1, 1, 1)
            Returns:
                nll_loss: the classification loss between
                    gt label and pred scores.
        """
        scores = F.log_softmax(scores, dim=-1)
        scores = scores.squeeze(1).squeeze(1)
        labels = labels.squeeze(1).squeeze(1).squeeze(1)
        nll_loss = F.nll_loss(scores, labels)

        return nll_loss

    def forward(self, output):
        """Update loss with the latest prediction results.

        Args:
            output: A dict of model output which includes the predicted and
                ground-truth trajectories.The following keys of output will
                be used in calculating metrics:
                1.  'valid_masks' (torch.Tensor, [num_obs, num_steps]): \
                    the masks of future trajectories.
                2.  'predict_trajs' (torch.Tensor, [num_obs, 1, \
                    num_steps, 2]): \
                3.  'predict_goal_scores' (torch.Tensor, [num_obs, 1, 1, \
                    num_goals]): \
                    the predicted goal scores.
                4.  'predict_road_scores' (torch.Tensor, [num_obs, 1, 1, \
                    num_poly]): \
                    the predicted road scores.
                5.  'nearest_goal_idxs' (torch.Tensor, [num_obs,1,1,1]): \
                    the index of the goal nearest to end point.
                6.  'nearest_road_idxs' (torch.Tensor, [num_obs,1,1,1]): \
                    the index of the road nearest to end point.
                7.  'future_trajectories' (torch.Tensor, [num_obs, 1, \
                    num_steps, 2]): \
                    the ground truth trajectories.

        Returns:
            all_loss: a dict contains all the loss.
        """
        # regression gt
        gt_fut_traj = output["future_trajectories"]
        if "diff_fut_trajs" in output:  # 计算差分loss or 轨迹loss
            gt_fut_traj = output["diff_fut_trajs"]
        # (num_obs,traj_len)
        mask_traj = copy.deepcopy(output["valid_masks"])
        # 若某条轨迹的终点无效，则整条轨迹都不参与traj_loss计算
        mask_traj[output["valid_masks"][:, -1] == 0] = 0
        # (num_obs,1,traj_len,2)
        mask_traj = mask_traj.repeat(2, 1, 1, 1).permute(2, 1, 3, 0)
        # (num_obs,)
        # 若某条轨迹的终点无效，则整条轨迹都不参与score_loss计算
        mask_goals = output["valid_masks"][:, -1] == 1

        # predictions
        pred_traj1 = (
            output["predict_trajs"]
            if output["predict_trajs"] is not None
            else gt_fut_traj.clone()
        )
        predict_goal_scores = output["predict_goal_scores"]
        if "predict_road_scores" in output:
            predict_road_scores = output["predict_road_scores"]

        # gt
        nearest_goal_idxs = output["nearest_goal_idxs"].long()
        nearest_road_idxs = (
            output["nearest_road_idxs"].long()
            if "predict_road_scores" in output
            else output["nearest_road_idxs"]
        )

        # traj_l1_loss
        traj_l1_loss = self.traj_l1_loss(
            pred_traj1 * mask_traj, gt_fut_traj * mask_traj
        )

        # scores_nll_loss (for goals)
        if predict_goal_scores is None:
            scores_nll_loss = torch.tensor(0).cuda()
        else:
            scores_nll_loss = self.scores_nll_loss(
                predict_goal_scores[mask_goals], nearest_goal_idxs[mask_goals]
            )

        # TODO (zifan.li): road nll loss might be used in future.
        # scores_nll_loss (for roads)
        if "predict_road_scores" not in output:
            stage_one_nll_loss = torch.tensor(0).cuda()
        else:
            stage_one_nll_loss = self.scores_nll_loss(
                predict_road_scores[mask_goals], nearest_road_idxs[mask_goals]
            )

        # total_loss
        total_loss = traj_l1_loss * self.l1_weight
        total_loss += scores_nll_loss * self.goal_nll_weight
        total_loss += stage_one_nll_loss * self.road_nll_weight

        return {
            "total_loss": total_loss,
            "traj_l1_loss": traj_l1_loss,
            "scores_nll_loss": scores_nll_loss,
            "stage_one_nll_loss": stage_one_nll_loss,
        }


@OBJECT_REGISTRY.register
class DenseTNTScoreloss(torch.nn.Module):
    """DenseTNT loss (scores only)."""

    def __init__(
        self,
        goal_nll_weight=1,
    ):
        """Initialize method.

        Args:
            goal_nll_weight: the weight of goal nll loss.
        """
        super(DenseTNTScoreloss, self).__init__()
        self.goal_nll_weight = goal_nll_weight

    def scores_nll_loss(self, scores: torch.Tensor, labels: torch.Tensor):
        """NLL loss calculate function.

        Args:
            scores: (num_obs, 1, 1, num_goals)
            labels: (num_obs, 1, 1, 1)
            Returns:
                nll_loss: the classification loss between
                    gt label and pred scores.
        """
        scores = F.log_softmax(scores, dim=-1)
        scores = scores.squeeze(1).squeeze(1)
        labels = labels.squeeze(1).squeeze(1).squeeze(1)
        nll_loss = F.nll_loss(scores, labels)

        return nll_loss

    def forward(self, output):
        """Update loss with the latest prediction results.

        Args:
            output: A dict of model output which includes the predicted and
                ground-truth trajectories.The following keys of output will
                be used in calculating metrics:
                1.  'valid_masks' (torch.Tensor, [num_obs, num_steps]): \
                    the masks of future trajectories.
                2.  'real_scores' (torch.Tensor, [num_obs, 1, 1, \
                    num_goals]): \
                    the predicted goal scores.
                3.  'select_set_head_pts' (torch.Tensor, [num_obs, 2, 1, \
                    k]): \
                    the top-k selected goals.
                4.  'end_points' (torch.Tensor, [num_obs, 2, 1, 1]): \
                    the end points.

        Returns:
            all_loss: a dict contains all the loss.
        """
        # (num_obs,traj_len)
        mask_traj = copy.deepcopy(output["valid_masks"])
        # 若某条轨迹的终点无效，则整条轨迹都不参与traj_loss计算
        mask_traj[output["valid_masks"][:, -1] == 0] = 0
        # (num_obs,1,traj_len,2)
        mask_traj = mask_traj.repeat(2, 1, 1, 1).permute(2, 1, 3, 0)
        # (num_obs,)
        # 若某条轨迹的终点无效，则整条轨迹都不参与score_loss计算
        mask_goals = output["valid_masks"][:, -1] == 1

        predict_goal_scores = output["real_scores"]
        set_predict_goals = output["select_set_head_pts"]

        end_points = output["end_points"]
        num_traj = set_predict_goals.shape[-1]
        distance = set_predict_goals - end_points.repeat(1, 1, 1, num_traj)
        distance = torch.sum(torch.square(distance), dim=1, keepdim=True)
        nearest_goal_idxs = torch.argmin(distance, dim=-1, keepdim=True)

        # scores_nll_loss (for goals)
        if predict_goal_scores is None:
            scores_nll_loss = torch.tensor(0).cuda()
        else:
            scores_nll_loss = self.scores_nll_loss(
                predict_goal_scores[mask_goals], nearest_goal_idxs[mask_goals]
            )

        return {
            "total_loss": scores_nll_loss * self.goal_nll_weight,
            "set_scores_nll_loss": scores_nll_loss,
        }


@OBJECT_REGISTRY.register
class SoftDenseTNTloss(torch.nn.Module):
    """DenseTNT loss of soft version.

    The DenseTNT loss of soft version consists of three parts: traj l1 loss,
    scores nll loss (for goal scores and road scores), road nll loss
    is None if stage one block is not used. For details,
    please visit/read/refer to:
    https://arxiv.org/abs/2108.09640
    """

    def __init__(
        self,
        l1_weight: float = 1,
        goal_nll_weight: float = 1,
        road_nll_weight: float = 1,
        is_soft_cls: bool = False,
    ):
        """Initialize method.

        Args:
            l1_weight: the weight of traj l1 loss.
            goal_nll_weight: the weight of goal nll loss.
            road_nll_weight: the weight of road nll loss.
            is_soft_cls: if use soft loss, if it is flase, Loss will
                became DenseTNTloss.
        """
        super(SoftDenseTNTloss, self).__init__()
        self.l1_weight = l1_weight
        self.goal_nll_weight = goal_nll_weight
        self.road_nll_weight = road_nll_weight
        self.is_soft_cls = is_soft_cls

    def traj_l1_loss(self, pred_traj: torch.Tensor, target: torch.Tensor):
        """Smooth L1 loss calculate function.

        Args:
            pred_traj: (num_obs, 1, num_steps, 2) the predicted traj
            target: (num_obs, 1, num_steps, 2) the GT trajectory

        Returns:
            l1_loss: the distance loss between
                gt trajectories and pred trajectories.
        """
        l1_loss = F.smooth_l1_loss(pred_traj, target)

        return l1_loss

    def scores_nll_loss(self, scores: torch.Tensor, labels: torch.Tensor):
        """NLL loss calculate function.

        Args:
            scores: (num_obs, 1, 1, num_goals) goal scores output by goal head
            labels: (num_obs, 1, 1, 1) the label to indicated the top1 goal

        Returns:
            nll_loss: the classification loss between
                gt label and pred scores.
        """
        scores = F.log_softmax(scores, dim=-1)
        scores = scores.squeeze(1).squeeze(1)
        labels = labels.squeeze(1).squeeze(1).squeeze(1)
        nll_loss = F.nll_loss(scores, labels)

        return nll_loss

    def l2_distance(
        self, goal_set: torch.Tensor, target_gt_point: torch.Tensor
    ):
        """Calculate L2 distance.

        Args:
            goal_set: [num_obs, 2, 1, num_goals]
            target_gt_point: [num_obs, 2, 1, 1]

        Returns:
            dist: [num_obs, 1, 1, num_goals],L2 distance of every goals.
        """
        target_gt_point_broadcasted = target_gt_point.expand_as(goal_set)

        dist = torch.sqrt(
            torch.sum(
                (goal_set - target_gt_point_broadcasted).permute(0, 2, 3, 1)
                ** 2,
                dim=-1,
            )
        )

        return dist

    def filter_with_zeros(
        self, goal_without_mask: torch.Tensor, mask: torch.Tensor
    ):
        """Filter the mask goal.

        Args:
            goal_without_mask: [num_obs, 1, 1, num_goals] goal before mask
            mask: [num_obs, 1, 1, num_goals] mask Boolean

        Returns:
            filtered_tensor: [num_obs, 1, 1, num_goals] goal after mask
        """
        selected_tensor = goal_without_mask[mask]

        filtered_tensor = torch.zeros_like(goal_without_mask)

        filtered_tensor[mask] = selected_tensor

        return filtered_tensor

    def forward(self, output):
        """Update loss with the latest prediction results.

        Args:
            output: A dict of model output which includes the predicted and
                ground-truth trajectories.The following keys of output will
                be used in calculating metrics:
                1.  'valid_masks' (torch.Tensor, [num_obs, num_steps]): \
                    the masks of future trajectories.
                2.  'predict_trajs' (torch.Tensor, [num_obs, 1, \
                    num_steps, 2]): \
                3.  'predict_goal_scores' (torch.Tensor, [num_obs, 1, 1, \
                    num_goals]): \
                    the predicted goal scores.
                4.  'predict_road_scores' (torch.Tensor, [num_obs, 1, 1, \
                    num_poly]): \
                    the predicted road scores.
                5.  'nearest_goal_idxs' (torch.Tensor, [num_obs,1,1,1]): \
                    the index of the goal nearest to end point.
                6.  'nearest_road_idxs' (torch.Tensor, [num_obs,1,1,1]): \
                    the index of the road nearest to end point.
                7.  'future_trajectories' (torch.Tensor, [num_obs, 1, \
                    num_steps, 2]): \
                    the ground truth trajectories.

        Returns:
            all_loss: a dict contains all the loss.
        """
        # regression gt
        gt_fut_traj = output["future_trajectories"]
        if (
            "diff_fut_trajs" in output
        ):  # Calculate differential loss or trajectory loss
            gt_fut_traj = output["diff_fut_trajs"]
        # (num_obs,traj_len)
        mask_traj = copy.deepcopy(output["valid_masks"])
        # If the end point of a trajectory is invalid, the entire trajectory
        # will not participate in the traj_loss calculation
        mask_traj[output["valid_masks"][:, -1] == 0] = 0
        # (num_obs,1,traj_len,2)
        mask_traj = mask_traj.repeat(2, 1, 1, 1).permute(2, 1, 3, 0)
        # (num_obs,)
        # If the end point of a track is invalid, the whole track will not
        # participate in the score_loss calculation
        mask_goals = output["valid_masks"][:, -1] == 1

        # predictions
        pred_traj1 = (
            output["predict_trajs"]
            if output["predict_trajs"] is not None
            else gt_fut_traj.clone()
        )
        predict_goal_scores = output["predict_goal_scores"]
        if "predict_road_scores" in output:
            predict_road_scores = output["predict_road_scores"]

        # gt
        nearest_goal_idxs = output["nearest_goal_idxs"].long()
        nearest_road_idxs = (
            output["nearest_road_idxs"].long()
            if "predict_road_scores" in output
            else output["nearest_road_idxs"]
        )

        # traj_l1_loss
        traj_l1_loss = self.traj_l1_loss(
            pred_traj1 * mask_traj, gt_fut_traj * mask_traj
        )

        # scores_nll_loss (for goals)
        if self.is_soft_cls:
            if predict_goal_scores is None:
                scores_nll_loss = torch.tensor(0).cuda()
            else:
                end_points = output["end_points"]
                goal_coords = output["goal_coords"]
                goal_masks = output["goal_masks"]

                l2 = self.l2_distance(goal_coords, end_points)
                l2_softmax = torch.exp(-l2)
                bool_tensor = goal_masks == 1
                filtered_tensor = self.filter_with_zeros(
                    l2_softmax.unsqueeze(1), bool_tensor
                )
                l2_prob_goal = filtered_tensor.squeeze() / torch.sum(
                    filtered_tensor, dim=-1
                ).squeeze().unsqueeze(1).repeat(1, 2048)
                goal_scores_log_softmax = F.log_softmax(
                    predict_goal_scores, dim=-1
                ).squeeze()

                # Use cross entropy: -sum (y_i * log P_i)
                #   y_i: L2 probabilities; P_i: anchor probabilities.
                scores_nll_loss = -torch.sum(
                    l2_prob_goal[mask_goals]
                    * goal_scores_log_softmax[mask_goals]
                )
                scores_nll_loss /= sum(mask_goals)
        else:
            if predict_goal_scores is None:
                scores_nll_loss = torch.tensor(0).cuda()
            else:
                scores_nll_loss = self.scores_nll_loss(
                    predict_goal_scores[mask_goals],
                    nearest_goal_idxs[mask_goals],
                )

        # TODO (dukai.dong): road nll loss might be used in future.
        # scores_nll_loss (for roads)
        if "predict_road_scores" not in output:
            road_nll_loss = torch.tensor(0).cuda()
        else:
            road_nll_loss = self.scores_nll_loss(
                predict_road_scores[mask_goals], nearest_road_idxs[mask_goals]
            )

        # total_loss
        total_loss = traj_l1_loss * self.l1_weight
        total_loss += scores_nll_loss * self.goal_nll_weight
        total_loss += road_nll_loss * self.road_nll_weight

        return {
            "total_loss": total_loss,
            "traj_l1_loss": traj_l1_loss,
            "scores_nll_loss": scores_nll_loss,
            "road_nll_loss": road_nll_loss,
        }


@OBJECT_REGISTRY.register
class DenseTNTStageTwoLoss(torch.nn.Module):
    """Loss of the two stage DenseTNT.

    DenseTNT二阶段loss，包括head_loss和set_loss两部分，DenseTNT
    二阶段设置了一个多head的目标点预测器（set predictor)，在输出
    预测点的同时也会给head打分，head_loss是这个head分数的loss。
    set_loss描述了set predictor输出goal本身的有效性。
    """

    def __init__(
        self,
        head_loss_weight: float = 1,
        set_loss_weight: float = 1,
        k_points: int = 6,
        pseudo_label_lr: float = 1,
        goal_coords_scale: float = 1,
        gen_pseudo_label_by: str = "gt",
    ):
        """Initialize method.

        Args:
            head_loss_weight: head loss权重
            set_loss_weight: set loss权重
            k_points: set predictor同时输出的goal的总数
            pseudo_label_lr: 伪标签（用于生成set_loss）优化算法学习率
            goal_coords_scale: 输入前会对goal点坐标进行缩放，这里是缩放比例
            gen_pseudo_label_by: 生成伪标签的方式，目前支持三种
                ["pred", "gt", "optim"]。"pred"是基于预测的goal点作为起点
                通过优化算法扰动优化后得到伪标签，与论文中的说法相近，但如果是
                从零开始训练，预测的goal点本身的不确定性会导致优化后的点并非
                理想的goal点，进而导致模型向错误的方向学习。"gt"是基于真值点
                通过优化算法扰动优化得到伪标签，这种可以保证输出的伪标签的有效性，
                但由于是同一个点开始优化，会导致多模态被消除。"optim"是基于goal
                score判定，随机选择较好的起点开始优化，会兼顾多模态和数值与gt的
                接近程度，但是强依赖与goal score质量，意味着一阶段训练要足够收敛。
        """
        super(DenseTNTStageTwoLoss, self).__init__()
        self.head_loss_weight = head_loss_weight
        self.set_loss_weight = set_loss_weight
        self.k_points = k_points
        self.pseudo_label_lr = pseudo_label_lr
        self.goal_coords_scale = goal_coords_scale
        self.gen_pseudo_label_by = gen_pseudo_label_by

    def forward(self, output):
        """Update loss with the latest prediction results."""

        all_set_head_scores = output["all_set_head_scores"]
        all_set_head_pts = output["all_set_head_pts"]
        goal_coords = output["goal_coords"]
        predict_goal_scores = output["predict_goal_scores"]
        end_points = output["end_points"]

        head_labels, pseudo_labels = self.gen_stage_two_gt(
            goal_coords,
            predict_goal_scores,
            all_set_head_scores,
            all_set_head_pts,
            end_points,
        )
        head_labels = torch.tensor(head_labels).cuda()
        pseudo_labels = torch.tensor(pseudo_labels).cuda()

        set_predict = []
        for i, j in enumerate(head_labels):
            set_predict.append(all_set_head_pts[j][i : (i + 1), :, :, :])
        set_predict = torch.cat(set_predict, dim=0)

        all_set_head_scores = F.log_softmax(all_set_head_scores, dim=1)
        head_loss = F.nll_loss(
            all_set_head_scores.squeeze(-1).squeeze(-1), head_labels
        )
        set_loss = (
            F.l1_loss(set_predict, pseudo_labels) / self.goal_coords_scale
        )

        total_loss = (
            head_loss * self.head_loss_weight + set_loss * self.set_loss_weight
        )
        return {
            "total_loss": total_loss,
            "head_loss": head_loss,
            "set_loss": set_loss,
            "pseudo_labels": pseudo_labels,
        }

    def gen_stage_two_gt(
        self,
        goal_coords: torch.Tensor,
        predict_goal_scores: torch.Tensor,
        all_set_head_scores: torch.Tensor,
        all_set_head_pts: List,
        end_points: torch.Tensor,
    ):
        """Generate the ground truth of the stage two.

        Args:
            goal_coords ([num_obs, 2, 1, num_sample_pts]): 用于产生伪标签
                的所有采样点
            predict_goal_scores ([num_obs, 1, 1, num_sample_pts]): 所有采样
                点的score值，和goal_coords共同构成一个描述全域作为goal的可行性
                的heatmap
            all_set_head_scores ([num_obs, num_head, 1, 1]): 所有set predictor
                head输出的对应head的打分
            all_set_head_pts (List of [num_obs, 2, 1, k_points]): 所有set
                predictor head的输出结果
            end_points ([num_obs, 2, 1, 1]): 各障碍物的gt终点

        Returns:
            argmin_costs ([num_heads]): cost最小的head的index
            pseudo_labels ([num_obs, 2, 1, k_points]): 伪标签
        """
        goals_2D = goal_coords.detach().cpu().numpy()
        goals_score = predict_goal_scores.detach().cpu().numpy()
        all_select_points = np.array(
            [i.detach().cpu().numpy() for i in all_set_head_pts]
        )
        all_select_scores = all_set_head_scores.detach().cpu().numpy()
        num_obs, num_heads = all_select_scores.shape[0:2]
        all_end_points = end_points.detach().cpu().numpy()

        # 坐标rescale到米为单位
        goals_2D /= self.goal_coords_scale
        all_end_points /= self.goal_coords_scale
        all_select_points /= self.goal_coords_scale

        # 概率归一化
        goals_score = np.exp(goals_score[:, 0, 0, :])
        goals_score /= np.reshape(np.sum(goals_score, -1), (-1, 1))

        # 获取优化起点
        costs = np.zeros([num_obs, num_heads])
        for j in range(num_heads):
            selected_points = all_select_points[j]
            for i in range(num_obs):
                tmp_pred_goals = selected_points[i, :, 0, :].transpose(1, 0)
                tmp_sample_goals = goals_2D[i, :, 0, :].transpose(1, 0)
                tmp_pred_goal_scores = goals_score[i]
                costs[i, j] = utils_cython.set_predict_get_value(
                    tmp_sample_goals,
                    tmp_pred_goal_scores,
                    tmp_pred_goals,
                    kwargs={"set_predict-MRratio": 0},
                )
        argmin_costs = np.argmin(costs, axis=1)

        # 获取伪标签
        pseudo_labels = np.zeros([num_obs, self.k_points, 2])
        for i in range(num_obs):
            tmp_sample_goals = goals_2D[i, :, 0, :].transpose(1, 0)
            tmp_pred_goal_scores = goals_score[i]
            if self.gen_pseudo_label_by == "gt":
                gt_end_points = np.repeat(
                    all_end_points[i : i + 1, :, 0, 0], self.k_points, axis=0
                )
                _, dynamic_label = utils_cython.set_predict_next_step(
                    tmp_sample_goals,
                    tmp_pred_goal_scores,
                    gt_end_points,
                    lr=self.pseudo_label_lr,
                )
            elif self.gen_pseudo_label_by == "pred":
                min_cost_idx = argmin_costs[i]
                tmp_pred_goals = all_select_points[min_cost_idx][
                    i, :, 0, :
                ].transpose(1, 0)
                _, dynamic_label = utils_cython.set_predict_next_step(
                    tmp_sample_goals,
                    tmp_pred_goal_scores,
                    tmp_pred_goals,
                    lr=self.pseudo_label_lr,
                    kwargs={"set_predict-MRratio": 0},
                )
            elif self.gen_pseudo_label_by == "optim":
                _, dynamic_label, _ = utils_cython.get_optimal_targets(
                    tmp_sample_goals,
                    tmp_pred_goal_scores,
                    "xxx",  # 无意义的参数
                    "MRminFDE",
                    0.1,
                    kwargs={"cnt_sample": 9, "MRratio": 0, "num_step": 1000},
                )
            pseudo_labels[i, :, :] = dynamic_label * self.goal_coords_scale
        pseudo_labels = pseudo_labels.transpose(0, 2, 1)[:, :, None, :]
        return argmin_costs, pseudo_labels


@OBJECT_REGISTRY.register
class DenseTNTPlusloss(torch.nn.Module):
    """DenseTNT Plus loss.

    The plus version of DenseTNT removes the goal sampling function and predict
    the goal site by model directly. And the loss consists of three parts:
    traj l1 loss, scores nll loss (for classification goal scores), and
    endpts l1 loss(for the regress loss of goal).
    """

    def __init__(
        self,
        l1_weight=1,
        goal_nll_weight=1,
        endpts_l1_weight=1,
    ):
        """Initialize method.

        Args:
            l1_weight: the weight of traj l1 loss.
            goal_nll_weight: the weight of goal nll loss.
            endpts_l1_weight: the weight of end points l1 loss.
        """
        super(DenseTNTPlusloss, self).__init__()
        self.l1_weight = l1_weight
        self.goal_nll_weight = goal_nll_weight
        self.endpts_l1_weight = endpts_l1_weight

    def forward(self, output):
        """Update loss with the latest prediction results.

        Args:
            output: A dict of model output which includes the predicted and
                ground-truth trajectories.The following keys of output will
                be used in calculating metrics:
                1.  'valid_masks' (torch.Tensor, [num_obs, num_steps]): \
                    the masks of future trajectories.
                2.  'predict_trajs' (torch.Tensor, [num_obs, num_modal, \
                    num_steps, 2]): \
                3.  'real_scores' (torch.Tensor, [num_obs, num_modal, \
                    1, 1]): \
                4.  'future_trajectories' (torch.Tensor, [num_obs, 1, \
                    num_steps, 2]): \
                    the ground truth trajectories.

        Returns:
            all_loss: a dict contains all the loss.
        """
        # regression gt
        # (num_obs,1,traj_len,2)
        gt_fut_traj = output["future_trajectories"]
        if "diff_fut_trajs" in output:  # 计算差分loss or 轨迹loss
            gt_fut_traj = output["diff_fut_trajs"]
        # (num_obs,traj_len)
        mask_traj = copy.deepcopy(output["valid_masks"])
        # 若某条轨迹的终点无效，则整条轨迹都不参与traj_loss计算
        mask_traj[output["valid_masks"][:, -1] == 0] = 0
        # (num_obs,1,traj_len,2)
        mask_traj = mask_traj.repeat(2, 1, 1, 1).permute(2, 1, 3, 0)
        # (num_obs,)
        # 若某条轨迹的终点无效，则整条轨迹都不参与score_loss计算
        mask_goals = output["valid_masks"][:, -1] == 1
        mask_goals = mask_goals.reshape(-1, 1)

        # 预测输出
        pred_traj = output["predict_trajs"]  # (num_obs,5,traj_len,2)
        pred_scores = output["real_scores"][
            :, :, 0, 0
        ]  # (num_obs,5,1,1) --> (num_obs,5)

        # 根据离gt终点最近计算match label
        norms = torch.norm(
            (gt_fut_traj[:, :, -1, :] - pred_traj[:, :, -1, :])
            * mask_traj[:, :, -1, :],
            dim=-1,
        )
        match_label = torch.argmin(norms, dim=-1)

        # 分类loss
        cls_loss = F.cross_entropy(pred_scores, match_label, reduction="none")
        cls_loss = torch.mean(cls_loss)

        # 轨迹回归loss
        gt_fut_traj = gt_fut_traj[:, 0, :, :]  # (num_obs,traj_len,2)
        mt_trajs_idx = [torch.arange(len(match_label)), match_label]
        pred_traj = pred_traj[mt_trajs_idx]  # (num_obs,traj_len,2)
        traj_reg_loss = F.smooth_l1_loss(
            pred_traj * mask_traj[:, 0, :, :],
            gt_fut_traj * mask_traj[:, 0, :, :],
            reduction="mean",
        )

        # 终点回归loss
        endpts_loss = F.smooth_l1_loss(
            pred_traj[:, -1, :] * mask_goals,
            gt_fut_traj[:, -1, :] * mask_goals,
            reduction="mean",
        )

        # total_loss
        total_loss = traj_reg_loss * self.l1_weight
        total_loss += cls_loss * self.goal_nll_weight
        total_loss += endpts_loss * self.endpts_l1_weight

        return {
            "total_loss": total_loss,
            "traj_l1_loss": traj_reg_loss,
            "scores_nll_loss": cls_loss,
            "endpts_l1_loss": endpts_loss,
        }
