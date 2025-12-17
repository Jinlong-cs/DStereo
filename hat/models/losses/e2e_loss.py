# Copyright (c) Horizon Robotics. All rights reserved.

import logging
from collections import defaultdict
from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from hat.core.box_utils import box_center_to_corner
from hat.models.losses.focal_loss import FocalLossV2
from hat.models.losses.real3d_losses import hm_l1_wing_loss
from hat.models.task_modules.e2e_dynamic.e2e_box_util import (
    complete_box_iou,
    distance_box_iou,
    generalized_box_iou,
)
from hat.models.task_modules.e2e_dynamic.e2e_dynamic_util import (  # noqa
    get_coef_of_psc,
)
from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)

__all__ = ["E2EDynamicLoss", "E2EDynamicTrackLoss", "E2EDynamicVeloLoss"]

EPS = 1e-6


@OBJECT_REGISTRY.register
class E2EDynamicLoss(nn.Module):
    """calculate all loss of e2e-dynamic task.

    Args:
        map_label: map of categorys' name and labels.(e.g. {"car": 0,
            "pedestrain": 2}).
        loss_weights: Global loss weight for each sub-loss.(e.g. loss_ce:1,
            loss_bbox:1...)
        iou_loss_type: iou type to calculate box loss. Should be one of
            ["giou", "ciou", "diou"]
        calc_velocity_loss: if true, calculate velocity loss.
        calc_trajectory_loss: if true, calculate trajectory loss.
        alpha: focal loss param.
        gamma: focal loss param.
        beta: smooth_l1 loss param.
        use_psc_rot: whether use psc to encode the heading angle.
        N_steps_PSC_rot: nums of steps to decode the heading angle by PSC.
            Refer to https://horizonrobotics.feishu.cn/docx/EwYEdysIBodB69xn6Mdc7HkSnif. # noqa
        rot_mod_threshold: threshold to filter decoded phase of the heading angle.
    """

    def __init__(
        self,
        map_label: dict,
        loss_weights: dict,
        iou_loss_type: str = "ciou",
        calc_velocity_loss: bool = False,
        calc_trajectory_loss: bool = False,
        alpha: float = 0.25,
        gamma: float = 2,
        beta: float = 0.01,
        use_psc_rot: bool = False,
        N_steps_PSC_rot: int = None,
        rot_mod_threshold: float = 0.0001,
    ):
        super().__init__()
        assert iou_loss_type in [
            "giou",
            "ciou",
            "diou",
        ], "iou type not support!"

        self.trackLoss = E2EDynamicTrackLoss(
            map_label=map_label,
            loss_weights=loss_weights,
            iou_loss_type=iou_loss_type,
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            use_psc_rot=use_psc_rot,
        )
        self.veloLoss = (
            E2EDynamicVeloLoss(
                loss_weights=loss_weights,
                alpha=alpha,
                gamma=gamma,
                beta=beta,
                use_psc_rot=use_psc_rot,
                N_steps_PSC_rot=N_steps_PSC_rot,
                rot_mod_threshold=rot_mod_threshold,
            )
            if (calc_velocity_loss)
            else None
        )
        self.trajLoss = (
            E2EDynamicTrajLoss(
                loss_weights=loss_weights,
            )
            if (calc_trajectory_loss)
            else None
        )

    @autocast(enabled=False)
    def forward(self, pred: List, target=None):
        # target is a placeholder, for compatiblity of input number of other
        # tasks.Such as crosspoint_loss, ...
        output_loss_items = {}
        batch_size = len(pred)
        for clip_pred in pred:
            loss_items = {}
            assert "output_for_losses" in clip_pred
            model_out = clip_pred.pop("output_for_losses")
            aux_outputs_for_loss = model_out.pop("aux_outputs_for_loss", {})
            if self.veloLoss is not None:
                loss_items.update(
                    self.veloLoss(model_out, "pre_matched_indices")
                )
            if self.trajLoss is not None:
                loss_items.update(
                    self.trajLoss(
                        model_out,
                        "pre_matched_indices",
                        num_sample_key="num_samples",
                    )
                )

            model_out.update(aux_outputs_for_loss)
            loss_items.update(
                self.trackLoss(model_out, num_sample_key="num_samples")
            )
            for key, value in loss_items.items():
                if key not in output_loss_items:
                    output_loss_items[key] = torch.tensor(0.0).to(value)
                output_loss_items[key] += value / batch_size

        return output_loss_items


class E2EDynamicTemplateLoss(nn.Module):
    """Forward loss of e2e-dynamic task.

    This is a template, only for code simplification. all loss share the same
    forward function.
    """

    def get_num_boxes(self, num_samples, device):
        num_boxes = torch.as_tensor(
            num_samples, dtype=torch.float, device=device
        )
        if (
            dist.is_available()
            and dist.is_initialized()
            and device.type != "cpu"
        ):
            torch.distributed.all_reduce(num_boxes)
            word_size = dist.get_world_size()
        else:
            word_size = 1
        num_boxes = torch.clamp(num_boxes / word_size, min=1).item()
        return num_boxes

    def forward(
        self,
        pred: dict,
        match_indice_key: str = "match_indices",
        num_sample_key: str = None,
    ):
        """Calculate e2e losses between pred and target items.

        Args:
            pred: List of each frame data used for loss calculation. It at
            least have `pred_output`, `gt_instances` and `match_indices` in
            each sample dict.

        """
        output_loss = defaultdict(list)
        num_samples = 0
        for frame_pred in pred.values():
            if num_sample_key == "num_samples":
                if "num_samples" in frame_pred:
                    num_samples += frame_pred["num_samples"]
                else:
                    # auxilary outputs don't have num_samples key
                    num_samples += 0
            elif num_sample_key == "prev_num_samples":
                src_idx, _ = frame_pred[match_indice_key]
                num_samples += src_idx.numel()
            else:
                # TODO: only use for velocity loss, remove it later.
                num_samples = 1
            for loss in self.loss_map:
                loss_out = self.loss_map[loss](
                    frame_pred["pred_output"],
                    frame_pred["gt_instances"],
                    frame_pred[match_indice_key],
                    frame_pred["matched_ious"]
                    if ("matched_ious") in frame_pred
                    else None,
                )
                # NOTE: match_indices construct from
                # [new_matched_indices, prev_matched_indices],
                # There is no -1 in both indices.
                for key, value in loss_out.items():
                    output_loss[key].append(value)
        if num_sample_key == "num_samples":
            num_samples = self.get_num_boxes(
                num_samples, frame_pred[match_indice_key][0].device
            )
        avg_output_loss = {}
        for k, v in output_loss.items():
            if k not in self.loss_weights:
                logger.info(f"{k} not in loss_weight. It is set as 1.0")
                self.loss_weights[k] = 1.0
            avg_output_loss[k] = (
                torch.stack(v).sum()
                * self.loss_weights[k]
                / max(num_samples, 1)
            )
        return avg_output_loss


class E2EDynamicTrackLoss(E2EDynamicTemplateLoss):
    """calculate track or detect loss of e2e-dynamic task.

    Args:
        map_label: map of categorys' name and labels.(e.g. {"car": 0,
            "pedestrain": 2}).
        loss_weights: Global loss weight for each sub-loss.(e.g. loss_ce:1,
            loss_bbox:1...)
        iou_loss_type: iou type to calculate box loss. Should be one of
            ["giou", "ciou", "diou"]
        alpha: focal loss param.
        gamma: focal loss param.
        beta: smooth_l1 loss param.
        use_psc_rot: whether use psc for rotate_yaw.
    """

    def __init__(
        self,
        map_label: dict,
        loss_weights: dict,
        iou_loss_type: str = "ciou",
        alpha: float = 0.25,
        gamma: float = 2,
        beta: float = 0.01,
        use_psc_rot: bool = False,
    ):
        super().__init__()
        self.map_label = map_label
        self.loss_weights = loss_weights
        assert iou_loss_type in [
            "giou",
            "ciou",
            "diou",
        ], "iou type not support!"
        self.iou_loss_type = iou_loss_type
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.loss_map = {
            "label": self.get_cls_loss,
            "box": self.get_box_loss,
            "zh": self.get_zheight_loss,
            "yaw": self.get_yaw_loss,
            "track_score": self.get_track_score_loss,
        }
        self.focal_loss = FocalLossV2(
            alpha=alpha,
            gamma=gamma,
            from_logits=False,
            reduction="none",
        )

        self.use_psc_rot = use_psc_rot

    def get_cls_loss(
        self,
        outputs: dict,
        gt_instances: dict,
        indices: tuple,
        matched_ious: torch.Tensor = None,
    ):
        """Classification loss, default using sigmoid focal loss.

        Args:
            outputs: pred outputs, must contains `pred_logits`.
            gt_instances: dict of gt instance.
            indices: src and gt match indices.
            matched_ious: ious between all queries and gt to filter the
                trk query by iou.

        """
        assert (
            "pred_logits" in outputs
        ), "`pred_logits` should be provided when get the lable loss"
        assert "matched_gt_idxes" in outputs
        num_classes = outputs["pred_logits"].shape[-1]
        pred_logits = outputs["pred_logits"]
        match_gt_idxes = outputs["matched_gt_idxes"]
        mask = match_gt_idxes > -2
        # predefine all the classes is background
        target_classes = torch.full(
            pred_logits.shape[:1],
            num_classes,
            dtype=torch.int64,
            device=pred_logits.device,
        )
        src_idx, gt_idx = indices
        if src_idx.numel() > 0:
            target_classes_o = gt_instances["labels"][gt_idx]
            target_classes[src_idx] = target_classes_o
            if matched_ious is not None:
                target_classes[matched_ious < 1e-9] = num_classes

        # using sigmoid focal loss and no loss for the background.
        gt_labels_target = F.one_hot(
            target_classes, num_classes=num_classes + 1
        )[..., :-1]
        gt_labels_target = gt_labels_target.to(pred_logits)
        loss_ce = self.focal_loss(
            pred_logits[mask],
            gt_labels_target[mask],
        )
        loss = {"loss_ce": loss_ce.sum()}
        return loss

    def get_track_score_loss(
        self,
        outputs: dict,
        gt_instances: dict,
        indices: tuple,
        matched_ious: torch.Tensor = None,
    ):
        """Compute focal_loss of the front and the background.

        Args:
            outputs: pred outputs, must contains `pred_track_scores`.
            gt_instances: dict of gt instance.
            indices: src and gt match indices.
            matched_ious: ious between all queries and gt to filter the
                trk query by iou.

        """
        # only last output layer calculate track score loss.
        if "pred_track_scores" not in outputs:
            return {
                "trackloss_ce": torch.tensor(0.0).to(
                    outputs["pred_logits"].device
                )
            }
        src_idx, _ = indices
        num_tracks = outputs["pred_track_scores"].shape[0]
        pred_logits = outputs["pred_track_scores"].reshape(num_tracks)
        target_classes = torch.zeros((num_tracks,)).to(pred_logits.device)
        if src_idx.numel() > 0:
            target_classes[src_idx] += 1
        loss_ce = self.focal_loss(
            pred_logits.squeeze(),
            target_classes.squeeze(),
        )
        loss = {"trackloss_ce": loss_ce.sum()}
        return loss

    def get_box_loss(
        self,
        outputs: dict,
        gt_instances: dict,
        indices: tuple,
        matched_ious: torch.Tensor = None,
    ):
        """Compute the losses related to the bounding boxes.

        the L1 regression loss and the GIoU loss targets dicts must contain
        the key "boxes" containing a tensor of dim [nb_target_boxes, 4].
        The target boxes are expected in format (center_x, center_y, h, w)
        normalized by the image size.

        Args:
            outputs: pred outputs, must contains `pred_boxes`.
            gt_instances: dict of gt instance.
            indices: src and gt match indices.
            matched_ious: ious between all queries and gt to filter the
                trk query by iou.

        """
        assert (
            "pred_boxes" in outputs
        ), "`pred_boxes` should be provided when get xy loss."
        src_idx, gt_idx = indices
        if src_idx.numel() == 0:
            # there are no matched indices, return loss to 0.
            losses = {
                "loss_xy": torch.tensor(0.0).to(src_idx),
                "loss_wh": torch.tensor(0.0).to(src_idx),
                f"loss_{self.iou_loss_type}": torch.tensor(0.0).to(src_idx),
            }
            return losses

        src_boxes = outputs["pred_boxes"][src_idx]
        src_yaws = outputs["pred_yaws"][src_idx]

        target_boxes = gt_instances["boxes"][gt_idx]
        target_yaws = gt_instances["yaws"][gt_idx]

        loss_bbox_xy = F.smooth_l1_loss(
            src_boxes[..., :2],
            target_boxes[..., :2],
            reduction="none",
            beta=self.beta,
        )
        encoder_src_wh = torch.log(src_boxes[..., 2:])
        encoder_tgt_wh = torch.log(target_boxes[..., 2:])
        loss_bbox_wh = F.l1_loss(
            encoder_src_wh, encoder_tgt_wh, reduction="none"
        )

        def _get_iou_loss(csrc_boxes, ctarget_boxes, csrc_yaws, ctgt_yaws):
            if self.iou_loss_type == "giou":
                loss_iou = 1 - torch.diag(
                    generalized_box_iou(
                        box_center_to_corner(csrc_boxes),
                        box_center_to_corner(ctarget_boxes),
                    )
                )
            elif self.iou_loss_type == "diou":
                loss_iou = 1 - torch.diag(
                    distance_box_iou(
                        box_center_to_corner(csrc_boxes),
                        box_center_to_corner(ctarget_boxes),
                    )
                )
            elif self.iou_loss_type == "ciou":
                loss_iou = 1 - torch.diag(
                    complete_box_iou(
                        box_center_to_corner(csrc_boxes),
                        box_center_to_corner(ctarget_boxes),
                        yaws1=None if self.use_psc_rot else csrc_yaws,
                        yaws2=None if self.use_psc_rot else ctgt_yaws,
                    )
                )
            else:
                loss_iou = torch.tensor(0.0).to(loss_bbox.device)
            return loss_iou

        loss_iou = _get_iou_loss(
            src_boxes,
            target_boxes,
            src_yaws,
            target_yaws,
        )

        losses = {
            "loss_xy": loss_bbox_xy.sum(),
            "loss_wh": loss_bbox_wh.sum(),
            f"loss_{self.iou_loss_type}": loss_iou.sum(),
        }

        if "pred_boxes_kalerman" in outputs:
            src_boxes = outputs["pred_boxes_kalerman"][src_idx]
            replace_index = ~outputs["replace_index"][src_idx]
            if replace_index.sum() > 0:
                loss_bbox = F.smooth_l1_loss(
                    src_boxes[replace_index],
                    target_boxes[replace_index],
                    reduction="none",
                    beta=self.beta,
                )
                loss_iou = _get_iou_loss(
                    src_boxes[replace_index],
                    target_boxes[replace_index],
                    src_yaws[replace_index],
                    target_yaws[replace_index],
                )

                losses["loss_xy"] += loss_bbox[..., :2].sum()
                losses["loss_wh"] += loss_bbox[..., 2:4].sum()
                losses[f"loss_{self.iou_loss_type}"] += loss_iou.sum()

        return losses

    def get_zheight_loss(
        self,
        outputs: dict,
        gt_instances: dict,
        indices: tuple,
        matched_ious: torch.Tensor = None,
    ):
        """Compute the losses of box z coordinate and height.

        using smooth l1 loss default.

        Args:
            outputs: pred outputs, must contains `pred_zheights`.
            gt_instances: dict of gt instance.
            indices: src and gt match indices.
            matched_ious: ious between all queries and gt to filter the
                trk query by iou.

        """
        # We ignore the regression loss of the track-disappear slots.
        assert "pred_zheights" in outputs
        src_idx, gt_idx = indices
        if src_idx.numel() == 0:
            return {"loss_zheight": torch.tensor(0.0).to(src_idx)}
        src_zheights = outputs["pred_zheights"][src_idx]
        target_zheights = torch.stack(
            (
                gt_instances["bev_loc_z"][gt_idx],
                gt_instances["heights"][gt_idx],
            ),
            dim=-1,
        )
        loss_zs = F.smooth_l1_loss(
            src_zheights[..., 0],
            target_zheights[..., 0],
            reduction="none",
            beta=self.beta,
        )
        loss_hs = F.l1_loss(
            torch.log(src_zheights[..., 1]),
            torch.log(target_zheights[..., 1]),
            reduction="none",
        )
        loss_zhs = torch.stack((loss_zs, loss_hs), dim=-1)

        losses = {"loss_zheight": loss_zhs.sum()}
        return losses

    def get_yaw_loss(
        self,
        outputs: dict,
        gt_instances: dict,
        indices: tuple,
        matched_ious: torch.Tensor = None,
    ):
        """Compute the losses related to the yaws of boxes.

        the smooth L1 regression loss targets dicts must contain
        the key "pred_yaws" containing a tensor of dim [num_target_boxes, 2].
        The target yaws are expected in format (num_pred_boxes, 2).

        Args:
            outputs: pred outputs, must contains `pred_yaws`.
            gt_instances: dict of gt instance.
            indices: src and gt match indices.
            matched_ious: ious between all queries and gt to filter the
                trk query by iou.

        """
        # We ignore the regression loss of the track-disappear slots.
        assert "pred_yaws" in outputs
        src_idx, gt_idx = indices
        if src_idx.numel() == 0:
            return {"loss_yaw": torch.tensor(0.0).to(src_idx)}

        src_yaws = outputs["pred_yaws"][src_idx]
        target_yaws = gt_instances["yaws"][gt_idx]

        if self.use_psc_rot:
            # hack implementation here, since e2e not have the weight_mask and ignore mask. # noqa
            yaw_weight_mask = torch.ones_like(src_yaws[:, 0:1])
            yaw_ignore_mask = torch.zeros_like(src_yaws[:, 0:1])
            loss_yaws = hm_l1_wing_loss(
                src_yaws,
                target_yaws,
                yaw_weight_mask,
                yaw_ignore_mask,
                avg_factor=5,  # hard here, it's the exp value.
            )
        else:
            # 计算输出的车身中轴线与target车身中轴线的夹角误差(<=90度)。因为yaw的取值范围是-180~180，所以通过取src_yaw
            # 与target_yaws，以及src_yaw与-target_yaws（相当于target_yaws+180度）的较小值来计算
            # Loss_yaw。这么做相较于车头角度回归，训练收敛的更好，evs的指标更优
            loss_yaws = F.smooth_l1_loss(
                src_yaws, target_yaws, reduction="none", beta=self.beta
            )
            loss_inverse_yaws = F.smooth_l1_loss(
                src_yaws, -target_yaws, reduction="none", beta=self.beta
            )
            loss_yaws = torch.min(loss_yaws, loss_inverse_yaws)
        losses = {"loss_yaw": loss_yaws.sum()}
        return losses


class E2EDynamicVeloLoss(E2EDynamicTemplateLoss):
    """calculate velocity loss of e2e-dynamic task.

    Args:
        loss_weights: Global loss weight for each sub-loss.(e.g.
            loss_velocity:1, ...)
        alpha: focal loss param.
        gamma: focal loss param.
        beta: smooth_l1 loss param.
        use_psc_rot: whether use psc to encode the heading angle.
        N_steps_PSC_rot: nums of steps to decode the heading angle by PSC.
            Refer to https://horizonrobotics.feishu.cn/docx/EwYEdysIBodB69xn6Mdc7HkSnif. # noqa
        rot_mod_threshold: threshold to filter decoded phase of the heading angle.
    """

    def __init__(
        self,
        loss_weights: dict,
        alpha: float = 0.25,
        gamma: float = 2,
        beta: float = 0.01,
        use_psc_rot: bool = False,
        N_steps_PSC_rot: int = None,
        rot_mod_threshold: float = 0.0001,
    ):
        super().__init__()
        self.loss_weights = loss_weights
        self.beta = beta
        self.loss_map = {
            "loss_velocity": self.get_velocity_loss,
        }
        self.focal_loss = FocalLossV2(
            alpha=alpha,
            gamma=gamma,
            from_logits=False,
            reduction="none",
        )
        self.use_psc_rot = use_psc_rot
        self.N_steps_PSC_rot = N_steps_PSC_rot
        self.rot_mod_threshold = rot_mod_threshold
        if self.use_psc_rot:
            assert self.N_steps_PSC_rot >= 3
            self.coef_sin, self.coef_cos = get_coef_of_psc(
                self.N_steps_PSC_rot, return_tensor=True
            )

    def decode_psc_rot(self, psc_rot):
        self.coef_sin = self.coef_sin.to(psc_rot)
        self.coef_cos = self.coef_cos.to(psc_rot)
        phase_sin = torch.sum(
            psc_rot[..., : self.N_steps_PSC_rot] * self.coef_sin,
            dim=-1,
            keepdim=False,
        )
        phase_cos = torch.sum(
            psc_rot[..., : self.N_steps_PSC_rot] * self.coef_cos,
            dim=-1,
            keepdim=False,
        )
        phase_mod = phase_cos ** 2 + phase_sin ** 2
        phase = -torch.atan2(phase_sin, phase_cos)

        phase[phase_mod < self.rot_mod_threshold] *= 0

        return phase

    def get_velocity_loss(
        self,
        outputs: dict,
        gt_instances: dict,
        indices: tuple,
        matched_ious: torch.Tensor = None,
    ):
        """Compute the smooth_l1_loss of pred velocity.

        Args:
            outputs: pred outputs, must contains `pred_velocities` and
                `pred_yaws`.
            gt_instances: dict of gt instance.
            indices: src and gt match indices.
            matched_ious: ious between all queries and gt to filter the
                trk query by iou.

        """
        assert "pred_velocities" in outputs
        assert "pred_yaws" in outputs
        src_idx, gt_idx = indices
        if src_idx.numel() == 0:
            return {
                "loss_velocity": torch.tensor(0.0).to(
                    outputs["pred_velocities"]
                ),
                # loss_velo_yaw_consistency push the direction of vero close
                # to pred-yaws, which slight benefit performance.
                "loss_velo_yaw_consistency": torch.tensor(0.0).to(
                    outputs["pred_yaws"]
                ),
            }
        src_velocities = outputs["pred_velocities"][src_idx]
        src_yaw = outputs["pred_yaws"][src_idx].clone()

        # convert src yaw from psc->(cos, sin)
        if self.use_psc_rot:
            src_yaw_rad = self.decode_psc_rot(src_yaw)
            src_yaw = torch.stack(
                [torch.cos(src_yaw_rad), torch.sin(src_yaw_rad)], dim=1
            )

        target_velocities = gt_instances["velocities"][gt_idx]

        def _calc_slice_velo_loss(cpred_velo, ctarget_velo, csrc_yaw):
            loss_velocities = torch.norm(
                cpred_velo[:, :2] - ctarget_velo[:, :2], dim=1
            )
            velo_yaw = torch.atan2(
                cpred_velo[:, 1], cpred_velo[:, 0]
            ).unsqueeze(1)
            velo_regress = torch.cat(
                [torch.cos(velo_yaw), torch.sin(velo_yaw)], dim=1
            )
            # NOTE: compare consistance should replace with old-code.
            loss_velocities_yaw_consistent = F.smooth_l1_loss(
                velo_regress, csrc_yaw, beta=self.beta
            )
            return loss_velocities, loss_velocities_yaw_consistent

        loss_velo, loss_velo_yaw_consistent = _calc_slice_velo_loss(
            src_velocities, target_velocities, src_yaw
        )

        loss_velo = loss_velo.sum() / (src_idx.numel() + EPS)
        loss_velo_yaw_consistent = loss_velo_yaw_consistent / (
            src_idx.numel() + EPS
        )

        losses = {
            "loss_velocity": loss_velo,
            "loss_velo_yaw_consistency": loss_velo_yaw_consistent,
        }

        if "pred_velocities_kalerman" in outputs:
            src_velocities = outputs["pred_velocities_kalerman"][src_idx]
            replace_index = ~outputs["replace_index"][src_idx]
            if replace_index.sum() > 0:
                loss_velo, loss_velo_yaw_consistent = _calc_slice_velo_loss(
                    src_velocities[replace_index],
                    target_velocities[replace_index],
                    src_yaw[replace_index],
                )

                loss_velo = loss_velo.sum() / (torch.sum(replace_index) + EPS)
                loss_velo_yaw_consistent = loss_velo_yaw_consistent / (
                    torch.sum(replace_index) + EPS
                )

                losses["loss_velocity"] += loss_velo
                losses["loss_velo_yaw_consistency"] += loss_velo_yaw_consistent
        return losses


class E2EDynamicTrajLoss(E2EDynamicTemplateLoss):
    """calculate trajectory-pred loss of e2e-dynamic task.

    Args:
        loss_weights: Global loss weight for each sub-loss.(e.g.
            loss_trajreg:1, ...)
    """

    def __init__(self, loss_weights: dict):
        super().__init__()
        self.loss_weights = loss_weights
        self.loss_map = {
            "trajectory": self.get_trajpred_loss,
        }

    def get_match_idx_use_l2_distance(
        self,
        gt_traj: torch.Tensor,
        pred_trajs: torch.Tensor,
        gt_mask: torch.Tensor = None,
        matched_ious: torch.Tensor = None,
    ):
        """
        Index of output trajs that is closest to GT-traj as GT-modal.

        Distance is measured by L2-norm.

        Args:
            pred_trajs: multi-modal predicted trajectory.
            gt_traj: GT trajectory.
            gt_mask: GT traj mask corresponding to gt_traj.
            matched_ious: ious between all queries and gt to filter the
                trk query by iou.

        """
        if gt_mask is None:
            gt_mask = torch.ones_like(gt_traj)
        dist = (gt_traj.unsqueeze(1) - pred_trajs) * gt_mask.unsqueeze(1)
        dist = torch.norm(dist, dim=-1)  # [num_obs, num_anchor, traj_len]
        dist[torch.isnan(dist)] = 0
        penalty = torch.sum(dist, dim=-1)  # [num_obs, num_anchor]
        return torch.argmin(penalty, dim=-1)

    def get_trajpred_loss(
        self,
        outputs: dict,
        gt_instances: dict,
        indices: tuple,
        matched_ious: torch.Tensor = None,
    ):
        """Compute the loss of pred trajs and modals.

        trajs-loss: smooth_l1_loss.
        modals: cross_entropy loss.
        Args:
            outputs: pred outputs, must contains `pred_TrajRegs` and
                `pred_TrajLogits`.
            gt_instances: dict of gt instance.
            indices: src and gt match indices.
            src_idx: the matched index of src in indices.
            matched_ious: ious between all queries and gt to filter the
                trk query by iou.

        """
        assert "pred_TrajRegs" in outputs
        assert "pred_TrajLogits" in outputs
        pred_trajregs = (
            outputs["pred_TrajRegs"].permute(0, 2, 1, 3, 4).squeeze(0)
        )
        # [-1, num_anchors, traj_len, 2]
        pred_trajlogits = (
            outputs["pred_TrajLogits"]
            .permute(0, 2, 1, 3)
            .squeeze(0)
            .squeeze(-1)
        )
        # [-1, num_anchors]

        with torch.no_grad():
            src_idx, gt_idx = indices

        if 0 == len(src_idx):
            losses = {
                "loss_trajprob": pred_trajregs.sum() * 0,
                "loss_trajreg": pred_trajlogits.sum() * 0,
            }
            return losses
        # get all matched rows in output trajs and logits
        # [num_obs, num_modal, traj_len, 2]
        match_obs_multimodal_traj = pred_trajregs[src_idx]
        # [num_obs, num_modal]
        match_log_cls = pred_trajlogits[src_idx]

        with torch.no_grad():
            gt_traj = gt_instances["gt_traj_regs"][gt_idx]
            gt_mask = gt_instances["gt_traj_masks"][gt_idx]

            match_label = self.get_match_idx_use_l2_distance(
                gt_traj, match_obs_multimodal_traj, gt_mask
            )

            valid_flag = (
                torch.sum(gt_mask, dim=[-2, -1])
                .to(torch.bool)
                .to(match_log_cls.dtype)
            )  # [num_obs]

        cls_loss = F.cross_entropy(
            match_log_cls, match_label, reduction="none"
        )

        cls_loss = torch.sum(cls_loss * valid_flag) / (EPS + valid_flag.sum())

        # [num_obs, traj_len, 2]
        mt_trajs_idx = [torch.arange(len(match_label)), match_label]
        match_pred_traj = match_obs_multimodal_traj[mt_trajs_idx]

        reg_loss = F.smooth_l1_loss(match_pred_traj, gt_traj, reduction="none")

        reg_loss = (reg_loss * gt_mask).sum() / (EPS + gt_mask[..., 0].sum())

        losses = {"loss_trajprob": cls_loss, "loss_trajreg": reg_loss}
        return losses
