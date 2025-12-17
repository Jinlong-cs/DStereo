# Copyright (c) Horizon Robotics. All rights reserved.

import copy
from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from hat.models.losses.focal_loss import FocalLossV2
from hat.models.losses.real3d_losses import (
    hm_focal_loss,
    hm_l1_loss,
    hm_l1_wing_loss,
    sigmoid_and_clip,
    softmax_and_clip,
)
from hat.models.losses.softmax_ce_loss import SoftmaxCELoss
from hat.models.task_modules.e2e_dynamic.e2e_dynamic_util import decode_psc_rot
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list

__all__ = [
    "ANCBEVDiscObjRelativeLoss",
    "ANCBEV3DLoss",
    "ANCBEVSegLoss",
    "ANCBEVDiscreteObjectLoss",
    "ANCBEVDiscreteObjectWithClsLoss",
]


@OBJECT_REGISTRY.register
class ANCBEVDiscObjRelativeLoss(nn.Module):
    """Calculate BEV discrete obj relative loss between different class.

        Relative loss compute the relative location and rot loss between
        different objects in the same timestamp input.
        If there only one object in input, the relative loss
        degrade to no-operation.

    Args:
        loss_weights: default: None
            Global loss weight for each sub-loss. Default
            weight is 1.0. (e.g. bev_discobj_rel_loc:1,
            bev_discobj_rel_rot:1 etc.).
        gt_name: Gt name of current branch.
    Returns:
        a dict contains loss.
    """

    def __init__(
        self,
        loss_weights: Optional[Dict] = None,
        gt_name: str = "bev_discrete_obj",
    ):
        super(ANCBEVDiscObjRelativeLoss, self).__init__()
        self.loss_weights = defaultdict(lambda: 1.0)
        if loss_weights is not None:
            self.loss_weights.update(**loss_weights)
        self.gt_name = gt_name

    def generate_grid_coord(self, data):
        h, w = data.shape[-2:]
        x = torch.arange(0, w, dtype=torch.float32).to(data.device)
        y = torch.arange(0, h, dtype=torch.float32).to(data.device)
        gridy, gridx = torch.meshgrid(y, x)
        grid_coord = torch.stack([gridx, gridy], dim=-1)
        return grid_coord

    def gather_inst_ct_loc_rot(self, hm, rot, ct_offset, inst_mask):
        grid_coord = self.generate_grid_coord(hm)  # (h, w, 2) <--> (u, v, 2)
        inst_hm = torch.sum(hm * inst_mask, dim=0, keepdim=False)
        # to handle many identical max values for pred hm
        inst_hm = (
            inst_hm
            + torch.rand(
                inst_hm.shape, dtype=inst_hm.dtype, device=inst_hm.device
            )
            * 1e-3
        )
        # inst_hm may have many identical max values such as
        # max is 0 at beginning training
        inst_ct_mask = inst_hm == inst_hm.max()  # (h, w)
        inst_ct_rot_cos = rot[0, :, :][inst_ct_mask]  # (n,)
        inst_ct_rot_sin = rot[1, :, :][inst_ct_mask]
        inst_ct_u = grid_coord[:, :, 0][inst_ct_mask]
        inst_ct_v = grid_coord[:, :, 1][inst_ct_mask]
        inst_ct_offset_u = ct_offset[0, :, :][inst_ct_mask]
        inst_ct_offset_v = ct_offset[1, :, :][inst_ct_mask]

        # sometimes stopline hm gt can have multiple 1's for the same object
        if inst_ct_rot_cos.shape[0] > 1:
            inst_ct_rot_cos = inst_ct_rot_cos[0:1]
            inst_ct_rot_sin = inst_ct_rot_sin[0:1]
            inst_ct_u = inst_ct_u[0:1]
            inst_ct_v = inst_ct_v[0:1]
            inst_ct_offset_u = inst_ct_offset_u[0:1]
            inst_ct_offset_v = inst_ct_offset_v[0:1]

        inst_ct_rot = torch.cat(
            [inst_ct_rot_cos, inst_ct_rot_sin], dim=0
        )  # (2,)
        inst_ct = torch.cat([inst_ct_u, inst_ct_v], dim=0)  # (2,) <--> (u, v)
        inst_ct_offset = torch.cat(
            [inst_ct_offset_u, inst_ct_offset_v], dim=0
        )  # (2,) <--> (u, v)
        inst_ct = inst_ct + inst_ct_offset
        return inst_ct, inst_ct_rot

    @autocast(enabled=False)
    def relative_loss(
        self,
        pred_rot: torch.Tensor,
        pred_ct_offset: torch.Tensor,
        pred_hm: torch.Tensor,
        target_rot: torch.Tensor,
        target_ct_offset: torch.Tensor,
        target_hm: torch.Tensor,
        target_instances: torch.Tensor,
    ):
        """Calculate BEV discrete obj relative loss between different class.

        Args:
            pred_rot: Predicted rotation angle.
            pred_ct_offset: Predicted center offset.
            pred_hm: Predictied center heatmap.
            target_rot: Target rotation angle.
            target_ct_offset: Target center offset.
            target_hm: Target center heatmap.
            target_instances: Target instance maps.
        Returns:
            relative class loss.
        """

        b = target_hm.shape[0]
        loss_rel_loc = (0.0 * pred_hm).sum()
        loss_rel_rot = (0.0 * pred_hm).sum()
        num_pairs = 0
        for bid in range(b):
            one_target_rot = target_rot[bid, :, :, :]
            one_target_ct_offset = target_ct_offset[bid, :, :, :]
            one_target_hm = target_hm[bid, :, :, :]  # (c, h, m)
            one_pred_rot = pred_rot[bid, :, :, :]  # (2, h, w)
            one_pred_ct_offset = pred_ct_offset[bid, :, :, :]
            one_pred_hm = pred_hm[bid, :, :, :]  # (c, h, m)
            num_pos = (one_target_hm == 1.0).sum()

            one_num_pairs = 0
            if num_pos > 1:
                inst_ids = torch.unique(
                    target_instances[bid, :, :, :], sorted=True
                )[
                    1:
                ]  # exclude background id 0
                while one_num_pairs < inst_ids.shape[0] // 2:
                    # get index corresponding to the first
                    # instance on the heatmap
                    inst0_mask = (
                        target_instances[bid, :, :, :]
                        == inst_ids[one_num_pairs * 2]
                    )
                    # get the predicted center and predicted
                    # rot of the instance through the instance index
                    inst0_ct, inst0_ct_rot = self.gather_inst_ct_loc_rot(
                        one_pred_hm,
                        one_pred_rot,
                        one_pred_ct_offset,
                        inst0_mask,
                    )
                    # get the gt center and gt rot of the instance
                    # through the instance index
                    inst0_gt_ct, inst0_gt_ct_rot = self.gather_inst_ct_loc_rot(
                        one_target_hm,
                        one_target_rot,
                        one_target_ct_offset,
                        inst0_mask,
                    )

                    # inst1
                    # get index corresponding to the second instance
                    # on the heatmap
                    inst1_mask = (
                        target_instances[bid, :, :, :]
                        == inst_ids[one_num_pairs * 2 + 1]
                    )
                    # get the predicted center and predicted rot
                    # of the instance through the instance index
                    inst1_ct, inst1_ct_rot = self.gather_inst_ct_loc_rot(
                        one_pred_hm,
                        one_pred_rot,
                        one_pred_ct_offset,
                        inst1_mask,
                    )
                    # get the gt center and gt rot of the instance
                    # through the instance index
                    inst1_gt_ct, inst1_gt_ct_rot = self.gather_inst_ct_loc_rot(
                        one_target_hm,
                        one_target_rot,
                        one_target_ct_offset,
                        inst1_mask,
                    )
                    # relative rot loss
                    # get the predicted rot difference between matched object
                    # diff_cos=cos(theta1-theta0)
                    # diff_sin=sin(theta1-theta0)
                    pred_rot_diff_cos = (
                        inst0_ct_rot[0] * inst1_ct_rot[0]
                        + inst0_ct_rot[1] * inst1_ct_rot[1]
                    )
                    pred_rot_diff_sin = (
                        inst0_ct_rot[1] * inst1_ct_rot[0]
                        - inst0_ct_rot[0] * inst1_ct_rot[1]
                    )
                    # get the gt rot difference between matched object
                    # diff_cos=cos(theta1-theta0)
                    # diff_sin=sin(theta1-theta0)
                    gt_rot_diff_cos = (
                        inst0_gt_ct_rot[0] * inst1_gt_ct_rot[0]
                        + inst0_gt_ct_rot[1] * inst1_gt_ct_rot[1]
                    )
                    gt_rot_diff_sin = (
                        inst0_gt_ct_rot[1] * inst1_gt_ct_rot[0]
                        - inst0_gt_ct_rot[0] * inst1_gt_ct_rot[1]
                    )
                    # align prediciton's rot loss with gt's rot loss between
                    # matched instance
                    loss_rot = F.l1_loss(
                        pred_rot_diff_cos, gt_rot_diff_cos, reduction="sum"
                    ) + F.l1_loss(
                        pred_rot_diff_sin, gt_rot_diff_sin, reduction="sum"
                    )
                    # relative location loss
                    # align prediction's center loss with gt's center loss
                    # between matched instance
                    loss_loc = F.l1_loss(
                        inst0_ct - inst1_ct,
                        inst0_gt_ct - inst1_gt_ct,
                        reduction="sum",
                    )
                    loss_rel_loc = loss_rel_loc + loss_loc
                    loss_rel_rot = loss_rel_rot + loss_rot
                    one_num_pairs += 1
                    num_pairs += 1
        loss_rel_loc = loss_rel_loc / max(num_pairs, 1)
        loss_rel_rot = loss_rel_rot / max(num_pairs, 1)
        # get the final matched instance loss
        loss_bev_discobj_rel_group = (
            loss_rel_loc * self.loss_weights["bev_discobj_rel_loc"]
            + loss_rel_rot * self.loss_weights["bev_discobj_rel_rot"]
        )
        return loss_bev_discobj_rel_group

    @autocast(enabled=False)
    def forward(self, pred_dict, target_dict, pred_hm):
        assert self.gt_name in target_dict
        target = target_dict[self.gt_name]

        target_hm = target["bev_discobj_hm"]
        assert "bev_discobj_instances" in target
        target_instances = target["bev_discobj_instances"]
        result_dict = {}
        if target_instances.max() != 0:
            loss_bev_discobj_rel_group = self.relative_loss(
                pred_rot=pred_dict["pred_bev_discobj_rot"],
                pred_ct_offset=pred_dict["pred_bev_discobj_ct_offset"],
                pred_hm=pred_hm,
                target_rot=target["bev_discobj_rot"],
                target_ct_offset=target["bev_discobj_ct_offset"],
                target_hm=target_hm,
                target_instances=target_instances,
            )
        else:
            loss_bev_discobj_rel_group = (0.0 * pred_hm).sum()
        result_dict["loss_bev_discobj_group_rel"] = loss_bev_discobj_rel_group

        return result_dict


@OBJECT_REGISTRY.register
class ANCBEV3DLoss(nn.Module):
    """Calculate bev3d losses.

    Classification uses focal loss.
    Regression (dimension, rotation, center_offset etc.) use L1 loss.

    Args:
        loss_weights:, default: None
            Global loss weight for each sub-loss. Default
            weight is 1.0. (e.g. bev3d_hm:1, bev3d_dim:1,
            bev3d_rot:1 etc.).
        class_weight: loss weight for each class.
        gamma: gamma paras of hm_focal_loss. Default: 2.
        beta: beta paras of hm_focal_loss. Default: 1.
        use_category_bce: Whether to use bce auxiliary loss for category.
        use_norm_rot: whether norm rot loss to [-1,1].
        gt_name: gt name.
        use_rot_wing_loss: whether use wing loss for bev3d rot regression.
        return_latest_flag: whether calculate loss for all frame in a clip.
    Returns:
        a dict contains loss.
    """

    def __init__(
        self,
        loss_weights: Optional[dict] = None,
        class_weight: Optional[list] = None,
        gamma: float = 2,
        beta: float = 1,
        use_category_bce: bool = False,
        use_norm_rot: bool = False,
        gt_name: str = "gt_bev_3d",
        use_rot_wing_loss: bool = False,
        return_latest_flag: bool = True,
        vcs_range: Sequence[float] = (-50, 50, -50, 50),
        use_psc_rot: bool = False,
        N_steps_PSC_rot: int = None,
        cls_dimension: Sequence[float] = (1.6, 1.8, 4.0),
    ):
        super(ANCBEV3DLoss, self).__init__()
        self.loss_weights = defaultdict(lambda: 1.0)
        self.class_weight = class_weight
        self.gamma = gamma
        self.beta = beta
        self.use_category_bce = use_category_bce
        self.use_norm_rot = use_norm_rot
        self.gt_name = gt_name
        self.bce_loss = FocalLossV2(
            alpha=0.5, gamma=self.gamma, from_logits=True, reduction="mean"
        )
        if loss_weights is not None:
            self.loss_weights.update(**loss_weights)

        self.ce_focal_loss = FocalLossV2(
            alpha=1.0,
            gamma=self.gamma,
            from_logits=True,
            reduction="mean",
        )
        self.use_rot_wing_loss = use_rot_wing_loss
        self.reg_keys = list(self.loss_weights.keys())
        for key in copy.deepcopy(self.reg_keys):
            if "hm" in key:
                self.reg_keys.remove(key)
        self.return_latest_flag = return_latest_flag
        self.vcs_range = vcs_range
        self.use_psc_rot = use_psc_rot
        self.N_steps_PSC_rot = N_steps_PSC_rot
        self.cls_dimension = cls_dimension
        if self.use_psc_rot:
            assert self.N_steps_PSC_rot >= 3
            self.coef_sin = torch.tensor(
                tuple(
                    torch.sin(
                        torch.tensor(2 * k * torch.pi / self.N_steps_PSC_rot)
                    )
                    for k in range(self.N_steps_PSC_rot)
                )
            )
            self.coef_cos = torch.tensor(
                tuple(
                    torch.cos(
                        torch.tensor(2 * k * torch.pi / self.N_steps_PSC_rot)
                    )
                    for k in range(self.N_steps_PSC_rot)
                )
            )

    def get_gereral_corner(self, pred):
        # tramsform ct to vcs coordinate
        offset = pred["bev3d_ct_offset"]
        _, _, height, width = offset.shape
        height_range = torch.arange(height).to(offset.device)
        width_range = torch.arange(width).to(offset.device)
        hw_heatmap = torch.stack(
            torch.meshgrid([height_range, width_range]), dim=-1
        ).flip(-1)
        bev_center = hw_heatmap + offset.permute(0, 2, 3, 1)
        vcs_range = torch.tensor(self.vcs_range).to(offset.device)
        m_perpixel = (
            abs(vcs_range[2] - vcs_range[0]) / height,
            abs(vcs_range[3] - vcs_range[1]) / width,
        )  # bev coord [y, x]

        vcs_x = vcs_range[2] - (bev_center[..., 1]) * m_perpixel[0]
        vcs_y = vcs_range[3] - (bev_center[..., 0]) * m_perpixel[1]
        vcs_center = torch.stack([vcs_x, vcs_y], dim=-1)  # BHW2

        # convert the rot to rad value (sin / cos), range=[-pi,pi]
        bev3d_rot = pred["bev3d_rot"].permute(0, 2, 3, 1)
        if self.use_psc_rot:
            yaw = decode_psc_rot(
                bev3d_rot=bev3d_rot,
                coef_sin=self.coef_sin,
                coef_cos=self.coef_cos,
                N_steps_PSC_rot=self.N_steps_PSC_rot,
                use_sigmoid=self.use_norm_rot,
            )
        else:
            if self.use_norm_rot:
                bev3d_rot = torch.add(
                    torch.mul(bev3d_rot.sigmoid(), 2), -1
                )  # norm rot
            yaw = torch.atan2(bev3d_rot[..., 1], bev3d_rot[..., 0])

        # process the pred residual dim to real dim.
        cls_dimension = (
            torch.from_numpy(self.cls_dimension).float().to(offset.device)
        )
        average_dim = cls_dimension[pred["bev3d_cls_hm"].argmax(dim=1)]
        dim = torch.exp(pred["bev3d_dim"].permute(0, 2, 3, 1)) * average_dim

        # get the pred corner
        half_w = dim[..., 1] / 2.0  # BHW
        half_l = dim[..., 2] / 2.0

        # Rotation matrix around the yaw axis
        rotation_matrix = torch.stack(
            [
                torch.stack([torch.cos(yaw), -torch.sin(yaw)], dim=-1),  # BHW2
                torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=-1),
            ],
            dim=-2,
        )  # BHW22

        # 2D box corners in object coordinate frame
        corners_object_frame = torch.stack(
            [
                torch.stack([-half_l, -half_w], dim=-1),  # BHW2
                torch.stack([half_l, -half_w], dim=-1),
                torch.stack([half_l, half_w], dim=-1),
                torch.stack([-half_l, half_w], dim=-1),
            ],
            dim=-2,
        )  # BHW42

        # Rotate and translate the corners to the global coordinate frame
        corners_global = corners_object_frame @ rotation_matrix.transpose(
            -2, -1
        ) + vcs_center.unsqueeze(
            -2
        )  # BHW42

        # reweight the corner by distance
        weight = torch.linalg.norm(
            corners_global, dim=-1, keepdim=True
        ).detach()  # BHW41
        weight = 100 / (weight + 1e-6)

        # mask the corner by vcs range
        weight_mask = (
            (
                (corners_global[..., 0] > -5)
                & (corners_global[..., 0] < 20)
                & (corners_global[..., 1] > -3.5)
                & (corners_global[..., 1] < 3.5)
            )
            .float()
            .unsqueeze(-1)
        )  # BHW41

        return corners_global, weight, weight_mask

    def occlusion_loss(self, pred, target, ignore_mask):
        loss_weight = self.loss_weights["bev3d_occlusion_hm"]
        if not self.return_latest_flag:
            ignore_occlusion_mask = target["bev3d_ignore_occlusion"].reshape(
                -1
            )
        else:
            ignore_occlusion_mask = target["bev3d_ignore_occlusion"].squeeze(
                dim=1
            )
        pred_occlusion_hm = pred["bev3d_occlusion_hm"][~ignore_occlusion_mask]
        target_occlusion_hm = target["bev3d_occlusion_hm"][
            ~ignore_occlusion_mask
        ]
        ignore_occlusion_mask = ignore_mask[~ignore_occlusion_mask]
        pos_mask = target_occlusion_hm * (1 - ignore_occlusion_mask)
        pos_num = pos_mask.sum()
        pos_num = torch.max(pos_num, torch.ones_like(pos_num))
        pred_occlusion_hm_softmax = softmax_and_clip(pred_occlusion_hm, dim=1)
        occlusion_heatmap_loss = (
            self.ce_focal_loss(
                pred=pred_occlusion_hm_softmax,
                target=pos_mask,
                avg_factor=pos_num,
            )
            * loss_weight
        )

        pred_occlusion_hm_sigmoid = sigmoid_and_clip(pred_occlusion_hm)
        pred_ch = pred_occlusion_hm_sigmoid.shape[1]
        bce_loss_mask, _ = torch.max(pos_mask, dim=1, keepdim=True)
        bce_loss_mask = torch.tile(bce_loss_mask, [1, pred_ch, 1, 1]) * 2.0
        category_bce_loss = self.bce_loss(
            pred=pred_occlusion_hm_sigmoid,
            target=pos_mask,
            weight=bce_loss_mask,
            avg_factor=pos_num,
        )
        occlusion_heatmap_loss += category_bce_loss * loss_weight
        return occlusion_heatmap_loss

    @autocast(enabled=False)
    def forward(self, pred: Mapping, target: Mapping):
        """Calculate bev3d losses between pred and target items.

        Args:
            pred: predict bev3d output (e.g. bev3d_hm, bev3d_dim etc.)
            target: target contains the bev3d ground truth
        """

        assert self.gt_name in target
        target = target[self.gt_name]

        # convert to float32 while using amp
        for k, v in pred.items():
            pred[k] = v.float()

        all_losses = {}

        ignore_mask = target.get(
            "bev3d_ignore_mask", torch.zeros_like(target["bev3d_weight_hm"])
        )

        bev3d_roi_weight = target["bev3d_roi_weight"]

        # heatmap use focal loss
        if "bev3d_hm" in self.loss_weights.keys():
            bev3d_background_weight = target["bev3d_background_weight"]
            loss_weight = self.loss_weights["bev3d_hm"]
            pred_heatmap = sigmoid_and_clip(pred["bev3d_hm"])
            target_heatmap = target["bev3d_hm"]
            class_weight_map = torch.ones_like(pred_heatmap)
            if self.class_weight is not None:
                assert len(self.class_weight) == class_weight_map.shape[1]
                for i, one_weight in enumerate(self.class_weight):
                    class_weight_map[:, i, ...] = one_weight
            if bev3d_background_weight is not None:
                bev3d_background_weight = (
                    bev3d_background_weight * class_weight_map
                )
            else:
                bev3d_background_weight = class_weight_map
            heatmap_loss = (
                hm_focal_loss(
                    pred=pred_heatmap,
                    gt=target_heatmap,
                    ignore_mask=ignore_mask,
                    gamma=self.gamma,
                    beta=self.beta,
                    pos_cls_weight=class_weight_map * bev3d_roi_weight,
                    neg_cls_weight=bev3d_background_weight * bev3d_roi_weight,
                )
                * loss_weight
            )
            all_losses["bev3d_hm_loss"] = heatmap_loss
        if "bev3d_cls_hm" in self.loss_weights.keys():
            if not self.return_latest_flag:
                ignore_veh_cls_mask = target["bev3d_ignore_obj_cls"].reshape(
                    -1
                )
            else:
                ignore_veh_cls_mask = target["bev3d_ignore_obj_cls"].squeeze(
                    dim=1
                )
            loss_weight = self.loss_weights["bev3d_cls_hm"]
            # the sample in batch which set ignore_obj_cls to true,
            # will not participate in the cls_hm loss calculation.
            # the shape of ignore_veh_cls_mask is [batch_size]
            pred_cls_hm = pred["bev3d_cls_hm"][~ignore_veh_cls_mask]
            target_cls_hm = target["bev3d_cls_hm"][~ignore_veh_cls_mask]
            ignore_cls_mask = ignore_mask[~ignore_veh_cls_mask]
            category_class_weight_hm = target["bev3d_category_class_weight"][
                ~ignore_veh_cls_mask
            ]
            bev3d_roi_cls_weight = bev3d_roi_weight[~ignore_veh_cls_mask]

            pos_mask = target_cls_hm * (1 - ignore_cls_mask)
            pos_num = pos_mask.sum()
            pos_num = torch.max(pos_num, torch.ones_like(pos_num))
            pred_cls_hm_softmax = softmax_and_clip(pred_cls_hm, dim=1)
            cls_heatmap_loss = (
                self.ce_focal_loss(
                    pred=pred_cls_hm_softmax,
                    target=pos_mask,
                    weight=pos_mask
                    * category_class_weight_hm
                    * bev3d_roi_cls_weight,
                    avg_factor=pos_num,
                )
                * loss_weight
            )
            if self.use_category_bce:
                pred_cls_hm_sigmoid = sigmoid_and_clip(pred_cls_hm)
                pred_ch = pred_cls_hm_sigmoid.shape[1]
                bce_loss_mask, _ = torch.max(pos_mask, dim=1, keepdim=True)
                bce_loss_mask = (
                    torch.tile(bce_loss_mask, [1, pred_ch, 1, 1]) * 2.0
                )
                category_bce_loss = self.bce_loss(
                    pred=pred_cls_hm_sigmoid,
                    target=pos_mask,
                    weight=bce_loss_mask
                    * category_class_weight_hm
                    * bev3d_roi_cls_weight,
                    avg_factor=pos_num,
                )
                cls_heatmap_loss += category_bce_loss * loss_weight
            all_losses["bev3d_cls_hm_loss"] = cls_heatmap_loss
        if "bev3d_occlusion_hm" in self.loss_weights.keys():
            occlusion_heatmap_loss = self.occlusion_loss(
                pred, target, ignore_mask
            )
            all_losses["bev3d_occlusion_hm_loss"] = occlusion_heatmap_loss

        # regression map use L1 loss
        if "bev3d_corner" in self.loss_weights.keys():
            pred_corner, _, _ = self.get_gereral_corner(pred)
            target_corner, weight, weight_mask = self.get_gereral_corner(
                target
            )
            pred["bev3d_corner"] = (
                (pred_corner * weight).flatten(-2, -1).permute(0, 3, 1, 2)
            )
            target["bev3d_corner"] = (
                (target_corner * weight).flatten(-2, -1).permute(0, 3, 1, 2)
            )
        for key in self.reg_keys:
            if key not in pred:
                continue
            if self.use_norm_rot and key == "bev3d_rot":
                pred[key] = 2 * torch.sigmoid(pred[key]) - 1
            loss_weight = self.loss_weights[key]
            reg_loss_func = hm_l1_loss
            reg_loss_input = {
                "output": pred[key],
                "target": target[key],
                "weight_mask": target["bev3d_weight_hm"] * bev3d_roi_weight,
                "ignore_mask": ignore_mask,
                "heatmap_type": None,
            }

            if self.use_rot_wing_loss and key == "bev3d_rot":
                reg_loss_func = hm_l1_wing_loss
                reg_loss_input.pop("heatmap_type")
                reg_loss_input.update(
                    {
                        "avg_factor": target["bev3d_weight_hm"].sum(),
                        "omega": 0.5,
                        "sigma": 0.02,
                        "weight_mask": target["bev3d_weight_hm"]
                        * target["bev3d_rot_reweight_mask"]
                        * bev3d_roi_weight
                        if "bev3d_rot_reweight_mask" in target
                        else target["bev3d_weight_hm"] * bev3d_roi_weight,
                    }
                )

            loss = reg_loss_func(**reg_loss_input) * loss_weight
            all_losses["{}_loss".format(key)] = loss
        return all_losses


@OBJECT_REGISTRY.register
class ANCBEVSegLoss(nn.Module):  # noqa: D205,D400
    """Calculate bev seg loss.

    BEV seg head and BEV occlusion head use cross entropy loss.
    BEV confidence head use L1 loss.
    Ground truth for confidence prediction is
    max(softmax(pred_seg_logit)) along channel dim.

    Args:
        pred_names: prediction names.
        loss_names: loss name for each output.
        loss_seg_cfg: loss for seg.
        loss_occlusion_cfg: loss for seg.
        conf_loss_weight: loss weight for confidence output.
        gt_name: gt name.
    """

    def __init__(
        self,
        pred_names: Union[List[str], str],
        loss_names: Union[List[str], str],
        loss_seg_cfg: Optional[torch.nn.Module] = None,
        loss_occlusion_cfg: Optional[torch.nn.Module] = None,
        conf_loss_weight: Optional[float] = None,
        conf_gt_name: str = "pred_bev_segs_frame0",
        gt_name: str = "gt_bev_seg",
    ):
        super(ANCBEVSegLoss, self).__init__()

        self.pred_names = _as_list(pred_names)
        self.loss_names = _as_list(loss_names)
        assert len(self.pred_names) == len(self.loss_names)
        self.gt_name = gt_name
        self.conf_gt_name = conf_gt_name

        self.loss_seg = loss_seg_cfg
        self.loss_occlusion = loss_occlusion_cfg
        self.conf_loss_weight = conf_loss_weight

    def forward(self, pred_dict, target_dict):

        assert self.gt_name in target_dict
        assert self.conf_gt_name in pred_dict

        result_dict = {}
        for i, pred_name in enumerate(self.pred_names):
            if "bev_conf" in pred_name:
                # L1 loss for confidence map
                target_conf, _ = torch.max(
                    torch.softmax(pred_dict[self.conf_gt_name][0], dim=1),
                    dim=1,
                    keepdim=True,
                )
                res = (
                    hm_l1_loss(
                        pred_dict[pred_name][0],
                        target_conf,
                        torch.ones_like(target_conf),
                        torch.zeros_like(target_conf),
                        heatmap_type=None,
                    )
                    * self.conf_loss_weight
                )
            elif "bev_occlusion" in pred_name:
                res = self.loss_occlusion(
                    pred_dict[pred_name], target_dict["occlusion"]
                )
            elif "bev_seg" in pred_name:
                res = self.loss_seg(
                    pred_dict[pred_name], target_dict[self.gt_name]
                )

            result_dict[self.loss_names[i]] = res

        return result_dict


@OBJECT_REGISTRY.register
class ANCBEVDiscreteObjectLoss(nn.Module):
    """Calculate BEV discrete obj heatmap loss and dimension loss.

    Classification uses focal loss by default.
    Regression (dimension, rotation, center_offset
    etc.) use L1 loss.

    Args:
        loss_weights: default: None
            Global loss weight for each sub-loss. Default
            weight is 1.0. (e.g. bev_discobj_hm:1, bev_discobj_dim:1,
            bev_discobj_rot:1 etc.).
        use_focal_hm_loss: Use focal loss for heatmap otherwise l1 loss.
        group_rel_loss: Group relative loss.
        class_weights: Class weights used for
            classification.
        use_target_ignore:  Will not compute the loss of
            labeled ignore regions if true.
        use_norm_rot: whether norm rot loss to [-1,1].
        ohem_fp_loss_weight: Loss weight for ohem fp instance.
        ohem_fp_threshold: Score threshold to select ohem fp instance.
        gt_name: Gt name of current branch.
    Returns:
        a dict contains loss.
    """

    def __init__(
        self,
        loss_weights: Optional[Dict] = None,
        use_focal_hm_loss: bool = True,
        group_rel_loss: Optional[torch.nn.Module] = None,
        class_weights: Optional[Sequence[float]] = None,
        ohem_fp_loss_weight: float = None,
        ohem_fp_threshold: float = 1.0,
        use_target_ignore: bool = False,
        use_norm_rot: bool = False,
        gt_name: str = "bev_discrete_obj",
    ):
        super(ANCBEVDiscreteObjectLoss, self).__init__()
        self.loss_weights = defaultdict(lambda: 1.0)
        self.use_focal_hm_loss = use_focal_hm_loss
        self.use_target_ignore = use_target_ignore

        if loss_weights is not None:
            self.loss_weights.update(**loss_weights)
        self.reg_keys = list(self.loss_weights.keys())
        if "bev_discobj_hm" in self.reg_keys:
            self.reg_keys.remove("bev_discobj_hm")

        self.group_rel_loss = group_rel_loss
        self.class_weights = class_weights
        self.ohem_fp_loss_weight = ohem_fp_loss_weight
        self.ohem_fp_threshold = ohem_fp_threshold
        self.gt_name = gt_name

        self.use_norm_rot = use_norm_rot

    @autocast(enabled=False)
    def forward(self, pred_dict: Dict, target_dict: Dict):
        """Calculate loss between pred and target items.

        Args:
            pred_dict: Predict bev discrete obj output (e.g.,
                pred_bev_discobj_hm, pred_bev_discobj_wh, etc).
            target_dict: Target contains bev discrete obj ground truth
        """
        assert self.gt_name in target_dict
        target = target_dict[self.gt_name]

        # class ignore mask for missing label
        cls_ignore_mask = target.get("bev_discobj_cls_ignore", None)
        # class valid range mask for ignore loss in outrange area
        cls_valid_range_mask = target.get("bev_discobj_cls_valid_mask", None)

        pred_hm = pred_dict["pred_bev_discobj_hm"]
        target_hm = target["bev_discobj_hm"]

        assert "bev_discobj_weight_hm" in target
        weight_mask = target["bev_discobj_weight_hm"]
        target_ignore = target["bev_discobj_ignore"]
        if self.use_target_ignore:
            ignore_mask = (
                target_ignore.bool() | (weight_mask == 0).bool()
            ).float()
        else:
            ignore_mask = (weight_mask == 0).float()

        pred_hm = sigmoid_and_clip(pred_hm)
        result_dict = {}

        if self.use_focal_hm_loss:
            loss_weight = self.loss_weights["bev_discobj_hm"]
            if self.class_weights is not None:
                num_cls = pred_hm.shape[1]
                pos_cls_weight = (
                    torch.tensor(self.class_weights)
                    .reshape(1, num_cls, 1, 1)
                    .to(pred_hm.device)
                )
            else:
                pos_cls_weight = None
            if self.ohem_fp_loss_weight is not None:
                ohem_fp_mask = (target_hm == 0) & (
                    pred_hm > self.ohem_fp_threshold
                )
                neg_cls_weight = torch.ones_like(pred_hm)
                neg_cls_weight[ohem_fp_mask] *= self.ohem_fp_loss_weight
            else:
                neg_cls_weight = None
            if self.use_target_ignore:
                hm_ignore_mask = (
                    torch.ones_like(pred_hm)
                    * (cls_ignore_mask.bool() | target_ignore.bool()).float()
                    if cls_ignore_mask is not None
                    else torch.ones_like(pred_hm) * target_ignore
                )
            else:
                hm_ignore_mask = (
                    torch.ones_like(pred_hm) * cls_ignore_mask
                    if cls_ignore_mask is not None
                    else torch.zeros_like(pred_hm)
                )
            if cls_valid_range_mask is not None:
                hm_ignore_mask = (
                    hm_ignore_mask.bool() | (1.0 - cls_valid_range_mask).bool()
                ).float()
            hm_loss = (
                hm_focal_loss(
                    pred=pred_hm,
                    gt=target_hm,
                    ignore_mask=hm_ignore_mask,
                    pos_cls_weight=pos_cls_weight,
                    neg_cls_weight=neg_cls_weight,
                )
                * loss_weight
            )
        else:
            loss_weight = self.loss_weights["bev_discobj_hm"]
            if cls_valid_range_mask is not None:
                ignore_mask_ = (
                    ignore_mask.bool() | (1.0 - cls_valid_range_mask).bool()
                ).float
            else:
                ignore_mask_ = ignore_mask
            hm_loss = (
                hm_l1_loss(
                    pred_hm,
                    target_hm,
                    weight_mask,
                    ignore_mask_,
                    heatmap_type=None,
                )
                * loss_weight
            )
        result_dict["loss_bev_discobj_hm"] = hm_loss

        # regression map use L1 loss
        for key in self.reg_keys:
            pred_key = "pred_%s" % key
            if pred_key not in pred_dict:
                continue
            if self.use_norm_rot and key == "bev_discobj_rot":
                pred_dict[pred_key] = (
                    2 * torch.sigmoid(pred_dict[pred_key]) - 1
                )
            loss_weight = self.loss_weights[key]
            loss = (
                hm_l1_loss(
                    pred_dict[pred_key],
                    target[key],
                    weight_mask,
                    ignore_mask,
                    heatmap_type=None,
                )
                * loss_weight
            )
            result_dict["loss_%s" % key] = loss

        if self.group_rel_loss:
            loss_bev_discobj_group_rel = (
                self.group_rel_loss(pred_dict, target_dict, pred_hm)[
                    "loss_bev_discobj_group_rel"
                ]
                * self.loss_weights["bev_discobj_group_rel"]
            )
            result_dict[
                "loss_bev_discobj_group_rel"
            ] = loss_bev_discobj_group_rel

        return result_dict


@OBJECT_REGISTRY.register
class ANCBEVDiscreteObjectWithClsLoss(ANCBEVDiscreteObjectLoss):
    """Calculate BEV discrete obj heatmap loss, cls loss and dimension loss.

    Classification and heatmap uses focal loss by default.
    Regression (dimension, rotation, center_offset
    etc.) use L1 loss.

    Args:
        loss_weights: default: None
            Global loss weight for each sub-loss. Default
            weight is 1.0. (e.g. bev_discobj_hm:1, bev_discobj_dim:1,
            bev_discobj_rot:1 etc.).
        use_focal_hm_loss: Use focal loss for heatmap
            otherwise l1 loss.
        use_focal_cls_loss: Use focal loss for classification
            otherwise l1 loss.
        use_softmax_focal_aux_loss:  Use focal loss for classification
            aux loss otherwise CE loss.
        use_target_ignore:  Will not compute the loss of
            labeled ignore regions if true.
        use_norm_rot: whether norm rot loss to [-1,1].
        ohem_fp_loss_weight: Loss weight for ohem fp instance.
        ohem_fp_threshold: Score threshold to select ohem fp instance.
        gt_name: Gt name of current branch.
    Returns:
        a dict contains loss.
    """

    def __init__(
        self,
        loss_weights: Optional[Dict] = None,
        use_focal_hm_loss: bool = True,
        use_focal_cls_loss: bool = False,
        use_softmax_focal_aux_loss: bool = False,
        use_target_ignore: bool = False,
        use_norm_rot: bool = False,
        ohem_fp_loss_weight: float = 1.0,
        ohem_fp_threshold: float = 1.0,
        gt_name: str = "bev_discrete_obj",
    ):
        super(ANCBEVDiscreteObjectWithClsLoss, self).__init__(
            loss_weights,
            use_focal_hm_loss,
            use_target_ignore=use_target_ignore,
            use_norm_rot=use_norm_rot,
            gt_name=gt_name,
        )
        if "bev_discobj_hm_cls" in self.reg_keys:
            self.reg_keys.remove("bev_discobj_hm_cls")
        self.use_focal_cls_loss = use_focal_cls_loss
        self.use_softmax_focal_aux_loss = use_softmax_focal_aux_loss
        self.ohem_fp_loss_weight = ohem_fp_loss_weight
        self.ohem_fp_threshold = ohem_fp_threshold

        assert "bev_discobj_hm_cls_aux" in self.loss_weights
        if self.use_softmax_focal_aux_loss:
            self.cls_aux_loss = FocalLossV2(
                from_logits=False, reduction="mean"
            )
        else:
            self.cls_aux_loss = SoftmaxCELoss(
                dim=1,
                reduction="mean",
            )

    @autocast(enabled=False)
    def forward(self, pred_dict: Dict, target_dict: Dict):
        """Calculate loss between pred and target items.

        Args:
            pred_dict: Predict bev discrete obj output (e.g.,
                pred_bev_discobj_hm, pred_bev_discobj_wh, etc).
            target_dict: Target contains bev discrete obj ground truth
        """
        assert self.gt_name in target_dict
        target = target_dict[self.gt_name]

        pred_hm = pred_dict["pred_bev_discobj_hm"].clone()
        pred_cls = pred_dict["pred_bev_discobj_hm_cls"].clone()
        target_hm = target["bev_discobj_hm"].clone()
        target_cls = target["bev_discobj_hm_cls"].clone()

        assert "bev_discobj_weight_hm" in target
        weight_mask = target["bev_discobj_weight_hm"]
        target_ignore = target["bev_discobj_ignore"]
        if self.use_target_ignore:
            ignore_mask = (
                target_ignore.bool() | (weight_mask == 0).bool()
            ).float()
        else:
            ignore_mask = (weight_mask == 0).float()

        # class valid range mask
        cls_valid_range_mask = target.get("bev_discobj_cls_valid_mask", None)

        pred_hm = sigmoid_and_clip(pred_hm)
        pred_cls = sigmoid_and_clip(pred_cls)
        result_dict = {}

        loss_hm_weight = self.loss_weights["bev_discobj_hm"]
        ohem_fp_mask = (target_hm == 0) & (pred_hm > self.ohem_fp_threshold)
        neg_cls_weight = torch.ones_like(pred_hm)
        neg_cls_weight[ohem_fp_mask] *= self.ohem_fp_loss_weight
        if self.use_focal_hm_loss:
            hm_loss = (
                hm_focal_loss(
                    pred=pred_hm,
                    gt=target_hm,
                    ignore_mask=torch.ones_like(pred_hm) * target_ignore
                    if self.use_target_ignore
                    else torch.zeros_like(pred_hm),
                    neg_cls_weight=neg_cls_weight,
                )
                * loss_hm_weight
            )
        else:
            hm_loss = (
                hm_l1_loss(
                    pred_hm,
                    target_hm,
                    weight_mask,
                    ignore_mask,
                    heatmap_type=None,
                )
                * loss_hm_weight
            )
        result_dict["loss_bev_discobj_hm"] = hm_loss

        loss_cls_weight = self.loss_weights["bev_discobj_hm_cls"]
        if self.use_focal_cls_loss:
            if cls_valid_range_mask is not None:
                ignore_mask_ = (
                    ignore_mask.bool() | (1.0 - cls_valid_range_mask).bool()
                ).float
            else:
                ignore_mask_ = ignore_mask
            cls_loss = (
                hm_focal_loss(
                    pred=pred_cls,
                    gt=target_cls,
                    ignore_mask=ignore_mask_,
                    beta=0.25,
                )
                * loss_cls_weight
            )
        else:
            weight_mask_ = (weight_mask != 0).float()
            if cls_valid_range_mask is not None:
                weight_mask_ = weight_mask_ * cls_valid_range_mask
            cls_loss = (
                F.binary_cross_entropy(
                    input=pred_cls,
                    target=target_cls,
                    reduction="sum",
                    weight=weight_mask_,
                )
                * loss_cls_weight
                / max(weight_mask_.sum(), 1)
            )
        result_dict["loss_bev_discobj_hm_cls"] = cls_loss

        # auxiliary loss
        loss_cls_aux_weight = self.loss_weights["bev_discobj_hm_cls_aux"]
        pred_cls_aux = pred_dict["pred_bev_discobj_hm_cls"].clone()
        _, c, _, _ = pred_cls_aux.size()
        pred_cls_aux = (
            pred_cls_aux.permute(0, 2, 3, 1).contiguous().view(-1, c)
        )

        target_cls_aux = target["bev_discobj_hm_cls"].clone()
        target_cls_aux = (
            target_cls_aux.permute(0, 2, 3, 1).contiguous().view(-1, c)
        )
        avg_factor = max(target_cls_aux.sum(), 1)
        weight_mask_aux = (
            (weight_mask != 0)
            .float()
            .repeat(1, c, 1, 1)
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(-1, c)
        )
        if cls_valid_range_mask is not None:
            weight_mask_aux = weight_mask_aux * cls_valid_range_mask
        if self.use_softmax_focal_aux_loss:
            pred_cls_aux = F.softmax(pred_cls_aux, dim=1)
        cls_aux_loss = self.cls_aux_loss(
            pred_cls_aux, target_cls_aux, weight_mask_aux, avg_factor
        )
        cls_aux_loss = cls_aux_loss * loss_cls_aux_weight
        result_dict["loss_bev_discobj_hm_cls_aux"] = cls_aux_loss

        # regression map use L1 loss
        for key in self.reg_keys:
            pred_key = "pred_%s" % key
            if pred_key not in pred_dict:
                continue
            if self.use_norm_rot and key == "bev_discobj_rot":
                pred_dict[pred_key] = (
                    2 * torch.sigmoid(pred_dict[pred_key]) - 1
                )
            loss_weight = self.loss_weights[key]
            loss = (
                hm_l1_loss(
                    pred_dict[pred_key],
                    target[key],
                    weight_mask,
                    ignore_mask,
                    heatmap_type=None,
                )
                * loss_weight
            )
            result_dict["loss_%s" % key] = loss

        return result_dict
