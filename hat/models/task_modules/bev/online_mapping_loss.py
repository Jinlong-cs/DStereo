# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from hat.models.losses.gaze.loss import WingLoss
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list

__all__ = [
    "ANCOnlineMappingLoss",
    "ANCEmbeddingLoss",
    "ANCOMFocalLoss",
    "ANCCrossPointLoss",
    "ANCDiceLoss",
]


@OBJECT_REGISTRY.register
class ANCOnlineMappingLoss(torch.nn.Module):
    """OnlineMappingLoss, used in online mapping task.

    Args:
        head_groups: the config of all head.
        head_cls_loss: cls loss module of each head.
        instance_loss: instance_loss.
        reg_loss_weight: regression loss weight.
        head_cls_dice_loss: dice loss for cls loss module.
        adaptive_loss_weight: adaptive_loss_weight.
        phi_loss_proportion_r: phi_loss_proportion_r.
        only_cal_cls_loss: only calculate cls loss for aux om loss,
            exclude instance loss, reg loss and so on.
        reg_loss: reg loss type.
        global_ignore_index: global ignore class index.
        cpts_loss: crosspoint loss, used in bev-crosspoint task include
            cpts cls loss and regression loss.
        gt_name: gt name.
        assign_near_instance_cfg: near instance gt assign config
    """

    def __init__(
        self,
        head_groups: Dict,
        head_cls_loss: Dict[str, torch.nn.Module],
        instance_loss: torch.nn.Module,
        reg_loss_weight: float,
        head_cls_dice_loss: torch.nn.Module = None,
        adaptive_loss_weight: bool = False,
        phi_loss_proportion_r: bool = True,
        only_cal_cls_loss: bool = False,
        reg_loss: str = "l1",
        global_ignore_index: int = -1,
        cpts_loss: torch.nn.Module = None,
        gt_name: str = "om_target",
        assign_near_instance_cfg: dict = None,
    ):
        super(ANCOnlineMappingLoss, self).__init__()
        self.head_groups = head_groups
        self.head_cls_loss = head_cls_loss
        self.head_cls_dice_loss = head_cls_dice_loss
        self.instance_loss = instance_loss
        if reg_loss == "wing":
            self.reg_loss = WingLoss(w=5)
        elif reg_loss == "l1":
            self.reg_loss = None
        else:
            raise NotImplementedError
        self.reg_loss_weight = reg_loss_weight
        self.adaptive_loss_weight = adaptive_loss_weight
        self.phi_loss_proportion_r = phi_loss_proportion_r
        self.only_cal_cls_loss = only_cal_cls_loss
        self.global_ignore_index = global_ignore_index
        self.gt_name = gt_name
        self.cpts_loss = cpts_loss
        self.assign_near_instance_cfg = assign_near_instance_cfg

        # generate init scale params
        self.s_params = {}
        if self.adaptive_loss_weight:
            self.s_cls = torch.nn.Parameter(-1.0 * torch.ones(1))
            if self.head_cls_dice_loss is not None:
                self.s_cls_dice = torch.nn.Parameter(-1.0 * torch.ones(1))
            self.s_instance = torch.nn.Parameter(-1.0 * torch.ones(1))
            self.s_r = torch.nn.Parameter(-1.0 * torch.ones(1))
            self.s_angle = torch.nn.Parameter(-1.0 * torch.ones(1))
            if self.cpts_loss is not None:
                self.s_cpts_cls = torch.nn.Parameter(-1.0 * torch.ones(1))
                self.s_cpts_x = torch.nn.Parameter(-1.0 * torch.ones(1))
                self.s_cpts_y = torch.nn.Parameter(-1.0 * torch.ones(1))
                if self.cpts_loss.cpts_dice_loss:
                    self.s_cpts_dice = torch.nn.Parameter(-1.0 * torch.ones(1))
            for head, head_infos in head_groups.items():
                if head_infos.get("multi", False):
                    continue
                self.s_params[head] = torch.nn.Parameter(-1.0 * torch.ones(1))
        self.s_params = torch.nn.ParameterDict(self.s_params)

        # gather group info and corresponding head info
        self.group_heads = {}
        self.group_max_patch = {}
        self.group_cls_loss_type = {}
        self.head_hist = {}
        for head, head_infos in head_groups.items():
            group = head_infos["group"]
            group_head = head_groups[group]
            loss_type = group_head.get("loss_type", "ce_loss")
            max_patch_segment = group_head.get("max_patch_segment", 1)
            multi_head = head_infos.get("multi", False)
            num_class = (
                max([v for _, v in head_infos["cls_remap"].items()]) + 1
            )
            global_auto_class_weight = head_infos.get(
                "global_auto_class_weight", None
            )
            if global_auto_class_weight is not None:
                self.head_hist[head] = torch.ones(
                    num_class, dtype=torch.float64
                )
            if group not in self.group_heads:
                self.group_heads[group] = {"sub_head": []}
            if multi_head:
                self.group_max_patch[group] = max_patch_segment
                self.group_cls_loss_type[group] = loss_type
                self.group_heads[group]["main_head"] = head
            else:
                self.group_heads[group]["sub_head"].append(head)

    def get_reg_loss(self, pred, target, group, positive, num_positive):
        weight_r = positive.unsqueeze(1)
        avg_factor_r = num_positive
        pred_r = pred[f"pred_online_mapping_r_{group}_frame0"][0]
        pred_sin = pred[f"pred_online_mapping_sin_{group}_frame0"][0]
        pred_cos = pred[f"pred_online_mapping_cos_{group}_frame0"][0]
        gt_r = target[self.gt_name][group]["r"].unsqueeze(1)
        gt_sin = target[self.gt_name][group]["sin"].unsqueeze(1)
        gt_cos = target[self.gt_name][group]["cos"].unsqueeze(1)
        if self.reg_loss is not None:
            loss_r_s = self.reg_loss(pred_r, gt_r)
            loss_sin_s = self.reg_loss(pred_sin, gt_sin)
            loss_cos_s = self.reg_loss(pred_cos, gt_cos)
        else:
            loss_r_s = F.l1_loss(
                pred_r,
                gt_r,
                reduction="none",
            )
            loss_sin_s = F.l1_loss(pred_sin, gt_sin, reduction="none")
            loss_cos_s = F.l1_loss(pred_cos, gt_cos, reduction="none")
        loss_r = self.reg_loss_weight * (
            torch.sum(loss_r_s * weight_r) / avg_factor_r
        )

        if self.phi_loss_proportion_r:
            weight_angle = positive.unsqueeze(1)
            weight_angle = weight_angle * torch.clamp(gt_r, 0.0, 0.5)
            avg_factor_angle = num_positive
        else:
            weight_angle = positive.unsqueeze(1)
            avg_factor_angle = num_positive

        loss_sin = self.reg_loss_weight * (
            torch.sum(loss_sin_s * weight_angle) / avg_factor_angle
        )

        loss_cos = self.reg_loss_weight * (
            torch.sum(loss_cos_s * weight_angle) / avg_factor_angle
        )

        loss_angle = loss_sin + loss_cos

        return loss_r, loss_angle

    def get_cls_dice_loss(self, pred, target, group, loss_type):
        pred_dice = pred[f"pred_online_mapping_cls_{group}_frame0"][0]
        if loss_type == "focal_loss":
            # [n,c,patch,h,w]
            pred_dice = torch.sigmoid(pred_dice)[:, 0, ...]
        elif loss_type == "ce_loss":
            # [n,c,patch,h,w]
            pred_dice = torch.softmax(pred_dice, dim=1)[:, 1, ...]
        dice_loss = self.head_cls_dice_loss(
            pred_dice, target[self.gt_name][group]["cls"]
        )
        return dice_loss

    def get_origin_loss(self, pred, target, group):
        res = {}
        group_head = self.group_heads[group]["main_head"]
        loss_cls = self.head_cls_loss[group_head](
            pred[f"pred_online_mapping_cls_{group}_frame0"],
            target[self.gt_name][group]["cls"],
        )["loss_cls"]
        if self.head_cls_dice_loss is not None:
            loss_cls_dice = self.get_cls_dice_loss(
                pred, target, group, self.head_groups[group]["loss_type"]
            )
        # check if enable dilate
        use_dilate = self.head_groups[group_head].get("dilate", True)
        if use_dilate:
            positive = (target[self.gt_name][group]["dilate"] > 0).int()
            num_positive = torch.sum(positive) + 1e-7
            dilate_weight = target[self.gt_name][group]["weight"]
            positive = positive * dilate_weight
            loss_cls = loss_cls * dilate_weight
        else:
            positive = (target[self.gt_name][group]["cls"] > 0).int()
            num_positive = torch.sum(positive) + 1e-7
        loss_cls = torch.mean(loss_cls)

        if not self.only_cal_cls_loss:
            # assign_near_instance only work on lane and
            # a grid must support at most two lanes
            if (
                self.assign_near_instance_cfg is not None
                and group == "lane"
                and self.group_max_patch[group] == 2
            ):
                ebd_pred = torch.chunk(
                    pred[f"pred_online_mapping_instance_{group}_frame0"][0],
                    chunks=2,
                    dim=2,
                )
                ebd_gt = torch.chunk(
                    target[self.gt_name][group]["instance"], chunks=2, dim=1
                )
                loss_instance = self.instance_loss(
                    ebd_pred,
                    ebd_gt,
                    ch_weight=self.assign_near_instance_cfg[
                        "ebd_channel_weight"
                    ],
                )
            else:
                loss_instance = self.instance_loss(
                    pred[f"pred_online_mapping_instance_{group}_frame0"],
                    target[self.gt_name][group]["instance"],
                )

            # compute regression loss
            loss_r, loss_angle = self.get_reg_loss(
                pred, target, group, positive, num_positive
            )
        # update loss
        res.update({f"loss_cls_{group}": loss_cls})
        if not self.only_cal_cls_loss:
            if self.head_cls_dice_loss is not None:
                res.update({f"loss_cls_dice_{group}": loss_cls_dice})
            res.update(
                {f"loss_instance_inner_{group}": loss_instance["loss_inner"]}
            )
            res.update(
                {f"loss_instance_outer_{group}": loss_instance["loss_outer"]}
            )
            res.update({f"loss_r_{group}": loss_r})
            res.update({f"loss_angle_{group}": loss_angle})

        # compute sub task attribute loss
        for head in self.group_heads[group]["sub_head"]:
            pred_key = f"pred_online_mapping_{head}_{group}_frame0"
            pred_v = pred[pred_key][0]
            target_v = target[self.gt_name][group][head].long()
            positive_sub = (target_v != self.global_ignore_index).int()
            if (
                "main_vcs_roi" in target[self.gt_name][group]
                and "only_focus_main_roi" in self.head_groups[head]
                and self.head_groups[head]["only_focus_main_roi"] is True
            ):
                positive_sub *= (
                    target[self.gt_name][group]["main_vcs_roi"] > 0
                ).int()

            avg_factor = torch.sum(positive_sub) + 1e-7
            global_weight = self.get_global_auto_cls_weight(head, target_v)
            head_loss = self.head_cls_loss[head](pred_v, target_v)["loss_cls"]
            head_loss = (
                torch.sum(head_loss * positive_sub * global_weight)
                / avg_factor
            )
            head_loss_weight = self.head_groups[head].get("loss_weight", 1.0)
            head_loss *= head_loss_weight
            if self.adaptive_loss_weight:
                head_loss = torch.exp(-self.s_params[head])[0] * head_loss
                loss_s_head = torch.abs(self.s_params[head])
                res.update({f"loss_s_{head}_{group}": loss_s_head})
            res.update({f"loss_{head}_{group}": head_loss})
        return res

    def get_global_auto_cls_weight(self, head, data):
        # TODO: fan.zhao, as a generic feature in HAT.
        if head not in self.head_hist:
            return 1.0
        num_class = self.head_hist[head].shape[0]
        mask = data >= 0
        self.head_hist[head] = self.head_hist[head].to(data.device)
        hist = torch.nn.functional.one_hot(data[mask], num_class).sum(dim=-2)
        self.head_hist[head] += hist
        min_w, max_w = self.head_groups[head]["global_auto_class_weight"]
        weights_table = torch.max(self.head_hist[head]) / self.head_hist[head]
        weights_table = torch.clamp(weights_table * min_w, min_w, max_w)
        weights = weights_table[data.long()]
        return weights

    def _get_cost(self, pred, target, group):
        # compute classification loss according to loss type
        group_head = self.group_heads[group]["main_head"]
        cost_cls = self.head_cls_loss[group_head](
            pred[f"pred_online_mapping_cls_{group}_frame0"],
            target["cls"].long(),
        )["loss_cls"].unsqueeze(1)

        if not self.only_cal_cls_loss:
            cost_r = F.l1_loss(
                pred[f"pred_online_mapping_r_{group}_frame0"][0],
                target["r"].unsqueeze(1),
                reduction="none",
            )
            cost_sin = F.l1_loss(
                pred[f"pred_online_mapping_sin_{group}_frame0"][0],
                target["sin"].unsqueeze(1),
                reduction="none",
            )
            cost_cos = F.l1_loss(
                pred[f"pred_online_mapping_cos_{group}_frame0"][0],
                target["cos"].unsqueeze(1),
                reduction="none",
            )
        if self.only_cal_cls_loss:
            cost = cost_cls
        else:
            cost = cost_cls + cost_r + cost_sin + cost_cos
        # add sub task cost
        for head in self.group_heads[group]["sub_head"]:
            add_cost = self.head_groups[head].get("add_cost", False)
            if not add_cost:
                continue
            pred_key = f"pred_online_mapping_{head}_{group}_frame0"
            pred_v = pred[pred_key][0]
            target_v = target[head].long()
            head_cost = self.head_cls_loss[head](pred_v, target_v)[
                "loss_cls"
            ].unsqueeze(1)
            cost += head_cost

        return cost

    def get_cost(self, pred_0, pred_1, target_0, target_1, group):
        cost_0 = self._get_cost(pred_0, target_0, group)
        cost_1 = self._get_cost(pred_1, target_1, group)
        cost = cost_0 + cost_1
        return cost

    def forward(self, pred: Tuple, target: Tuple[Dict]) -> Dict:
        res = {}
        default_group = list(self.group_heads.keys())[0]
        default_key = f"pred_online_mapping_cls_{default_group}_frame0"
        default_device = pred[default_key][0].device
        loss_cls_all = torch.tensor(
            0.0,
            dtype=torch.float64,
            device=default_device,
        )
        if self.head_cls_dice_loss is not None:
            loss_cls_dice_all = torch.tensor(
                0.0,
                dtype=torch.float64,
                device=default_device,
            )
        loss_instance_inner_all = torch.tensor(
            0.0,
            dtype=torch.float64,
            device=default_device,
        )
        loss_instance_outer_all = torch.tensor(
            0.0,
            dtype=torch.float64,
            device=default_device,
        )
        loss_r_all = torch.tensor(
            0.0,
            dtype=torch.float64,
            device=default_device,
        )
        loss_angle_all = torch.tensor(
            0.0,
            dtype=torch.float64,
            device=default_device,
        )

        for group, max_patch in self.group_max_patch.items():
            if max_patch == 2:
                for k, v in pred.items():
                    if group not in k:
                        continue
                    slice1, slice2 = torch.chunk(v[0], 2, dim=1)
                    pred[k] = [
                        torch.cat(
                            [slice1.unsqueeze(2), slice2.unsqueeze(2)], dim=2
                        )
                    ]
            group_res = self.get_origin_loss(pred, target, group)
            res.update(group_res)
            group_head = self.group_heads[group]["main_head"]
            group_loss_weight = self.head_groups[group_head].get(
                "loss_weight", 1.0
            )
            loss_cls_all += group_res[f"loss_cls_{group}"] * group_loss_weight

            if not self.only_cal_cls_loss:
                if self.head_cls_dice_loss is not None:
                    loss_cls_dice_all += (
                        group_res[f"loss_cls_dice_{group}"] * group_loss_weight
                    )
                loss_instance_inner_all += (
                    group_res[f"loss_instance_inner_{group}"]
                    * group_loss_weight
                )
                loss_instance_outer_all += (
                    group_res[f"loss_instance_outer_{group}"]
                    * group_loss_weight
                )
                loss_r_all += group_res[f"loss_r_{group}"] * group_loss_weight
                loss_angle_all += (
                    group_res[f"loss_angle_{group}"] * group_loss_weight
                )

        # adjust loss with learned scale
        if self.adaptive_loss_weight:
            loss_cls_all = torch.exp(-self.s_cls)[0] * loss_cls_all
            loss_s_cls = torch.abs(self.s_cls)
            res.update({"loss_s_cls": loss_s_cls})
            if not self.only_cal_cls_loss:
                if self.head_cls_dice_loss is not None:
                    loss_cls_dice_all = (
                        torch.exp(-self.s_cls_dice)[0] * loss_cls_dice_all
                    )
                    # weaken the scale constraint on the dice loss
                    loss_s_cls_dice = torch.abs(2 * self.s_cls_dice)
                    res.update({"loss_s_cls_dice": loss_s_cls_dice})
                loss_instance_inner_all = (
                    torch.exp(-self.s_instance)[0] * loss_instance_inner_all
                )
                loss_instance_outer_all = (
                    torch.exp(-self.s_instance)[0] * loss_instance_outer_all
                )
                loss_s_instance = torch.abs(self.s_instance)
                res.update({"loss_s_instance": loss_s_instance})
                loss_r_all = torch.exp(-self.s_r)[0] * loss_r_all
                loss_s_r = torch.abs(self.s_r)
                res.update({"loss_s_r": loss_s_r})
                loss_angle_all = torch.exp(-self.s_angle)[0] * loss_angle_all
                loss_s_angle = torch.abs(self.s_angle)
                res.update({"loss_s_angle": loss_s_angle})
        res.update({"loss_cls_all": loss_cls_all})
        if not self.only_cal_cls_loss:
            if self.head_cls_dice_loss is not None:
                res.update({"loss_cls_dice_all": loss_cls_dice_all})
            res.update({"loss_instance_inner_all": loss_instance_inner_all})
            res.update({"loss_instance_outer_all": loss_instance_outer_all})
            res.update({"loss_r_all": loss_r_all})
            res.update({"loss_angle_all": loss_angle_all})

        if self.cpts_loss is not None:
            cpts_loss = self.cpts_loss(pred, target)
            cpts_cls_loss = cpts_loss["loss_cls_all"]
            cpts_x_loss = cpts_loss["loss_x_all"]
            cpts_y_loss = cpts_loss["loss_y_all"]
            if self.cpts_loss.cpts_dice_loss:
                cpts_dice_loss = cpts_loss["loss_dice_all"]
            if self.adaptive_loss_weight:
                cpts_cls_loss = torch.exp(-self.s_cpts_cls) * cpts_cls_loss
                loss_s_head = torch.abs(self.s_cpts_cls)
                res.update({"loss_s_cpts_cls": loss_s_head})
                cpts_x_loss = torch.exp(-self.s_cpts_x)[0] * cpts_x_loss
                cpts_y_loss = torch.exp(-self.s_cpts_y)[0] * cpts_y_loss
                loss_s_x = torch.abs(self.s_cpts_x)
                res.update({"loss_s_cpts_x": loss_s_x})
                loss_s_y = torch.abs(self.s_cpts_y)
                res.update({"loss_s_cpts_y": loss_s_y})
                if self.cpts_loss.cpts_dice_loss:
                    cpts_dice_loss = (
                        torch.exp(-self.s_cpts_dice)[0] * cpts_dice_loss
                    )
                    loss_s_cpts_dice = torch.abs(self.s_cpts_dice)
                    res.update({"loss_s_cpts_dice": loss_s_cpts_dice})
            res.update({"loss_cpts_cls": cpts_cls_loss})
            res.update({"loss_cpts_x": cpts_x_loss})
            res.update({"loss_cpts_y": cpts_y_loss})
            if self.cpts_loss.cpts_dice_loss:
                res.update({"loss_cpts_dice": cpts_y_loss})
        return res


@OBJECT_REGISTRY.register
class ANCDiceLoss(torch.nn.Module):
    """DiceLoss, used in online mapping cls task.

    Args:
        smooth: Smoothing value to avoid division by zero.

    """

    def __init__(self, smooth=1.0):
        super(ANCDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        intersection = torch.sum(prediction * target)
        union = torch.sum(prediction) + torch.sum(target) + self.smooth

        dice_coefficient = (2.0 * intersection + self.smooth) / union
        loss = 1.0 - dice_coefficient

        return loss


@OBJECT_REGISTRY.register
class ANCEmbeddingLoss(torch.nn.Module):
    """EmbeddingLoss used in online mapping task.

    # TODO(yueyu.wang, senyao.du): improve docstring.

    Args:
        embed_dim: embedding dimensions.
        delta_v: delta_v
        delta_d: delta_d
        inner_loss_weight: loss_weight of class inner
        outer_loss_weight: loss_weight of class outer
    """

    def __init__(
        self,
        embed_dim: int = 4,
        delta_v: float = 0.5,
        delta_d: float = 4.0,
        inner_loss_weight: float = 1.0,
        outer_loss_weight: float = 1.0,
    ):
        super(ANCEmbeddingLoss, self).__init__()
        self.embed_dim = embed_dim
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.inner_loss_weight = inner_loss_weight
        self.outer_loss_weight = outer_loss_weight

    @autocast(enabled=False)
    def forward(self, pred, target, ch_weight: Optional[list] = None):
        """Compute embedding loss.

        Args:
            pred: Model prediction.
            target: Ground truth.
            ch_weight: Embedding loss channel weight.

        """
        var_loss = 0
        dist_loss = 0
        preds = _as_list(pred)
        target = _as_list(target)
        for idx, (each_pred, each_gt) in enumerate(zip(preds, target)):
            var_loss_, dist_loss_ = self.discriminative_loss(
                each_pred.float(), each_gt
            )
            if ch_weight is None:
                var_loss += self.inner_loss_weight * var_loss_
                dist_loss += self.outer_loss_weight * dist_loss_
            else:
                assert len(ch_weight) == len(pred)
                var_loss += self.inner_loss_weight * var_loss_ * ch_weight[idx]
                dist_loss += (
                    self.outer_loss_weight * dist_loss_ * ch_weight[idx]
                )
        return {"loss_inner": var_loss, "loss_outer": dist_loss}

    def discriminative_loss(self, embedding, seg_gt):
        batch_size = embedding.shape[0]

        var_loss = torch.tensor(
            0, dtype=embedding.dtype, device=embedding.device
        )
        dist_loss = torch.tensor(
            0, dtype=embedding.dtype, device=embedding.device
        )

        for b in range(batch_size):
            embedding_b = embedding[b]  # (embed_dim, H, W)
            seg_gt_b = seg_gt[b]

            labels = torch.unique(seg_gt_b)
            labels = labels[labels != 0]
            num_lanes = len(labels)
            if num_lanes == 0:
                # please refer to issue here: https://github.com/harryhan618/LaneNet/issues/12  # noqa
                _nonsense = embedding.sum()
                _zero = torch.zeros_like(_nonsense)
                var_loss = var_loss + _nonsense * _zero
                dist_loss = dist_loss + _nonsense * _zero
                continue

            centroid_mean = []
            for lane_idx in labels:
                seg_mask_i = seg_gt_b == lane_idx
                if not seg_mask_i.any():
                    continue
                embedding_i = embedding_b[:, seg_mask_i]

                mean_i = torch.mean(embedding_i, dim=1)
                centroid_mean.append(mean_i)

                # ---------- var_loss -------------
                var_loss = (
                    var_loss
                    + torch.mean(
                        F.relu(
                            torch.norm(
                                embedding_i
                                - mean_i.reshape(self.embed_dim, 1),
                                dim=0,
                            )
                            - self.delta_v
                        )
                        ** 2
                    )
                    / num_lanes
                )
            centroid_mean = torch.stack(centroid_mean)  # (n_lane, embed_dim)

            if num_lanes > 1:
                centroid_mean1 = centroid_mean.reshape(-1, 1, self.embed_dim)
                centroid_mean2 = centroid_mean.reshape(1, -1, self.embed_dim)
                dist = torch.norm(
                    centroid_mean1 - centroid_mean2, dim=2
                )  # shape (num_lanes, num_lanes)
                dist = (
                    dist
                    + torch.eye(
                        num_lanes, dtype=dist.dtype, device=dist.device
                    )
                    * self.delta_d
                )  # diagonal elements are 0, now mask above delta_d

                # divided by two for double calculated loss above, for implementation convenience   # noqa
                dist_loss = (
                    dist_loss
                    + torch.sum(F.relu(-dist + self.delta_d) ** 2)
                    / (num_lanes * (num_lanes - 1))
                    / 2
                )

        var_loss = var_loss / batch_size
        dist_loss = dist_loss / batch_size
        return var_loss, dist_loss


@OBJECT_REGISTRY.register
class ANCOMFocalLoss(torch.nn.Module):
    """Focal loss with N*C*(...).

    Args:
        loss_name: The key of loss in return dict.
        num_classes: Num_classes including background, C+1, C is number
            of foreground categories.
        alpha: A weighting factor for pos-sample, (1-alpha) is for
            neg-sample.
        gamma: Gamma used in focal loss to compress the contribution
            of easy examples.
        loss_weight: Global weight of loss. Defaults is 1.0.
        eps: A small value to avoid zero denominator.
        gt_offset: substract this value to remap gt label.
        enable_ignore: if enable ignore negative label.
        reduction: The method used to reduce the loss. Options are
            [`none`, `mean`, `sum`].

    Returns:
        dict: A dict containing the calculated loss, the key of loss is
        loss_name.
    """

    def __init__(
        self,
        loss_name: str,
        num_classes: int,
        alpha: float = 0.25,
        gamma: float = 2.0,
        loss_weight: float = 1.0,
        eps: float = 1e-12,
        gt_offset: float = 0.0,
        enable_ignore: bool = True,
        reduction: str = "mean",
    ):
        super(ANCOMFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.gt_offset = gt_offset
        self.enable_ignore = enable_ignore
        self.loss_name = loss_name

    @autocast(enabled=False)
    def forward(
        self,
        pred,
        target,
    ):
        # extract from list
        pred = _as_list(pred)
        pred = pred[0]
        # convert to float32 while using amp
        pred = pred.float().sigmoid()
        # align shape
        multi_dims = pred.ndim > 2
        if multi_dims:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = pred.shape[1]
            pred = pred.permute(0, *range(2, pred.ndim), 1)
            pred = pred.contiguous().reshape(-1, c)
        target_shape = target.shape
        mask = (target >= 0).reshape(-1, 1)
        target = target.reshape(-1) - self.gt_offset

        target[target < 0] = self.num_classes - 1
        one_hot = F.one_hot(target, self.num_classes)  # N x C+1
        one_hot = one_hot[..., : self.num_classes - 1]  # N x C
        pt = torch.where(torch.eq(one_hot, 1), pred, 1 - pred)
        t = torch.ones_like(one_hot)
        at = torch.where(
            torch.eq(one_hot, 1), self.alpha * t, (1 - self.alpha) * t
        )
        loss = (
            -at
            * torch.pow((1 - pt), self.gamma)
            * torch.log(torch.minimum(pt + self.eps, t))
        )
        if self.enable_ignore:
            loss = loss * mask
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        else:
            loss = torch.mean(loss, dim=-1)
            loss = loss.reshape(target_shape)

        result_dict = {}
        result_dict[self.loss_name] = loss * self.loss_weight
        return result_dict


@OBJECT_REGISTRY.register
class ANCCrossPointLoss(torch.nn.Module):
    """CrossPointLoss, used in bev-crosspoint task.

    Args:
        group_loss_weight: Loss weight of each group.
        reg_loss: Regression loss type.
        cls_loss_weight: Class loss weight.
        focal_loss_beta: Args beta for focal loss.
        cpts_pos_weight: pos weight in focal loss
        reg_loss_weight: Regression ignore cls label.
        subtype_loss_weight: Subtype class loss weight.
        empty_sample_weight: empty frame loss weight.
        ignore_index: Ignored class id in train and val stage.
        aux_om_loss: Online mapping loss object.
        gt_name: gt name.
        cpts_dice_loss: Dice loss for cpts cls.
    """

    def __init__(
        self,
        group_loss_weight: Dict[str, float],
        reg_loss: torch.nn.Module,
        cls_loss_weight: float,
        focal_loss_beta: float,
        cpts_pos_weight: float,
        reg_loss_weight: float,
        subtype_loss_weight: dict,
        empty_sample_weight: dict = 1,
        ignore_index: int = 255,
        aux_om_loss: Optional[object] = None,
        gt_name: str = "crosspoint_target",
        cpts_dice_loss: bool = False,
    ):
        super(ANCCrossPointLoss, self).__init__()
        self.group_loss_weight = group_loss_weight
        self.reg_loss = reg_loss
        self.cls_loss_weight = cls_loss_weight
        self.reg_loss_weight = reg_loss_weight
        self.subtype_loss_weight = subtype_loss_weight
        self.focal_loss_beta = focal_loss_beta
        self.cpts_pos_weight = cpts_pos_weight
        self.empty_sample_weight = empty_sample_weight
        self.ignore_index = ignore_index
        self.aux_om_loss = aux_om_loss
        self.gt_name = gt_name
        self.cpts_dice_loss = cpts_dice_loss

    def _heatmap_focal_loss(
        self,
        pred,
        gt,
        ignore_mask,
        weight_mask,
        gamma=2,
        beta=1,
        alpha=2,
        cpts_pos_weight=1.0,
    ):
        pos_mask = (gt == 1) * (1.0 - ignore_mask)
        neg_mask = (gt < 1) * (1.0 - ignore_mask)

        neg_weights = torch.pow(1 - gt, alpha)
        pos_loss = (
            torch.log(pred)
            * torch.pow(1 - pred, gamma)
            * pos_mask
            * weight_mask
        ) * cpts_pos_weight
        neg_loss = (
            beta * torch.log(1 - pred) * torch.pow(pred, gamma) * neg_mask
        )
        neg_loss = neg_loss * neg_weights * weight_mask

        num_pos = pos_mask.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        num_pos = torch.max(num_pos, torch.ones_like(num_pos))
        loss = -(pos_loss + neg_loss) / num_pos
        return loss

    def _sigmoid_and_clip(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def get_origin_loss(self, pred, target, group):
        res = {}
        ignore_mask = (
            target[self.gt_name][group]["cls"] == self.ignore_index
        ).float()
        pred_hm = pred[f"pred_crosspoint_cls_{group}_frame0"][0]
        pred_hm = self._sigmoid_and_clip(pred_hm)
        subtype_weight_mask = torch.ones_like(pred_hm)
        for idx, subtype_weight in enumerate(self.subtype_loss_weight[group]):
            subtype_weight_mask[:, idx, :, :] *= subtype_weight
        gt_prob, _ = torch.max(target[self.gt_name][group]["cls"], 1)
        positive_mask = (gt_prob == 1).int() * (
            gt_prob != self.ignore_index
        ).int()
        num_positive = torch.sum(positive_mask) + 1e-7
        if num_positive < 1:
            sample_weight = self.empty_sample_weight
        else:
            sample_weight = 1
        loss_cls = (
            self.cls_loss_weight
            * self._heatmap_focal_loss(
                pred_hm,
                target[self.gt_name][group]["cls"],
                ignore_mask,
                subtype_weight_mask,
                beta=self.focal_loss_beta,
                cpts_pos_weight=self.cpts_pos_weight,
            )
            * sample_weight
        )

        cpts_dice_loss = self.cpts_dice_loss
        if cpts_dice_loss:
            pred_dice = pred_hm * (1.0 - ignore_mask) * subtype_weight_mask
            gt_dice = (
                target[self.gt_name][group]["cls"]
                * (1.0 - ignore_mask)
                * subtype_weight_mask
            )
            intersection = torch.sum(pred_dice * gt_dice)
            union = torch.sum(pred_dice) + torch.sum(gt_dice) + 1.0
            dice_coefficient = (2.0 * intersection + 1.0) / union
            dice_loss = 1.0 - dice_coefficient
            dice_loss = dice_loss * 15

        loss_x = (
            self.reg_loss_weight
            * (
                torch.sum(
                    self.reg_loss(
                        pred[f"pred_crosspoint_x_{group}_frame0"][0].sigmoid(),
                        target[self.gt_name][group]["x"],
                    )
                    * positive_mask
                )
                / num_positive
            )
            * sample_weight
        )

        loss_y = (
            self.reg_loss_weight
            * (
                torch.sum(
                    self.reg_loss(
                        pred[f"pred_crosspoint_y_{group}_frame0"][0].sigmoid(),
                        target[self.gt_name][group]["y"],
                    )
                    * positive_mask
                )
                / num_positive
            )
            * sample_weight
        )

        res.update({f"loss_cls_{group}": loss_cls})
        res.update({f"loss_x_{group}": loss_x})
        res.update({f"loss_y_{group}": loss_y})
        if cpts_dice_loss:
            res.update({f"loss_dice_{group}": dice_loss})
        return res

    def forward(self, pred: Tuple, target: Tuple[Dict]) -> Dict:
        res = {}
        loss_cls_all = 0
        loss_x_all = 0
        loss_y_all = 0
        loss_dice_all = 0

        for group in self.group_loss_weight:
            group_res = self.get_origin_loss(pred, target, group)
            res.update(group_res)
            loss_cls_all += (
                group_res[f"loss_cls_{group}"] * self.group_loss_weight[group]
            )
            loss_x_all += (
                group_res[f"loss_x_{group}"] * self.group_loss_weight[group]
            )
            loss_y_all += (
                group_res[f"loss_y_{group}"] * self.group_loss_weight[group]
            )
            if self.cpts_dice_loss:
                loss_dice_all += (
                    group_res[f"loss_dice_{group}"]
                    * self.group_loss_weight[group]
                )

        if self.aux_om_loss is not None:
            om_loss = self.aux_om_loss(pred, target)
            res.update({"aux_om_loss_cls": om_loss["loss_cls_all"]})

        res.update({"loss_cls_all": loss_cls_all})
        res.update({"loss_x_all": loss_x_all})
        res.update({"loss_y_all": loss_y_all})
        if self.cpts_dice_loss:
            res.update({"loss_dice_all": loss_dice_all})
        return res
