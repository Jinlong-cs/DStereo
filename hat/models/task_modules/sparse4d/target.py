import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from hat.registry import OBJECT_REGISTRY

__all__ = ["SparseBox3DTarget"]

X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ = list(range(11))
YAW = 6


@OBJECT_REGISTRY.register
class SparseBox3DTarget(object):
    def __init__(
        self,
        cls_weight=2.0,
        alpha=0.25,
        gamma=2,
        eps=1e-12,
        box_weight=0.25,
        reg_weights=None,
        cls_wise_reg_weights=None,
        ignore_reg_weight=0.0,
        ignore_cls_weight=0.0,
        num_dn_groups=0,
        dn_noise_scale=0.5,
        max_dn_gt=30,
        add_neg_dn=True,
    ):
        super(SparseBox3DTarget, self).__init__()
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.reg_weights = reg_weights
        if self.reg_weights is None:
            self.reg_weights = [1.0] * 8 + [0.0] * 2
        self.cls_wise_reg_weights = cls_wise_reg_weights
        self.ignore_reg_weight = ignore_reg_weight
        self.ignore_cls_weight = ignore_cls_weight
        self.num_dn_groups = num_dn_groups
        self.dn_noise_scale = dn_noise_scale
        self.max_dn_gt = max_dn_gt
        self.add_neg_dn = add_neg_dn

    def encode_reg_target(self, box_target, device=None):
        outputs = []
        for box in box_target:
            output = torch.cat(
                [
                    box[..., [X, Y, Z]],
                    box[..., [W, L, H]].log(),
                    torch.sin(box[..., YAW]).unsqueeze(-1),
                    torch.cos(box[..., YAW]).unsqueeze(-1),
                    torch.where(
                        torch.isnan(box[..., YAW + 1 :]),
                        box.new_tensor(0),
                        box[..., YAW + 1 :],
                    ),
                ],
                dim=-1,
            )
            if device is not None:
                output = output.to(device=device)
            outputs.append(output)
        return outputs

    def __call__(
        self,
        cls_pred,
        box_pred,
        cls_target,
        box_target,
        gt_ignore=None,
    ):
        bs, num_pred, num_cls = cls_pred.shape

        cls_cost = self._cls_cost(cls_pred, cls_target)

        box_target = self.encode_reg_target(box_target, box_pred.device)

        instance_reg_weights = []
        for i in range(len(box_target)):
            weights = torch.logical_not(box_target[i].isnan()).to(
                dtype=box_target[i].dtype
            )
            if self.cls_wise_reg_weights is not None:
                for cls, weight in self.cls_wise_reg_weights.items():
                    weights = torch.where(
                        (cls_target[i] == cls)[:, None],
                        weights.new_tensor(weight),
                        weights,
                    )
            instance_reg_weights.append(weights)

        box_cost = self._box_cost(box_pred, box_target, instance_reg_weights)

        indices = []
        for i in range(bs):
            cost = (cls_cost[i] + box_cost[i]).detach().cpu().numpy()
            cost = np.where(np.isneginf(cost) | np.isnan(cost), 1e8, cost)
            indices.append(
                [
                    cls_pred.new_tensor(x, dtype=torch.int64)
                    for x in linear_sum_assignment(cost)
                ]
            )

        output_cls_target = cls_target[0].new_ones([bs, num_pred]) * num_cls
        output_box_target = box_pred.new_zeros(box_pred.shape)
        output_reg_weights = box_pred.new_zeros(box_pred.shape)
        ignore = box_pred.new_zeros([bs, num_pred])
        for i, (pred_idx, target_idx) in enumerate(indices):
            if len(cls_target[i]) == 0:
                continue
            output_cls_target[i, pred_idx] = cls_target[i][target_idx]
            output_box_target[i, pred_idx] = box_target[i][target_idx]
            output_reg_weights[i, pred_idx] = instance_reg_weights[i][
                target_idx
            ]
            if gt_ignore is not None and len(gt_ignore) != 0:
                ignore[i, pred_idx] = gt_ignore[i][target_idx].to(
                    dtype=ignore.dtype
                )

        output_reg_weights = torch.where(
            ignore[..., None] == 0,
            output_reg_weights,
            output_reg_weights * self.ignore_reg_weight,
        )
        output_cls_weights = ignore.new_ones(ignore.shape)
        output_cls_weights = torch.where(
            ignore == 0,
            output_cls_weights,
            output_cls_weights * self.ignore_cls_weight,
        )

        return (
            output_cls_target,
            output_box_target,
            output_cls_weights,
            output_reg_weights,
        )

    def _cls_cost(self, cls_pred, cls_target):
        bs = cls_pred.shape[0]
        cls_pred = cls_pred.sigmoid()
        cost = []
        for i in range(bs):
            neg_cost = (
                -(1 - cls_pred[i] + self.eps).log()
                * (1 - self.alpha)
                * cls_pred[i].pow(self.gamma)
            )
            pos_cost = (
                -(cls_pred[i] + self.eps).log()
                * self.alpha
                * (1 - cls_pred[i]).pow(self.gamma)
            )
            cost.append(
                (pos_cost[:, cls_target[i]] - neg_cost[:, cls_target[i]])
                * self.cls_weight
            )
        return cost

    def _box_cost(self, box_pred, box_target, instance_reg_weights):
        bs = box_pred.shape[0]
        cost = []
        for i in range(bs):
            cost.append(
                torch.sum(
                    torch.abs(box_pred[i, :, None] - box_target[i][None])
                    * instance_reg_weights[i][None]
                    * box_pred.new_tensor(self.reg_weights),
                    dim=-1,
                )
                * self.box_weight
            )
        return cost

    def get_dn_anchors(self, cls_target, box_target, gt_ignore=None):
        if self.num_dn_groups <= 0:
            return None

        # put all non-ignore gt to front and delete the gt out of max_dn_gt
        if gt_ignore is not None:
            cls_target = [
                torch.where(gt_ignore[i] == 0, x, x.new_tensor(-2))
                for i, x in enumerate(cls_target)
            ]
            cls_target = [
                torch.cat([x[gt_ignore[i] == 0], x[gt_ignore[i] != 0]], dim=0)
                for i, x in enumerate(cls_target)
            ]
            box_target = [
                torch.cat([x[gt_ignore[i] == 0], x[gt_ignore[i] != 0]], dim=0)
                for i, x in enumerate(box_target)
            ]
        if self.max_dn_gt > 0:
            cls_target = [x[: self.max_dn_gt] for x in cls_target]
            box_target = [x[: self.max_dn_gt] for x in box_target]

        max_dn_gt = max([len(x) for x in cls_target])
        cls_target = torch.stack(
            [
                F.pad(x, (0, max_dn_gt - x.shape[0]), value=-1)
                for x in cls_target
            ]
        )
        box_target = torch.stack(
            self.encode_reg_target(
                [
                    F.pad(x, (0, 0, 0, max_dn_gt - x.shape[0]))
                    for x in box_target
                ]
            )
        )
        box_target = torch.where(
            cls_target[..., None] == -1, box_target.new_tensor(0), box_target
        )
        bs, num_gt, state_dims = box_target.shape
        if self.num_dn_groups > 1:
            cls_target = cls_target.tile(self.num_dn_groups, 1)
            box_target = box_target.tile(self.num_dn_groups, 1, 1)

        noise = torch.rand_like(box_target) * 2 - 1
        noise *= box_target.new_tensor(self.dn_noise_scale)
        dn_anchor = box_target + noise
        if self.add_neg_dn:
            noise_neg = torch.rand_like(box_target) + 1
            flag = torch.where(
                torch.rand_like(box_target) > 0.5,
                noise_neg.new_tensor(1),
                noise_neg.new_tensor(-1),
            )
            noise_neg *= flag
            noise_neg *= box_target.new_tensor(self.dn_noise_scale)
            dn_anchor = torch.cat([dn_anchor, box_target + noise_neg], dim=1)
            num_gt *= 2

        box_cost = self._box_cost(
            dn_anchor, box_target, torch.ones_like(box_target)
        )
        dn_box_target = torch.zeros_like(dn_anchor)
        dn_cls_target = -torch.ones_like(cls_target) * 3
        if self.add_neg_dn:
            dn_cls_target = torch.cat([dn_cls_target, dn_cls_target], dim=1)

        for i in range(dn_anchor.shape[0]):
            cost = box_cost[i].cpu().numpy()
            anchor_idx, gt_idx = linear_sum_assignment(cost)
            anchor_idx = dn_anchor.new_tensor(anchor_idx, dtype=torch.int64)
            gt_idx = dn_anchor.new_tensor(gt_idx, dtype=torch.int64)
            dn_box_target[i, anchor_idx] = box_target[i, gt_idx]
            dn_cls_target[i, anchor_idx] = cls_target[i, gt_idx]
        dn_anchor = (
            dn_anchor.reshape(self.num_dn_groups, bs, num_gt, state_dims)
            .permute(1, 0, 2, 3)
            .flatten(1, 2)
        )
        dn_box_target = (
            dn_box_target.reshape(self.num_dn_groups, bs, num_gt, state_dims)
            .permute(1, 0, 2, 3)
            .flatten(1, 2)
        )
        dn_cls_target = (
            dn_cls_target.reshape(self.num_dn_groups, bs, num_gt)
            .permute(1, 0, 2)
            .flatten(1)
        )
        valid_mask = dn_cls_target >= 0
        if self.add_neg_dn:
            cls_target = (
                torch.cat([cls_target, cls_target], dim=1)
                .reshape(self.num_dn_groups, bs, num_gt)
                .permute(1, 0, 2)
                .flatten(1)
            )
            valid_mask = torch.logical_or(
                valid_mask, ((cls_target >= 0) & (dn_cls_target == -3))
            )
        attn_mask = dn_box_target.new_ones(
            num_gt * self.num_dn_groups, num_gt * self.num_dn_groups
        )
        for i in range(self.num_dn_groups):
            start = num_gt * i
            end = start + num_gt
            attn_mask[start:end, start:end] = 0
        attn_mask = attn_mask == 1
        return dn_anchor, dn_box_target, dn_cls_target, attn_mask, valid_mask
