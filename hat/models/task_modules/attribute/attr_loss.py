from typing import List

import torch
import torch.nn.functional as F

from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class AttrMultiLabelLoss(torch.nn.CrossEntropyLoss):
    """Multi-label loss for attribute classification.

    Args:
        attr_type_list (List): List of all types.
        attr_type_numcls (List): Number of categories for each type.
        ignore_idx (List): The index of the category to be ignored.

    Returns:
        torch.Tensor: loss value
    """

    def __init__(
        self,
        attr_type_list: List,
        attr_type_numcls: List,
        ignore_idx: List = None,
    ):
        super(AttrMultiLabelLoss, self).__init__()
        self.attr_type_list = attr_type_list
        self.attr_type_numcls = attr_type_numcls
        self.ignore_idx = ignore_idx

    def get_pred_target(self, idx, pred, target):
        if idx == 0:
            t_pred = pred[:, 0 : self.attr_type_numcls[idx]]
            t_target = target[:, 0 : self.attr_type_numcls[idx]]
        else:
            t_pred = pred[
                :,
                sum(self.attr_type_numcls[:idx]) : sum(
                    self.attr_type_numcls[: idx + 1]
                ),
            ]
            t_target = target[
                :,
                sum(self.attr_type_numcls[:idx]) : sum(
                    self.attr_type_numcls[: idx + 1]
                ),
            ]
        return t_pred, t_target

    def forward(self, pred, target):
        loss_dict = {}
        for idx, attr_type in enumerate(self.attr_type_list):
            t_pred, t_target = self.get_pred_target(idx, pred, target)

            o_weight = t_target.sum(axis=1)
            o_weight[o_weight > 0] = 1

            t_target = torch.argmax(t_target, axis=1)
            c_loss = F.cross_entropy(
                t_pred,
                t_target,
                weight=self.weight,
                ignore_index=self.ignore_index,
                reduction="none",
            )
            c_loss = c_loss * o_weight

            if (
                attr_type != "ignore"
                and attr_type != "occlusion"
                and "confidence" not in attr_type
                and self.ignore_idx is not None
            ):
                c_loss_weight = target[:, self.ignore_idx].sum(axis=1) + 1
                c_loss_weight[c_loss_weight > 1] = 0
                c_loss = c_loss * c_loss_weight

            loss_dict[attr_type] = c_loss.mean()
        return sum(loss_dict.values())
