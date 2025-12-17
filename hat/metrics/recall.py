# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Sequence, Union

import torch

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list
from .metric import EvalMetric

__all__ = ["Recall"]


@OBJECT_REGISTRY.register
class Recall(EvalMetric):
    """Computes recall classification score.

    Args:
        axis (int): The axis that represents classes.
        cls_num (int): The number of categories.
        task_name (str):  Task name of this metric instance for display.
    """

    def __init__(
        self,
        axis: int = 1,
        cls_num: int = 2,
        task_name: str = "",
    ):
        self.task_name = task_name
        self.axis = axis
        self.cls_num = cls_num
        self.name = [
            self.task_name + "_recall_" + str(i) for i in range(self.cls_num)
        ]
        super(Recall, self).__init__(self.name)

    def _init_states(self):
        self.add_state(
            "sum_metric",
            default=torch.zeros(self.cls_num),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "num_inst", default=torch.zeros(self.cls_num), dist_reduce_fx="sum"
        )

    def update(
        self,
        labels: Union[torch.Tensor, Sequence[torch.Tensor]],
        preds: Union[torch.Tensor, Sequence[torch.Tensor]],
    ):
        labels = _as_list(labels)
        preds = _as_list(preds)
        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = torch.argmax(pred_label, self.axis)

            # flatten before checking shapes to avoid shape miss match
            label = label.flatten()
            pred_label = pred_label.flatten()

            correct_idx = pred_label == label
            for cls in range(self.cls_num):
                self.num_inst[cls] += torch.sum(label == cls)
                self.sum_metric[cls] += torch.sum(label[correct_idx] == cls)

    def compute(self):
        values = [
            (sum / num).item() if num != 0 else float("nan")
            for sum, num in zip(self.sum_metric, self.num_inst)
        ]
        return values
