# Copyright (c) Horizon Robotics. All rights reserved.

import json
from typing import Sequence

import horizon_plugin_pytorch as horizon
import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY, build_from_registry
from hat.utils.apply_func import flatten
from tests.data.toy_modules import *  # noqa: F401,F403


@OBJECT_REGISTRY.register
class MultitaskModel(nn.Module):
    def __init__(
        self, per_task_pp: Sequence[dict], per_task_pred_num: Sequence[int]
    ):
        super(MultitaskModel, self).__init__()
        assert len(per_task_pp) == len(per_task_pred_num)
        self.per_task_pp = nn.ModuleList(
            [build_from_registry(pp) for pp in per_task_pp]
        )
        self.per_task_pred_num = per_task_pred_num

    def forward(self, x):
        # TODO (shuqian,qu, ?), get_output_annotation supports more complicated
        #  prediction format.
        # now, horizon.get_output_annotation only support prediction format
        # like:
        # (
        #     (task1_tensor1, task1_tensor2, ...),
        #     (task2_tensor1, task2_tensor2, ...),
        #     ...
        # )
        per_task_pred = tuple(
            tuple(x + i + j for j in range(i)) for i in self.per_task_pred_num
        )
        assert len(per_task_pred) == len(self.per_task_pp)

        outs = []
        for pred, pp in zip(per_task_pred, self.per_task_pp):
            outs.append(pp(pred))

        return tuple(outs)


def test_add_multitask_desc():
    per_task_desc = [
        [
            json.dumps(
                dict(
                    task="task1",
                    output_name="output1",
                )
            ),
            json.dumps(
                dict(
                    task="task1",
                    output_name="output2",
                )
            ),
        ],
        [
            json.dumps(
                dict(
                    task="task2",
                    output_name="output1",
                )
            ),
            json.dumps(
                dict(
                    task="task2",
                    output_name="output2",
                )
            ),
        ],
    ]
    per_task_pp = [
        dict(type="AddDesc", per_tensor_desc=per_task_desc[i])
        for i in range(len(per_task_desc))
    ]
    per_task_pred_num = [len(i) for i in per_task_desc]
    model = MultitaskModel(
        per_task_pp=per_task_pp, per_task_pred_num=per_task_pred_num
    )

    outs = model(torch.randn(2, 2))
    # 1. get desc from tensor.annotation
    assert len(outs) == len(per_task_desc)
    for task_out, task_desc in zip(outs, per_task_desc):
        for tensor, desc in zip(task_out, task_desc):
            assert tensor.annotation == desc, f"{tensor.annotation} vs. {desc}"

    # 2. get desc from script_module
    script_module = torch.jit.trace(
        func=model.eval(),
        example_inputs=(torch.randn(2, 2)),
    )

    # Note, get_output_annotation will not maintain the output format anymore,
    # it returns flat outputs, keep accordance with hbdk compiled model.
    flat_anno = horizon.get_output_annotation(script_module)
    flat_outs, _ = flatten(outs)
    assert len(flat_outs) == len(flat_anno)
    for tensor, desc in zip(flat_outs, flat_anno):
        assert tensor.annotation == desc, f"{tensor.annotation} vs. {desc}"
