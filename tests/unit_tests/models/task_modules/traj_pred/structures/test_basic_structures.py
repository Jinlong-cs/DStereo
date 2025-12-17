# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Callable, Dict, Optional

import torch
from torch import nn

from hat.models.task_modules.traj_pred.structures.basic_structures import (
    BasicTrajPredStructure,
)


class TestNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_dict: bool = False,
        loss: Optional[Callable] = None,
        post_process: Optional[Callable] = None,
    ):
        super(TestNet, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1
        )
        self.output_dict = output_dict
        self.loss = loss
        self.post_process = post_process

    def forward(self, data):
        out = self.conv(data)
        if self.output_dict:
            return {"out": out}
        else:
            return out


class TestStructure(BasicTrajPredStructure):
    def __init__(
        self,
        backbone: Callable,
        necks: Optional[Dict] = None,
        heads: Optional[Dict] = None,
        post_process: Optional[Callable] = None,
        losses: Optional[Dict] = None,
        is_int_infer_model: bool = False,
    ):
        kwargs = {
            "backbone": backbone,
            "necks": necks,
            "heads": heads,
            "post_process": post_process,
            "losses": losses,
            "is_int_infer_model": is_int_infer_model,
        }
        super(TestStructure, self).__init__(**kwargs)

    def custom_data_preprocess(self, data: Dict):
        return data, data["input"], data


class TestLoss(nn.Module):
    def __init__(self):
        super(TestLoss, self).__init__()

    def forward(self, model_output, target=None):
        gts = model_output["gts"].reshape([-1])
        out = model_output["out"].reshape([-1])
        loss = torch.mean((gts - out) ** 2)
        return {"loss": loss}


def test_traj_pred_basic_structure():
    model = TestStructure(
        backbone=TestNet(
            in_channels=16,
            out_channels=32,
        ),
        heads={
            "normal_head": TestNet(
                in_channels=32,
                out_channels=64,
                output_dict=True,
                loss=TestLoss(),
            )
        },
    )

    data = {
        "input": torch.ones([11, 16, 1, 1], dtype=torch.float),
        "gts": torch.ones([11, 64, 1, 1], dtype=torch.float),
    }
    output = model(data)
    assert "normal_head_out" in output
    assert "normal_head_loss" in output
    assert output["normal_head_out"].shape == (11, 64, 1, 1)
