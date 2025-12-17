# Copyright (c) Horizon Robotics. All rights reserved.

import torch


def module_list_forward(
    layers: torch.nn.ModuleList, input: torch.Tensor
) -> torch.Tensor:
    output = input
    for layer in layers:
        output = layer(output)
    return output
