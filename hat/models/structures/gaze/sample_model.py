# Copyright (c) Horizon Robotics. All rights reserved.
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from hat.registry import OBJECT_REGISTRY

__all__ = ["SampleModel"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class SampleModel(nn.Module):
    """
    Model for calling F.grid_sample.

    The model has two inputs, namely image and grid_sample.
    Among them, the image is of type int8, grid_sample is of type int16.
    The quantization scale is calculated based on the actual numerical range.
    Image input needs to quantize floating-point from -1.0 to 1.0 to
    -128 to 128 (int8), so the quantized scale is 1.0/128.0.
    Similarly, the quantized scale of grid_sample is 1.0/32768.0.
    """

    def __init__(
        self,
        output_size: tuple = (320, 150),  # (width, height)
        deploy: bool = False,
    ):
        super(SampleModel, self).__init__()
        self.output_size = output_size
        self.deploy = deploy
        self.quant1 = QuantStub(scale=1.0 / 128.0, zero_point=0)
        self.quant2 = QuantStub(scale=1.0 / 32768.0, zero_point=0)
        self.dequant = DeQuantStub()
        self.linear_layer = nn.Linear(
            self.output_size[0] * self.output_size[1], 1
        )

    def forward(self, data):
        image = data["img"]
        grid = data["grid"]
        # grid.shape == [n, c, h, w] and c == 2
        assert len(grid.shape) == 4 and grid.shape[1] == 2
        image = self.quant1(image)
        grid = self.quant2(grid)
        grid = torch.permute(grid, (0, 2, 3, 1))
        allout = F.grid_sample(
            image,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        allout_deq = self.dequant(allout)
        if self.deploy:
            return allout_deq

        allout_flat = torch.flatten(allout, 1, 3)
        # linear_layer is to make the model output 1 number during training,
        # making it easy to prepare gt and loss. This layer is not output
        # during deployment, so it has no impact on the actual model.
        allout_lin = self.linear_layer(allout_flat)
        allout_lin = self.dequant(allout_lin)
        total_loss = F.l1_loss(allout_lin, data["label"])
        return allout_deq, total_loss

    def fuse_model(self):
        pass

    def set_qconfig(self):
        import horizon_plugin_pytorch as horizon

        self.qconfig = horizon.quantization.get_default_qat_qconfig()
        self.quant2.qconfig = horizon.quantization.get_default_qat_qconfig(
            dtype="qint16"
        )
