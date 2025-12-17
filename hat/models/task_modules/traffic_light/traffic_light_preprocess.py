# Copyright (c) Horizon Robotics. All rights reserved.
import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub

from hat.registry import OBJECT_REGISTRY

__all__ = ["TrafficLightPreprocess"]


@OBJECT_REGISTRY.register
class TrafficLightPreprocess(nn.Module):
    """Input preprocess for Traffic Light.

    Args:
        mask_in_bpu: For more details, see py:class:`TrafficLightClassifier`
        transpose_hw: Whether the input data is already transpose.
            For example, If transpose=False, the input is currently (1,1,1,96)
            and transpose op needs to be called, which becomes (1,1,96,1).
            Mainly to solve the problem that J3 does not support transpose
            operation, the process of transpose is put on the software.
    """

    def __init__(
        self,
        mask_in_bpu: bool = False,
        transpose_hw: bool = False,
    ):
        super(TrafficLightPreprocess, self).__init__()
        self.mask_in_bpu = mask_in_bpu
        self.transpose_hw = transpose_hw
        self.quant = QuantStub(scale=1.0 / 128.0)
        if mask_in_bpu:
            self.width_mask_quant = QuantStub(scale=1)
            self.height_mask_quant = QuantStub(scale=1)
            self.image_q_op = nn.quantized.FloatFunctional()
            self.mask_q_op = nn.quantized.FloatFunctional()

    def forward(self, *inputs):
        image = inputs[0]
        x = self.quant(image)
        if self.mask_in_bpu:
            height_crop_mask, width_crop_mask = inputs[1], inputs[2]

            assert width_crop_mask.is_floating_point(), (
                "Only float types are supported"
                f"for attn_mask, not {width_crop_mask.dtype}"
            )
            width_crop_mask = self.width_mask_quant(width_crop_mask)
            height_crop_mask = self.height_mask_quant(height_crop_mask)
            if not self.transpose_hw:
                height_crop_mask = height_crop_mask.transpose(2, 3)

            x = self.image_q_op.mul(
                self.mask_q_op.mul(x, width_crop_mask), height_crop_mask
            )

        return x
