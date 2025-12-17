import logging
from collections import OrderedDict
from typing import Dict, Optional

import numpy as np
import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY

__all__ = ["PupilSegNet"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class PupilSegNet(nn.Module):
    """
    The structure of pupil segmentation.

    note: it contains two heads, one of which outputs the mask of the pupil \
          and the other outputs the five parameters of the ellipse \
            that fits the pupil.

    Args:
        encoder: Encoder module.
        decode: Decode mask head module.
        encoder_out_channels: The number of channels of encoder output.
        mode: Options includes `train`, `val`, or `test` mode.
        loss: Loss module for pupil segmentation. Default: None.
        bn_kwargs: Extra keyword arguments for bn layers. Default: None.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        encoder_out_channels: int,
        mode: str = "train",
        losses: nn.Module = None,
        bn_kwargs: Optional[Dict] = None,
    ):
        super(PupilSegNet, self).__init__()
        if bn_kwargs is None:
            bn_kwargs = {}
        self.mode = mode
        assert self.mode in {"train", "val", "deploy"}
        self.enc = encoder
        self.dec = decoder
        self.losses = losses
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.regress_head = nn.Sequential(
            ConvModule2d(
                in_channels=encoder_out_channels,
                out_channels=encoder_out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_layer=nn.BatchNorm2d(encoder_out_channels, **bn_kwargs),
                # act_layer=nn.ReLU(inplace=True),
                act_layer=nn.LeakyReLU(inplace=True),
            ),
            ConvModule2d(
                in_channels=encoder_out_channels,
                out_channels=5,
                kernel_size=4,
                stride=1,
                padding=0,
                norm_layer=None,
                act_layer=None,
            ),
        )
        self._initialize_weights()

    def forward(self, data: dict):
        img = self.quant(data["img"])
        outputs = OrderedDict()
        x4, x3, x2, x1, x = self.enc(img)
        ellipse_param_pred = self.regress_head(x)
        mask_pred = self.dec(x4, x3, x2, x1, x)

        ellipse_param_pred = self.dequant(ellipse_param_pred)
        mask_pred = self.dequant(mask_pred)

        outputs["ellipse_param_pred"] = ellipse_param_pred
        outputs["mask_pred"] = mask_pred

        if self.mode == "train":
            outputs.update(self.losses(outputs, data))
        elif self.mode == "deploy":
            outputs = list(outputs.values())
        return outputs

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def fuse_model(self):
        for module in [[self.enc], [self.dec], self.regress_head]:
            for m in module:
                if hasattr(m, "fuse_model"):
                    m.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        if hasattr(self.dec, "set_qconfig"):
            self.dec.set_qconfig()
        # disable output quantization for last quanti layer.
        getattr(
            self.regress_head, "1"
        ).qconfig = qconfig_manager.get_default_qat_out_qconfig()
        if self.losses is not None:
            self.losses.qconfig = None

    def set_calibration_qconfig(self):
        from hat.utils import qconfig_manager

        self.calibration_qconfig = (
            qconfig_manager.get_default_calibration_qconfig()
        )
