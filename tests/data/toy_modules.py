from typing import Sequence

import torch
import torch.nn as nn
from torch.quantization import DeQuantStub, QuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY, build_from_registry
from hat.utils import qconfig_manager

__all__ = [
    "ToyBackbone",
    "ToyHead",
    "ToyLoss",
    "ToyTarget",
    "ToyPostProcess",
    "ToyHeadParser",
    "ToyModel",
]


@OBJECT_REGISTRY.register
class ToyBackbone(nn.Module):
    def __init__(
        self, strides=(1, 2, 4, 8, 16, 32), channels=(3, 8, 8, 16, 32, 64)
    ):
        super(ToyBackbone, self).__init__()
        assert len(strides) == len(channels), "%d vs. %d" % (
            len(strides),
            len(channels),
        )

        self.quant = QuantStub()
        down_blocks = []
        for i in range(len(strides) - 1):
            in_stride = strides[i]
            out_stride = strides[i + 1]
            down_blocks.append(
                ConvModule2d(
                    in_channels=channels[strides.index(in_stride)],
                    out_channels=channels[strides.index(out_stride)],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    act_layer=nn.ReLU(inplace=True),
                    norm_layer=nn.BatchNorm2d(
                        channels[strides.index(out_stride)]
                    ),
                )
            )

        self.down_blocks = nn.Sequential(*down_blocks)

    def forward(self, x):
        x = self.quant(x)
        return self.down_blocks(x)

    def fuse_model(self):
        for m in self.down_blocks:
            if hasattr(m, "fuse_model"):
                m.fuse_model()

    def set_qconfig(self):
        for module in self.down_blocks:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()


@OBJECT_REGISTRY.register
class ToyHead(nn.Module):
    def __init__(
        self,
        in_channels,
        fc_filter,
        num_classes,
        with_dequant=False,
        reshape_output=True,
    ):
        super(ToyHead, self).__init__()
        self.reshape_output = reshape_output
        self.last_conv = ConvModule2d(
            in_channels=in_channels,
            out_channels=fc_filter,
            kernel_size=1,
            stride=1,
            padding=0,
            act_layer=nn.ReLU(inplace=True),
            norm_layer=nn.BatchNorm2d(fc_filter),
        )
        self.pool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Conv2d(
            in_channels=fc_filter,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False,
        )
        if with_dequant:
            self.dequant = DeQuantStub()
        self.num_classes = num_classes

    def forward(self, x, *args):
        x = self.last_conv(x)
        x = self.pool(x)
        x = self.fc(x)
        if hasattr(self, "dequant"):
            x = self.dequant(x)
        # reshape after de-quantize, then x is Tensor, not QTensor which does
        # not has reshape
        if self.reshape_output:
            x = x.reshape(-1, self.num_classes)
        return x

    def fuse_model(self):
        self.last_conv.fuse_model()

    def set_qconfig(self):
        self.fc.qconfig = qconfig_manager.get_default_qat_out_qconfig()


@OBJECT_REGISTRY.register
class ToyLoss(nn.Module):
    """
    ToyLoss stands for algorithm-related loss, such as Real3dLoss.
    """

    def __init__(self):
        super(ToyLoss, self).__init__()
        self.loss = build_from_registry(dict(type="CEWithLabelSmooth"))

    def forward(self, pred, label, *args):
        pred = pred.reshape(pred.shape[0], pred.shape[1])
        return self.loss(pred, label)


@OBJECT_REGISTRY.register
class ToyTarget(nn.Module):
    def __init__(self):
        super(ToyTarget, self).__init__()

    def forward(self, label, *args):
        target = label
        return target


@OBJECT_REGISTRY.register
class ToyPostProcess(nn.Module):
    def __init__(self):
        super(ToyPostProcess, self).__init__()

    def forward(self, pred, *args, **kwargs):
        # return tuple instead of list, cos constant container is recommended
        # by `torch.jit.trace()`
        # return multi tensors just for test
        pred = pred[0] if isinstance(pred, Sequence) else pred
        cat_pred = torch.cat([pred, pred], dim=1)
        pred1, pred2 = torch.split(
            cat_pred, split_size_or_sections=cat_pred.shape[1] // 2, dim=1
        )
        return pred1, pred2


@OBJECT_REGISTRY.register
class ToyHeadParser(nn.Module):
    """
    ToyHeadParser is optional, used to parse toy head output to next
    operators.
    """

    def __init__(self):
        super(ToyHeadParser, self).__init__()

    def forward(self, pred):
        parse_output = pred
        return parse_output


@OBJECT_REGISTRY.register
class ToyModel(nn.Module):
    def __init__(self, backbone, head, loss=None):
        super(ToyModel, self).__init__()
        self.backbone = build_from_registry(backbone)
        self.head = build_from_registry(head)
        self.loss = build_from_registry(loss) if loss is not None else None

    def forward(self, x, label=None):
        if isinstance(x, list):
            x, label = x[0], x[1]
        assert label is not None, "`label` can't be None!"
        x = self.backbone(x)
        x = self.head(x)
        if self.loss is not None:
            loss = self.loss(x, label)
        else:
            loss = None
        return dict(loss=loss, predict=x)

    def fuse_model(self):
        for module in [self.backbone, self.head]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        for module in [self.backbone, self.head]:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()
        if self.loss:
            self.loss.qconfig = None
