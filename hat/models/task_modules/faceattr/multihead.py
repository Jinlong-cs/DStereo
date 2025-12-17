# Copyright (c) Horizon Robotics. All rights reserved.
import torch.nn as nn
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY

__all__ = ["FaceAttrHead"]


@OBJECT_REGISTRY.register
class FaceAttrHead(nn.Module):
    """Multihead for face attr.

    Args:
        in_channels: Channels of each input feature map.
        age_classes: Num_class of age.
        gender_classes: Num_class of gender.
        bn_kwargs: Dict for BN layer.
    """

    def __init__(
        self,
        in_channels: int,
        age_classes: int,
        gender_classes: int,
        bn_kwargs: dict,
        deploy: bool = False,
    ):

        super(FaceAttrHead, self).__init__()

        self.feature = ConvModule2d(
            in_channels=in_channels,
            out_channels=128,
            kernel_size=7,
            groups=16,
            norm_layer=nn.BatchNorm2d(128, **bn_kwargs),
            act_layer=nn.ReLU(inplace=True),
        )

        self.head_ordinal = ConvModule2d(
            in_channels=128,
            out_channels=age_classes,
            kernel_size=1,
            norm_layer=nn.BatchNorm2d(age_classes, **bn_kwargs),
        )

        self.head_gender = ConvModule2d(
            in_channels=128,
            out_channels=gender_classes,
            kernel_size=1,
            norm_layer=nn.BatchNorm2d(gender_classes, **bn_kwargs),
        )

        self.dequant = DeQuantStub()
        self.age_classes = age_classes
        self.gender_classes = gender_classes
        self.deploy = deploy

    def forward(self, x):
        x = self.feature(x)
        pred_age = self.head_ordinal(x)
        pred_age = self.dequant(pred_age)

        pred_gender = self.head_gender(x)
        pred_gender = self.dequant(pred_gender)
        if not self.deploy:
            if self.age_classes == 1:
                pred_age = pred_age.flatten()
            else:
                pred_age = pred_age.squeeze()
            if self.gender_classes == 1:
                pred_gender = pred_gender.flatten()
            else:
                pred_gender = pred_gender.squeeze()
        return pred_age, pred_gender

    def fuse_model(self):
        modules = [self.feature, self.head_ordinal, self.head_gender]
        for module in modules:
            for m in module:
                if hasattr(m, "fuse_model"):
                    m.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        self.head_ordinal.qconfig = (
            qconfig_manager.get_default_qat_out_qconfig()
        )
        self.head_gender.qconfig = (
            qconfig_manager.get_default_qat_out_qconfig()
        )
