# Copyright (c) Horizon Robotics. All rights reserved.
from collections import OrderedDict
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.task_modules.landmark.ldmk_head import (
    BandConvModule,
    BandPoolModule,
)
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "SmokeKpsHeatmapHead",
    "SmokeKpsVectorHead",
    "SmokeKpsClsHead",
]


@OBJECT_REGISTRY.register
class SmokeKpsHeatmapHead(nn.Module):
    """Convert feature map to predicted heatmap.

    Use 1x1 conv2d to transform feature to num_ldmk-channel heatmap.
    Post-processing on heatmap transforms each channel to a pair of coord,
    which is not implemented here.

    The pipeline is like Backbone->Decoder(upscale)->heatmap.

    Args:
        in_channels: channel number of decoder module.
        num_ldmk: number of landmark.
        is_train: train mode (with loss calculation) or not. Defaults to False.
        loss_func: loss function. Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        num_ldmk: int,
        is_train: bool = False,
        loss_func: Optional[Callable] = None,
    ):
        super().__init__()
        self.num_ldmk = num_ldmk
        self.is_train = is_train
        self.loss_func = loss_func
        self.head = ConvModule2d(in_channels, self.num_ldmk + 1, 1)
        self.dequant = DeQuantStub()

    def forward(self, data):
        feat = data["feat"]
        outputs = OrderedDict()
        loss = OrderedDict()

        heatmap_pred = self.head(feat)
        heatmap_pred = self.dequant(heatmap_pred)
        outputs["pr_heatmap"] = heatmap_pred
        if self.is_train:
            heatmap_label = data["gt_heatmap"]
            heatmap_weight = data["gt_heatmap_weight"]
            loss_weight = data.get("loss_weight", 1.0)
            loss["heatmap_loss"] = loss_weight * self.loss_func(
                heatmap_label, heatmap_pred, heatmap_weight
            )
            return loss
        return outputs

    def fuse_model(self):
        self.head.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()


@OBJECT_REGISTRY.register
class SmokeKpsVectorHead(nn.Module):
    """Convert upscaled feature map to predicted vectors.

    The pipeline is like Backbone->Decoder(upscale)->vectors.

    Args:
        in_channels: channel number of decoder module.
        num_ldmk: number of landmark.
        band_width: band width of band_conv or band_pool
        vector_size: output vector length, (W, H). W should equal
            to H until now.
        band_module_type: "conv" or "pool" to choose BandConvModule or
            BandPoolModule. "conv" is recommended in quantization model.
            Defaults to "conv".
        is_train: train mode or val mode. Defaults to False.
        loss_func: loss function. Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        num_ldmk: int,
        band_width: int,
        vector_size: Tuple[int, int],
        band_module_type: str = "conv",
        is_train: bool = False,
        loss_func: Optional[Callable] = None,
    ):
        super().__init__()
        self.num_ldmk = num_ldmk
        band_module_type = band_module_type.lower()
        self.is_train = is_train and loss_func is not None
        self.loss_func = loss_func
        width, height = vector_size

        self.share_conv = ConvModule2d(
            in_channels,
            self.num_ldmk,
            1,
            norm_layer=nn.BatchNorm2d(self.num_ldmk),
            act_layer=nn.ReLU(inplace=True),
        )
        self.vector_x_head = []
        self.vector_y_head = []
        self.dequant = DeQuantStub()

        if band_module_type == "pool":
            self.vector_x_head.append(
                BandPoolModule(width, band_width, horizontal=True)
            )
            self.vector_y_head.append(
                BandPoolModule(height, band_width, horizontal=False)
            )
        elif band_module_type == "conv":
            self.vector_x_head.append(
                BandConvModule(width, band_width, num_ldmk, horizontal=True)
            )
            self.vector_y_head.append(
                BandConvModule(height, band_width, num_ldmk, horizontal=False)
            )
        else:
            raise ValueError(
                f"Not supported band_module_type: {band_module_type}."
            )
        self.vector_x_head.append(ConvModule2d(num_ldmk, num_ldmk, 1, 1, 0))
        self.vector_y_head.append(ConvModule2d(num_ldmk, num_ldmk, 1, 1, 0))
        self.vector_x_head = nn.Sequential(*self.vector_x_head)
        self.vector_y_head = nn.Sequential(*self.vector_y_head)

    def forward(self, data):
        feat = data["feat"]
        outputs = OrderedDict()
        loss = OrderedDict()
        feat = self.share_conv(feat)
        vector_x_pred = self.dequant(self.vector_x_head(feat))
        vector_y_pred = self.dequant(self.vector_y_head(feat))
        outputs["pr_vector_x"] = vector_x_pred
        outputs["pr_vector_y"] = vector_y_pred

        if self.is_train:
            vector_label_x = data["gt_vector_x"]
            vector_label_y = data["gt_vector_y"]
            vector_weight_x = data["gt_vector_weight_x"]
            vector_weight_y = data["gt_vector_weight_y"]
            loss_weight = data.get("loss_weight", 1.0)
            loss["vector_loss"] = loss_weight * self.loss_func(
                vector_label_x, vector_x_pred.squeeze(2), vector_weight_x
            ) + loss_weight * self.loss_func(
                vector_label_y, vector_y_pred.squeeze(3), vector_weight_y
            )
            return loss
        return outputs

    def fuse_model(self):
        for m in [self.share_conv, self.vector_x_head, self.vector_y_head]:
            if hasattr(m, "fuse_model"):
                m.fuse_model()
            else:
                for mm in m:
                    mm.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()


@OBJECT_REGISTRY.register
class SmokeKpsClsHead(nn.Module):
    """Convert feature map to visiable and classes.

    Use 2 conv2d layers to transform feature to class_head.
    The first unit of class_head means visable of cigaret,
    which is resized by sigmoid function.
    The remaining unit of class_head means classes of smoke,
    which is resized by softmax function.

    The pipeline is like Backbone->Decoder(upscale)->cls_head.

    Args:
        in_channels: channel number of decoder module.
        middle_dim: output dim of first conv2d layer.
        output_dim: output dim of second conv2d layer.
        is_train: train mode (with loss calculation) or not. Defaults to False.
        loss_func: loss function. Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        middle_dim: int,
        output_dim: int,
        bias=True,
        is_train: bool = False,
        loss_func: Optional[Callable] = None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.is_train = is_train
        self.loss_func = loss_func
        self.head = nn.Sequential(
            ConvModule2d(
                in_channels,
                middle_dim,
                kernel_size=(2, 2),
                stride=1,
                padding=(0, 0),
                norm_layer=nn.BatchNorm2d(middle_dim),
                act_layer=nn.ReLU(inplace=True),
            ),
            ConvModule2d(
                middle_dim,
                output_dim,
                kernel_size=(3, 3),
                stride=1,
                padding=(0, 0),
                bias=bias,
            ),
        )
        self.dequant = DeQuantStub()

    def forward(self, data):
        feat = data["feat"]
        outputs = OrderedDict()
        loss = OrderedDict()

        cls_pred = self.head(feat)
        cls_pred = self.dequant(cls_pred).squeeze(-1).squeeze(-1)
        outputs["pr_visable"] = torch.sigmoid(cls_pred[:, :1])
        outputs["pr_classes"] = torch.softmax(cls_pred[:, 1:], dim=1)
        if self.is_train:
            vis_label = data["gt_visable"]
            vis_weight = data["gt_vis_weight"]
            cls_label = data["gt_classes"]
            cls_weight = data["gt_cls_weight"]
            vis_loss_weight = data.get("vis_loss_weight", 1.0)
            cls_loss_weight = data.get("cls_loss_weight", 1.0)

            loss["vis_loss"] = vis_loss_weight * self.loss_func(
                vis_label, outputs["pr_visable"], vis_weight
            )
            loss["cls_loss"] = cls_loss_weight * self.loss_func(
                cls_label, outputs["pr_classes"], cls_weight
            )
            return loss
        return outputs

    def fuse_model(self):
        for m in self.head:
            if hasattr(m, "fuse_model"):
                m.fuse_model()
            else:
                for mm in m:
                    mm.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
