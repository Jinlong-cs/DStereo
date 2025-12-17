from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import (
    ConvModule2d,
    ConvTransposeModule2d,
)
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list

__all__ = [
    "LdmkHeatmapHead",
    "LdmkCoordsHead",
    "LdmkDecoder",
    "LdmkVectorHead",
    "KPSDetectHead",
]


@OBJECT_REGISTRY.register
class LdmkDecoder(nn.Module):
    """Upsample features to get landmark heatmap without FPN.

    Args:
        in_stride: input feature (usually from backbone) stride. Set it to 32
            if there are 5 stages in the backbone.
        out_stride: desirable output feature map stride. Set it to 4 in most
            cases. Set it to 1 if you do not need any downsampling.
        in_channels: backbone feature map channels.
        out_channels: intermedieate and final output channels.
        bn_kwargs: BN parmams. Defaults to None.
    """

    def __init__(
        self,
        in_stride: int,
        out_stride: int,
        in_channels: int,
        out_channels: int,
        bn_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        if bn_kwargs is None:
            bn_kwargs = {}
        self.channel_mapper = ConvModule2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
            act_layer=nn.ReLU(True),
        )
        self.decoder = nn.ModuleList()
        stride = in_stride
        while stride > out_stride:
            self.decoder.append(
                ConvTransposeModule2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=2,
                    stride=2,
                    norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
                    act_layer=nn.ReLU(True),
                )
            )
            self.decoder.append(
                ConvModule2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
                    act_layer=nn.ReLU(True),
                )
            )
            stride /= 2

    def forward(self, feat: Union[torch.Tensor, List]):
        if isinstance(feat, list):
            feat = feat[-1]
        feat = self.channel_mapper(feat)
        for layer in self.decoder:
            feat = layer(feat)
        return [feat]

    def fuse_model(self):
        for m in self.decoder:
            if hasattr(m, "fuse_model"):
                m.fuse_model()


@OBJECT_REGISTRY.register
class LdmkHeatmapHead(nn.Module):
    """Convert feature map to predicted heatmap.

    Use 1x1 conv2d to transform feature to num_ldmk-channel heatmap.
    Post-processing on heatmap transforms each channel to a pair of coord,
    which is not implemented here.

    The pipeline is like Backbone->Decoder(upscale)->heatmap.

    Args:
        in_channels: channel number of decoder module.
        num_ldmk: number of landmark.
        clip_negative: clip negative output to zero. Defaults to True.
        loss_func: loss function. Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        num_ldmk: int,
        clip_negative: bool = True,
        loss_func: Optional[Callable] = None,
    ):
        super().__init__()
        self.num_ldmk = num_ldmk
        self.loss_func = loss_func
        # TODO(yuhao.dou): add ReLU in train mode and remove it in test mode.
        self.head = ConvModule2d(
            in_channels,
            self.num_ldmk,
            1,
            act_layer=nn.ReLU(inplace=True) if clip_negative else None,
        )
        self.dequant = DeQuantStub()

    def forward(self, data):
        feat = data["feat"]
        outputs = OrderedDict()
        loss = OrderedDict()

        heatmap_pred = self.head(feat)
        heatmap_pred = self.dequant(heatmap_pred)
        outputs["pr_heatmap"] = heatmap_pred
        if self.training:
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
class LdmkVectorHead(nn.Module):
    """Convert upscaled feature map to predicted vectors.

    The pipeline is like Backbone->Decoder(upscale)->heatmap->vectors.

    Args:
        in_channels: channel number of decoder module.
        num_ldmk: number of landmark.
        band_width: band width of band_conv or band_pool
        vector_size: output vector length, (W, H). W should equal
            to H until now.
        band_module_type: "conv" or "pool" to choose BandConvModule or
            BandPoolModule. "conv" is recommended in quantization model.
            Defaults to "conv".
        loss_func: loss function. Defaults to None.
        clip_negative: clip negative output to zero. Defaults to False.
        bn_kwargs: Dict for Bn layer. Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        num_ldmk: int,
        band_width: int,
        vector_size: Tuple[int, int],
        band_module_type: str = "conv",
        loss_func: Optional[Callable] = None,
        clip_negative: bool = True,
        bn_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        self.num_ldmk = num_ldmk
        band_module_type = band_module_type.lower()
        self.loss_func = loss_func
        width, height = vector_size
        if bn_kwargs is None:
            bn_kwargs = {}

        self.share_conv = ConvModule2d(
            in_channels,
            self.num_ldmk,
            1,
            norm_layer=nn.BatchNorm2d(self.num_ldmk, **bn_kwargs),
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
        self.vector_x_head.append(
            ConvModule2d(
                in_channels=num_ldmk,
                out_channels=num_ldmk,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                act_layer=nn.ReLU(inplace=True) if clip_negative else None,
            )
        )
        self.vector_y_head.append(
            ConvModule2d(
                in_channels=num_ldmk,
                out_channels=num_ldmk,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                act_layer=nn.ReLU(inplace=True) if clip_negative else None,
            )
        )
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

        if self.training:
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

        self.vector_x_head[0].set_qconfig()
        self.vector_y_head[0].set_qconfig()
        self.vector_x_head[
            -1
        ].qconfig = qconfig_manager.get_default_qat_out_qconfig()  # noqa
        self.vector_y_head[
            -1
        ].qconfig = qconfig_manager.get_default_qat_out_qconfig()  # noqa

        if self.loss_func is not None:
            self.loss_func.qconfig = None


class BandConvModule(nn.Module):
    """Apply convolution on heatmap to get vectors.

    A simple way to downsample the heatmap along one given dimension.
    Generally, it is better than BandPoolModule in quantized model.

    Args:
        vector_size: vector length.
        band_width: band conv width.
        num_channels: input and output channels.
        horizontal: output horizontal or vertical vector. The horizontal vector
            encodes `x` while the vertical encodes `y`. Defaults to True.
    """

    def __init__(
        self,
        vector_size: int,
        band_width: int,
        num_channels: int,
        horizontal: bool = True,
    ):
        super().__init__()
        stride_list = self.get_stride_list(vector_size)
        self.band_conv = []
        for s in stride_list:
            if horizontal:
                kernel_size = (s, band_width)
                stride = (s, 1)
                padding = (0, (band_width - 1) // 2)
            else:
                kernel_size = (band_width, s)
                stride = (1, s)
                padding = ((band_width - 1) // 2, 0)
            self.band_conv.append(
                ConvModule2d(
                    num_channels,
                    num_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    norm_layer=nn.BatchNorm2d(num_channels),
                    act_layer=nn.ReLU(inplace=True),
                )
            )
        self.band_conv = nn.Sequential(*self.band_conv)

    def get_stride_list(self, size):
        stride_list = []
        while size > 1:
            stride_list.append(2)
            size = size // 2
        stride_list.append(size)
        return stride_list

    def forward(self, feat):
        return self.band_conv(feat)

    def fuse_model(self):
        for m in self.band_conv:
            if hasattr(m, "fuse_model"):
                m.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()


class BandPoolModule(nn.Module):
    """Downsample heatmap with pool module to get vectors.

    It was proposed in https://arxiv.org/abs/2010.01318.

    Args:
        See BancConvModule.
    """

    def __init__(
        self,
        vector_size: int,
        band_width: int,
        horizontal: bool = True,
    ):
        super().__init__()
        stride_list = self.get_stride_list(vector_size)
        self.band_pool = []
        for s in stride_list:
            if horizontal:
                kernel_size = (s, band_width)
                stride = (s, 1)
                padding = (0, (band_width - 1) // 2)
            else:
                kernel_size = (band_width, s)
                stride = (1, s)
                padding = ((band_width - 1) // 2, 0)
            self.band_pool.append(
                nn.AvgPool2d(
                    kernel_size=kernel_size, stride=stride, padding=padding
                )
            )
        self.band_pool = nn.Sequential(*self.band_pool)

    def get_stride_list(self, size):
        stride_list = []
        base_stride = 4
        while size % base_stride == 0:
            stride_list.append(base_stride)
            size = size // base_stride
        if size != 1:
            stride_list.append(size)  # [4, 4, 2] for 32
        return stride_list

    def forward(self, feat):
        return self.band_pool(feat)

    def fuse_model(self):
        for m in self.band_pool:
            if hasattr(m, "fuse_model"):
                m.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()


class AggregatedBandPoolModule(nn.Module):
    # TODO(yuhao.dou)
    pass


class AggregatedBandConvModule(nn.Module):
    # TODO(yuhao.dou)
    pass


@OBJECT_REGISTRY.register
class LdmkCoordsHead(nn.Module):
    """Directly output landmark coords from backbone feature.

    Args:
        in_channels: channels of the previous layer(i.e. backbone).
        kernel_size: feature size of the previous layer(i.e. backbone).
        num_ldmk: number of landmark.
        loss_func: loss function. Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        num_ldmk: int,
        loss_func: Optional[Callable] = None,
    ):
        super(LdmkCoordsHead, self).__init__()
        self.num_ldmk = num_ldmk
        self.loss_func = loss_func
        self.pool = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size))
        self.coords_head = ConvModule2d(
            in_channels, self.num_ldmk * 2, kernel_size=1
        )
        self.dequant = DeQuantStub()

    def forward(self, data):
        loss = OrderedDict()
        outputs = OrderedDict()
        feat = data["feat"]
        feat = self.pool(feat)
        pr_ldmk = self.dequant(self.coords_head(feat))
        outputs["pr_ldmk"] = pr_ldmk

        if self.training:
            ldmk_label = data["gt_ldmk"]
            ldmk_pred = pr_ldmk.reshape(ldmk_label.shape)
            ldmk_weight = data["gt_ldmk_weight"]
            loss["ldmk_loss"] = data["loss_weight"] * self.loss_func(
                ldmk_label, ldmk_pred, ldmk_weight
            )
            return loss
        return outputs

    def fuse_model(self):
        self.coords_head.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()


@OBJECT_REGISTRY.register
class KPSDetectHead(nn.Module):
    """Output block for key point detection.

    Args:
        out_stride: output feature map stride.
        in_strides: input feature map strides.
        kps_label_dim: The number of dimension of the classification branch.
        kps_pos_offset_dim: The number of dimension of
            the position offset regression branch.
        bn_kwargs: Kwargs for batch normalization block.
        gc_group_base: The number of channels for each group.
        factor: Channel Expand factor, by default 1.0.
        input_channels: Conv input channels.
        kernel: Conv kernel size.
        stride: Conv stride.
        pad: Conv pad size.
        loss_func: Loss function.
        loss_weights: weights of loss.
    """

    def __init__(
        self,
        out_stride: int,
        in_strides: List,
        kps_label_dim: int,
        kps_pos_offset_dim: int,
        bn_kwargs: Dict,
        gc_group_base: int = 8,
        factor: int = 1,
        input_channels: int = 1,
        kernel: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (1, 1),
        pad: Tuple[int, int] = (1, 1),
        loss_func: Optional[Callable] = None,
        loss_weights: Optional[dict] = None,
    ):
        super(KPSDetectHead, self).__init__()
        in_strides = _as_list(in_strides)
        self.in_strides = in_strides

        self.out_stride = out_stride

        self.out_feat_block = ConvModule2d(
            input_channels,
            input_channels * factor,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            groups=int(input_channels / gc_group_base),
            bias=False,
            norm_layer=nn.BatchNorm2d(input_channels * factor, **bn_kwargs)
            if bn_kwargs
            else None,
            act_layer=nn.ReLU(inplace=True),
        )

        self.label_out_block = ConvModule2d(
            input_channels * factor,
            kps_label_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            norm_layer=None,
            act_layer=None,
        )
        self.pos_offset_out_block = ConvModule2d(
            input_channels * factor,
            kps_pos_offset_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            norm_layer=None,
            act_layer=None,
        )
        self.dequant = DeQuantStub()
        self.loss_func = loss_func
        self.loss_weights = loss_weights

    def forward(self, data):
        if isinstance(data, dict):
            features = data["features"]
        else:
            features = data
        idx = self.in_strides.index(self.out_stride)
        x = features[idx]

        x = self.out_feat_block(x)
        kps_label_pred = self.label_out_block(x)
        kps_pos_offset_pred = self.pos_offset_out_block(x)
        kps_label_pred = self.dequant(kps_label_pred)
        kps_pos_offset_pred = self.dequant(kps_pos_offset_pred)
        ret = OrderedDict()
        ret["kps_label_pred"] = kps_label_pred
        ret["kps_pos_offset_pred"] = kps_pos_offset_pred
        if self.loss_func is not None:
            assert self.loss_weights is not None
            loss_output = self.loss_func(ret, data)["wheel_kps_loss"]
            loss_kps_cls = loss_output[
                "kps_class_loss"
            ] * self.loss_weights.get("cls", 1)
            loss_kps_reg = loss_output["kps_reg_loss"] * self.loss_weights.get(
                "reg", 1
            )
            total_loss = loss_kps_cls + loss_kps_reg
            ret.update(
                {
                    "loss_kps_cls": loss_kps_cls,
                    "loss_kps_reg": loss_kps_reg,
                    "total_loss": total_loss,
                }
            )
        return ret

    def fuse_model(self):
        for m in [
            self.out_feat_block,
            self.label_out_block,
            self.pos_offset_out_block,
        ]:
            if hasattr(m, "fuse_model"):
                m.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.label_out_block[
            0
        ].qconfig = qconfig_manager.get_default_qat_out_qconfig()
        self.pos_offset_out_block[
            0
        ].qconfig = qconfig_manager.get_default_qat_out_qconfig()

        if self.loss_func is not None:
            self.loss_func.qconfig = None
