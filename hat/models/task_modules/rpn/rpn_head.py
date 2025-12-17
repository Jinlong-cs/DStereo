# Copyright (c) Horizon Robotics. All rights reserved.

from collections import OrderedDict
from itertools import chain
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from horizon_plugin_pytorch.dtype import qinfo
from horizon_plugin_pytorch.quantization import (
    FakeQuantize,
    FixedScaleObserver,
    default_weight_8bit_fake_quant,
)
from torch.quantization import DeQuantStub, QConfig

from hat.models.base_modules.basic_vargnet_module import OnePathResUnit
from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.base_modules.postprocess import FilterModule
from hat.registry import OBJECT_REGISTRY

__all__ = ["RPNVarGNetHead", "RPNDetFilter"]


@OBJECT_REGISTRY.register
class RPNVarGNetHead(nn.Module):
    """RPN head composed of VarGNet units.

    To support all possible use cases, this module has 3 outputs in each
    run, keyed by 'rpn_head_out', 'rpn_cls_pred', and 'rpn_reg_pred'
    respectively. Generally, the first output, with shape (B,
    (4 + num_classes) * num_anchor, H, W) in each feature stride, is used
    in DPP module, while the latter two, sliced from the first one, work with
    loss modules in training stage.

    In QAT mode, this module works in INT8 precision to work well with DPP
    module.

    Args:
        in_channels: Number of channels of input feature maps. If input
            as int, it means all feature maps share the same number of
            channels.
        num_channels: Number of output channels of output of rpn_conv
            modules.
        feat_strides: Strides of each input feature maps.
        num_classes: Number of classes should be predicted.
        is_dim_match: Arg of OnePathResUnit.
        bn_kwargs: Keyword-arguments of BatchNorm layers.
        factor: Arg of OnePathResUnit.
        group_base: group base of rpn_conv modules.
        qat_dtype: qconfig dtype, choose int8 or int 16.
    """

    def __init__(
        self,
        in_channels: Union[List[int], int],
        num_channels: List[int],
        num_anchors: List[int],
        feat_strides: List[int],
        num_classes: int,
        is_dim_match: bool,
        bn_kwargs: Dict,
        factor: float,
        group_base: int,
        output_shift: int = 4,
        qat_dtype: float = "int8",
    ):
        super().__init__()
        if isinstance(in_channels, int):
            in_channels = [in_channels] * len(feat_strides)

        assert len(num_channels) == len(feat_strides) == len(num_anchors)

        self.num_anchors = num_anchors
        self.feat_strides = feat_strides
        self.output_shift = output_shift
        self.qat_dtype = qat_dtype

        self.dequant = DeQuantStub()

        self.rpn_conv = nn.ModuleDict()
        for i, stride in enumerate(self.feat_strides):
            self.rpn_conv[f"stride_{stride}"] = OnePathResUnit(
                in_filter=in_channels[i],
                dw_num_filter=num_channels[i],
                group_base=group_base,
                pw_num_filter=num_channels[i],
                pw_num_filter2=num_channels[i],
                bn_kwargs=bn_kwargs,
                stride=1,
                factor=factor,
                is_dim_match=is_dim_match,
                use_bias=True,
            )

        out_channels = [
            (4 + num_classes) * num_anchor for num_anchor in num_anchors
        ]
        self.out_channel = out_channels
        self.rpn_fc = nn.ModuleDict()
        for i, stride in enumerate(self.feat_strides):
            self.rpn_fc[f"stride_{stride}"] = ConvModule2d(
                in_channels=num_channels[i],
                out_channels=out_channels[i],
                kernel_size=1,
                stride=1,
                bias=False,
                padding=0,
                norm_layer=nn.BatchNorm2d(out_channels[i], **bn_kwargs),
            )

    def forward(
        self, x: List[torch.TensorType]
    ) -> Dict[str, List[torch.Tensor]]:
        cls_pred, reg_pred, outputs = [], [], []

        for i, (stride, num_anchor) in enumerate(
            zip(self.feat_strides, self.num_anchors)
        ):

            rpn_feature = self.rpn_fc[f"stride_{stride}"](
                self.rpn_conv[f"stride_{stride}"](x[i])
            )

            # head output without slicing
            outputs.append(rpn_feature)

            # dequant to output to loss module
            rpn_feature = self.dequant(rpn_feature)

            # slice output for loss computation, the order matters
            dim = rpn_feature.shape[1] // num_anchor
            rpn_feature = rpn_feature.view(
                -1, num_anchor, dim, *rpn_feature.shape[2:]
            )
            _cls = rpn_feature[:, :, 4:].flatten(1, 2)
            _reg = rpn_feature[:, :, :4].flatten(1, 2)

            cls_pred.append(_cls)
            reg_pred.append(_reg)

        return OrderedDict(
            rpn_head_out=outputs, rpn_cls_pred=cls_pred, rpn_reg_pred=reg_pred
        )

    def fuse_model(self):
        for m in chain(self.rpn_conv.values(), self.rpn_fc.values()):
            m.fuse_model()

    def set_qconfig(self):

        # shift 4 for dpp v1
        if self.qat_dtype == "int16":
            qcfg = QConfig(
                activation=FakeQuantize.with_args(
                    observer=FixedScaleObserver,
                    quant_min=qinfo("qint16").min,
                    quant_max=qinfo("qint16").max,
                    dtype="qint16",
                    scale=1 / 2 ** self.output_shift,
                ),
                weight=default_weight_8bit_fake_quant,
            )
        else:
            qcfg = QConfig(
                activation=FakeQuantize.with_args(
                    observer=FixedScaleObserver,
                    quant_min=qinfo("qint8").min,
                    quant_max=qinfo("qint8").max,
                    dtype="qint8",
                    scale=1 / 2 ** self.output_shift,
                ),
                weight=default_weight_8bit_fake_quant,
            )
        # disable output quantization for last quanti layer.
        for m in self.rpn_fc.values():
            m.qconfig = qcfg


@OBJECT_REGISTRY.register
class RPNDetFilter(nn.Module):
    """Filter RPN result by threshold.

    Args:
        threshold: Threshold, the lower bound of output.
        num_anchor: Number of anchors.
        strides: strides of the multi-scale feature maps. Must equal [1] or None.
            "[1]" Indicates that front unet outputs stride1 feature, and "None" indicates that
            this class does not pay attention to strides and directly takes the first output feature.
        idx_range: The index range of values counted in compare of the first input.
            Defaults to None which means use all the values.
    """  # noqa

    def __init__(
        self,
        threshold: float,
        num_anchor: int,
        strides: Optional[Sequence[int]] = None,
        idx_range: Optional[Tuple[int, int]] = None,
    ):
        super(RPNDetFilter, self).__init__()
        assert strides == [1] or strides is None
        self.filter_module = FilterModule(
            threshold=threshold,
            idx_range=idx_range,
        )
        self.num_anchor = num_anchor

    def forward(
        self,
        anchors,
        head_out: Dict[str, List[torch.Tensor]],
    ) -> Sequence[torch.Tensor]:

        rpn_feature = head_out["head_predict_rpn_head_out"][0]

        dim = rpn_feature.shape[1] // self.num_anchor
        rpn_feature = rpn_feature.view(
            -1, self.num_anchor, dim, *rpn_feature.shape[2:]
        )
        cls_pred = rpn_feature[:, :, 4:].flatten(1, 2)
        reg_pred = rpn_feature[:, :, :4].flatten(1, 2)

        filter_output = self.filter_module(*[cls_pred, reg_pred])

        return {
            "filter_coord": filter_output[0][-3],
            "filter_scores": filter_output[0][-2],
            "filter_bboxes": filter_output[0][-1],
        }
