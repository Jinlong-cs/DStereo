# Copyright (c) Horizon Robotics. All rights reserved.

import warnings
from collections import OrderedDict
from typing import Dict, Sequence, Tuple

import horizon_plugin_pytorch as horizon
import torch.nn as nn
from torch import Tensor
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.base_modules.separable_conv_module import SeparableConvModule2d
from hat.models.weight_init import bias_init_with_prob, normal_init
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list, multi_apply

__all__ = ["MTFCOS3DHead"]

INF = 1e8


@OBJECT_REGISTRY.register
class MTFCOS3DHead(nn.Module):
    """Anchor-free head for Multi-task-fused-FCOS.

    This head support simultaneously outputs of 2d and 3d classification
    and regression results, where 2d and 3d results are correlated.

    Args:
        num_classes: Number of categories excluding the background category.
        in_strides: A list contains the strides of feature maps from backbone
            or neck.
        out_strides: A list contains the strides of this head will output.
        head_channels: A dict to tell which head channel should be outputed.
        stride2channels: A stride to channel dict.
        upscale_bbox_pred: If true, upscale bbox pred by FPN strides.
        feat_channels: Number of hidden channels.
        stacked_convs: Number of stacking convs of the head.
        head_conv_kernel_size: Kernel size of last conv module.
        head_sep_conv: Whether to use SeparableConvModule2d.
        use_sigmoid: Whether to apply sigmoid to the classification output.
        share_bn: Whether to share bn between multiple levels. Default
            is False.
        dequant_output: Whether to dequant output. Default: True
        int8_output: If True, output int8, otherwise output int32.
            Default is True.
        share_conv: Only the number of all stride channels is the same,
            share_conv can be True, branches share conv, otherwise not.
            Default is True.
    """

    def __init__(
        self,
        num_classes: int,
        in_strides: Sequence[int],
        out_strides: Sequence[int],
        head_channels: OrderedDict,
        stride2channels: dict,
        upscale_bbox_pred: bool = True,
        feat_channels: int = 256,
        stacked_convs: int = 4,
        head_conv_kernel_size: int = 1,
        head_sep_conv: bool = False,
        use_sigmoid: bool = True,
        share_bn: bool = False,
        dequant_output: bool = True,
        int8_output: bool = True,
        share_conv: bool = True,
    ):
        super(MTFCOS3DHead, self).__init__()
        if upscale_bbox_pred:
            assert dequant_output, (
                "dequant_output should be True to convert "
                "QTensor to Tensor when upscale_bbox_pred is True"
            )
        self.num_classes = num_classes
        self.in_strides = sorted(_as_list(in_strides))
        self.head_channels = head_channels
        self.out_strides = sorted(_as_list(out_strides))
        assert set(self.out_strides).issubset(
            self.in_strides
        ), "out_strides must be a subset of in_strides"
        self.feat_indexs = [
            self.in_strides.index(stride) for stride in self.out_strides
        ]
        self.stride2channels = stride2channels
        self.in_channels = (
            [stride2channels[stride] for stride in self.in_strides]
            if not share_conv
            else stride2channels[self.in_strides[0]]
        )
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.head_conv_kernel_size = head_conv_kernel_size
        self.head_sep_conv = head_sep_conv
        self.use_sigmoid = use_sigmoid
        self.share_bn = share_bn
        self.upscale_bbox_pred = upscale_bbox_pred
        self.dequant_output = dequant_output
        self.int8_output = int8_output
        self.share_conv = share_conv
        self.single_relu = nn.ReLU(inplace=True)
        assert self.share_bn is False if self.share_conv is False else True
        assert (
            len(set(stride2channels.values())) != 1 and not self.share_conv
        ) or len(set(stride2channels.values())) == 1
        self.background_label = num_classes
        if use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
            raise NotImplementedError

        self._init_layers()
        self._init_weights()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.dequant = DeQuantStub()
        if self.share_conv:
            if self.share_bn:
                self._init_cls_convs()
                self._init_reg_convs()
            else:
                self._init_cls_reg_convs_with_independent_bn()  # noqa
            self._init_predictor()
        else:
            self._init_cls_no_shared_convs()
            self._init_reg_no_shared_convs()
            self._init_no_shared_predictor()

    def _init_cls_no_shared_convs(self):
        self.cls_convs_list = nn.ModuleList()
        for _ in range(self.stacked_convs):
            cls_convs = nn.ModuleList()
            for j in self.feat_indexs:
                in_chn = self.in_channels[j]
                cls_convs.append(
                    SeparableConvModule2d(
                        in_chn,
                        in_chn,
                        kernel_size=3,
                        padding=1,
                        pw_norm_layer=nn.BatchNorm2d(in_chn),
                        pw_act_layer=nn.ReLU(inplace=True),
                    )
                )
            self.cls_convs_list.append(cls_convs)

    def _init_reg_no_shared_convs(self):
        self.reg_convs_list = nn.ModuleList()
        for _ in range(self.stacked_convs):
            reg_convs = nn.ModuleList()
            for j in self.feat_indexs:
                in_chn = self.in_channels[j]
                reg_convs.append(
                    SeparableConvModule2d(
                        in_chn,
                        in_chn,
                        kernel_size=3,
                        padding=1,
                        pw_norm_layer=nn.BatchNorm2d(in_chn),
                        pw_act_layer=nn.ReLU(inplace=True),
                    )
                )
            self.reg_convs_list.append(reg_convs)

    def _init_no_shared_predictor(self):
        self.out_block_names = []
        for j in self.feat_indexs:
            in_chn = self.in_channels[j]
            for name, num_channel in self.head_channels.items():
                assert len(num_channel) >= 1, "head channel can't be empty!"
                if j == self.feat_indexs[0]:
                    setattr(self, name, nn.ModuleList())
                    self.out_block_names += [name]

                num_channel = _as_list(num_channel)
                block = []
                for _ in range(0, len(num_channel) - 1):
                    if self.head_sep_conv:
                        _block = SeparableConvModule2d(
                            in_channels=in_chn,
                            out_channels=in_chn,
                            kernel_size=3,
                            padding=1,
                            stride=1,
                            pw_norm_layer=nn.BatchNorm2d(in_chn),
                            pw_act_layer=nn.ReLU(inplace=True),
                        )
                    else:
                        _block = ConvModule2d(
                            in_chn,
                            out_channels=in_chn,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                            norm_layer=nn.BatchNorm2d(in_chn),
                            act_layer=nn.ReLU(inplace=True),
                        )
                    block.append(_block)
                block.append(
                    nn.Conv2d(
                        in_chn,
                        num_channel[-1],
                        self.head_conv_kernel_size,
                        1,
                        0,
                        groups=1,
                        bias=False,
                    )
                )
                block = nn.Sequential(*block)
                getattr(self, name).append(block)

    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                SeparableConvModule2d(
                    chn,
                    self.feat_channels,
                    kernel_size=3,
                    padding=1,
                    pw_norm_layer=nn.BatchNorm2d(self.feat_channels),
                    pw_act_layer=nn.ReLU(inplace=True),
                )
            )

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.reg_convs.append(
                SeparableConvModule2d(
                    chn,
                    self.feat_channels,
                    kernel_size=3,
                    padding=1,
                    pw_norm_layer=nn.BatchNorm2d(self.feat_channels),
                    pw_act_layer=nn.ReLU(inplace=True),
                )
            )

    def _init_cls_reg_convs_with_independent_bn(self):
        """Initialize convs of cls head and reg head.

        Depth-wise and point-wise convs are shared by all stride, but BN is
        independent, i.e. not shared, experiment shows that this will improve
        performance.
        """
        num_strides = len(self.out_strides)
        self.cls_convs = nn.ModuleList(
            [nn.ModuleList() for i in range(num_strides)]
        )
        self.reg_convs = nn.ModuleList(
            [nn.ModuleList() for i in range(num_strides)]
        )

        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            for j in range(num_strides):
                if j == 0:
                    # to create new conv
                    cls_shared_dw_conv = None
                    cls_shared_pw_conv = None
                    reg_shared_dw_conv = None
                    reg_shared_pw_conv = None
                else:
                    # share convs of the first out stride, not create
                    cls_shared_dw_conv = self.cls_convs[0][i][0][0]
                    cls_shared_pw_conv = self.cls_convs[0][i][1][0]
                    reg_shared_dw_conv = self.reg_convs[0][i][0][0]
                    reg_shared_pw_conv = self.reg_convs[0][i][1][0]

                # construct cls_convs
                if cls_shared_dw_conv is None:
                    self.cls_convs[j].append(
                        SeparableConvModule2d(
                            chn,
                            self.feat_channels,
                            kernel_size=3,
                            padding=1,
                            pw_norm_layer=nn.BatchNorm2d(self.feat_channels),
                            pw_act_layer=nn.ReLU(inplace=True),
                        )
                    )
                else:
                    self.cls_convs[j].append(
                        nn.Sequential(
                            cls_shared_dw_conv,
                            cls_shared_pw_conv,
                            nn.BatchNorm2d(self.feat_channels),
                            nn.ReLU(inplace=True),
                        )
                    )

                # construct reg_convs
                if reg_shared_dw_conv is None:
                    self.reg_convs[j].append(
                        SeparableConvModule2d(
                            chn,
                            self.feat_channels,
                            kernel_size=3,
                            padding=1,
                            pw_norm_layer=nn.BatchNorm2d(self.feat_channels),
                            pw_act_layer=nn.ReLU(inplace=True),
                        )
                    )
                else:
                    self.reg_convs[j].append(
                        nn.Sequential(
                            reg_shared_dw_conv,
                            reg_shared_pw_conv,
                            nn.BatchNorm2d(self.feat_channels),
                            nn.ReLU(inplace=True),
                        )
                    )

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.out_block_names = []
        for name, num_channel in self.head_channels.items():
            assert len(num_channel) >= 1, "head channel can't be empty!"
            num_channel = _as_list(num_channel)
            block = []
            for i in range(len(num_channel) - 1):
                if self.head_sep_conv:
                    _block = SeparableConvModule2d(
                        in_channels=self.feat_channels
                        if i == 0
                        else num_channel[i - 1],
                        out_channels=num_channel[i],  # self.feat_channels
                        kernel_size=3,
                        padding=1,
                        stride=1,
                        pw_norm_layer=nn.BatchNorm2d(self.feat_channels),
                        pw_act_layer=nn.ReLU(inplace=True),
                    )
                else:
                    _block = ConvModule2d(
                        in_channels=self.feat_channels
                        if i == 0
                        else num_channel[i - 1],
                        out_channels=num_channel[i],  # self.feat_channels
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        norm_layer=nn.BatchNorm2d(self.feat_channels),
                        act_layer=nn.ReLU(inplace=True),
                    )
                block.append(_block)

            # out module should set bias
            block.append(
                nn.Conv2d(
                    self.feat_channels
                    if len(num_channel) == 1
                    else num_channel[-2],
                    num_channel[-1],
                    self.head_conv_kernel_size,
                    1,
                    padding=self.head_conv_kernel_size // 2,
                    groups=1,
                    bias=True,
                )
            )
            block = nn.Sequential(*block)
            setattr(self, name, block)
            self.out_block_names += [name]

    def _init_dfs(self, module, bias_cls=None):
        if isinstance(module, (list, nn.Sequential, nn.ModuleList)):
            for m in module:
                self._init_dfs(m, bias_cls)
        elif isinstance(module, nn.Conv2d):
            if bias_cls:
                normal_init(module, std=0.01, bias=bias_cls)
            else:
                normal_init(module, std=0.01)
        elif isinstance(module, (nn.BatchNorm2d, nn.ReLU)):
            pass
        else:
            warnings.warn(
                f"_init_dfs func get instance of {module}" f"is not expected!"
            )

    def _init_weights_out_block(self):
        for name in self.out_block_names:
            bias_cls = None
            # hard code
            if name in ["hm", "cls"]:
                bias_cls = bias_init_with_prob(0.01)
            out_module = getattr(self, name)
            self._init_dfs(out_module, bias_cls)

    def _init_weights(self):
        """Initialize weights of the head."""
        if self.share_conv:
            self._init_dfs(self.cls_convs)
            self._init_dfs(self.reg_convs)
        else:
            self._init_dfs(self.cls_convs_list)
            self._init_dfs(self.reg_convs_list)
        self._init_weights_out_block()

    def forward_single(self, x: Tensor, i: int, stride: int) -> Tuple:
        """Forward features of a single scale levle.

        Args:
            x: FPN feature maps of the specified stride.
            i: Index of feature level.
            stride: The corresponding stride for feature maps, only used to
                upscale bbox pred when self.upscale_bbox_pred is True.
        Returns:
            Scores for each class, bbox and direction class predictions,
                centerness predictions of input feature maps.
        """
        cls_feat = x
        reg_feat = x
        if self.share_conv:
            if self.share_bn:
                for cls_layer in self.cls_convs:
                    cls_feat = cls_layer(cls_feat)
            else:
                for cls_layer in self.cls_convs[i]:
                    cls_feat = cls_layer(cls_feat)

            if self.share_bn:
                for reg_layer in self.reg_convs:
                    reg_feat = reg_layer(reg_feat)
            else:
                for reg_layer in self.reg_convs[i]:
                    reg_feat = reg_layer(reg_feat)
        else:
            for cls_layer in self.cls_convs_list:
                cls_feat = cls_layer[i](cls_feat)

            for reg_layer in self.reg_convs_list:
                reg_feat = reg_layer[i](reg_feat)

        out = OrderedDict()
        for name in self.out_block_names:
            block = getattr(self, name)
            bbox_offset_flag = False
            # only one cls feature out
            if "cls" == name:
                feat = cls_feat
            elif "dir_reg" == name:
                feat = cls_feat
            elif "reg" in name:
                feat = reg_feat
            else:
                raise NotImplementedError(
                    f"output keys must contains *cls*/*reg*" f"but got {name}"
                )

            # point offset, default: ltrb/ctr3d
            if "offset" in name:
                bbox_offset_flag = True

            if isinstance(block, nn.ModuleList):
                block_out = block[i](feat)
            elif isinstance(block, nn.Sequential):
                block_out = block(feat)
            else:
                raise NotImplementedError(
                    f"head out branch only support type {nn.ModuleList}"
                    f"{nn.Sequential} but got {type(block)}"
                )

            if self.dequant_output:
                out[name] = self.dequant(block_out)
            else:
                out[name] = block_out

            if self.upscale_bbox_pred and bbox_offset_flag:
                # Only used in eval mode when upscale_bbox_pred = True.
                # Because the ele-mul operation is not supported currently,
                # this part will be conduct in filter after dequant
                assert not isinstance(
                    out[name][0], horizon.quantization.QTensor
                ), (
                    "QTensor not support multiply op, you can set "
                    "dequant_output=True to convert QTensor to Tensor"
                )
                out[name] *= stride

        return tuple(out.values())

    def forward(self, feats: Dict[str, Tensor]):
        """Forward features from the upstream network.

        Args:
            feats: Features from the upstream network, each is a 4D-tensor.

        Returns:
            A dict according to self.out_block_names.
        """

        feats = [_as_list(feats)[index] for index in self.feat_indexs]
        out_tuple = multi_apply(
            self.forward_single,
            feats,
            range(len(self.out_strides)),
            self.out_strides,
        )
        out = OrderedDict()
        for i, name in enumerate(self.out_block_names):
            out[name] = out_tuple[i]
        return out

    def fuse_model(self):
        def fuse_model_convs(convs):
            if self.share_conv:
                for modules in convs:
                    for m in modules:
                        if self.share_bn:
                            m.fuse_model()
                        else:
                            if isinstance(m, SeparableConvModule2d):
                                modules_to_fuse = [["1.0", "1.1", "1.2"]]
                            elif isinstance(m, nn.Sequential):
                                modules_to_fuse = [["1", "2", "3"]]
                            else:
                                raise NotImplementedError(
                                    f"Not support type{type(m)} to fuse"
                                )
                            horizon.quantization.fuse_conv_shared_modules(
                                m, modules_to_fuse, inplace=True
                            )
            else:
                for modules in convs:
                    for m in modules:
                        for n in m:
                            n.fuse_model()

        if self.share_conv:
            fuse_model_convs(self.cls_convs)
            fuse_model_convs(self.reg_convs)
        else:
            fuse_model_convs(self.cls_convs_list)
            fuse_model_convs(self.reg_convs_list)

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        # disable output quantization for last quanti layer.
        if not self.int8_output:
            for name, _ in self.head_channels.items():
                _out_head = getattr(self, name)
                _out_head.qconfig = (
                    qconfig_manager.get_default_qat_out_qconfig()
                )
