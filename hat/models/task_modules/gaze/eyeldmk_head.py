# Copyright (c) Horizon Robotics. All rights reserved.
from copy import copy
from typing import Dict, List, Optional

import torch.nn as nn
from torch.quantization import DeQuantStub

from hat.models.base_modules.basic_resnet_module import BasicResBlock
from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.base_modules.extend_container import ExtSequential
from .eyeldmk_utils import HMBackboneModel

__all__ = ["EyeldmkHead"]


class EyeldmkHead(nn.Module):
    """Eye landmark pathway after backbone feature extraction.

    Use multiple regression branches to output eye landmark coords

    Args:
        feat_size: feature size for every single branch.
        params: basic model hyper-parameters.
    """

    def __init__(
        self, feat_size: int, params: Optional[Dict] = None, **kwargs
    ):
        super(EyeldmkHead, self).__init__()
        assert params, "eye_ldmk config params is None..."
        self.mbreg = EyeldmkMultiBranch(feat_size=feat_size, params=params)

    def forward(self, x):
        output_eyeldmk = self.mbreg(x["shared_feat"])
        return output_eyeldmk

    def fuse_model(self):
        if hasattr(self.mbreg, "fuse_model"):
            self.mbreg.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        if hasattr(self.mbreg, "set_qconfig"):
            self.mbreg.set_qconfig()


class EyeldmkMultiBranch(nn.Module):
    """Multu-branch regression model for eye landmarks prediction.

    Eye landmarks are divided into 6 groups: left eye lid, left eye iris,
    left eye pupil, right eye lid, right eye iris, right eye pupil.
    lid and iris has 8 landmarks, pupil has 5 landmarks.
    There are 6 branches for each group.

    Args:
        feat_size: feature size for every single branch.
        params: basic model hyper-parameters
    """

    def __init__(
        self, feat_size: int, params: Optional[Dict] = None, **kwargs
    ):
        super(EyeldmkMultiBranch, self).__init__()
        assert params, "EyeldmkMultiBranch config params is None..."
        name = kwargs.get("name", "eyeldmk_multi_branch")
        self.j5_efficient = params.get("J5_efficient", False)
        self.use_groupstyle = params.get("use_groupstyle", self.j5_efficient)

        SingleBranch = (
            EyeldmkSingleBranchJ5Efficient
            if self.j5_efficient
            else EyeldmkSingleBranch
        )
        if not self.j5_efficient and self.use_groupstyle:
            raise NotImplementedError

        # common branch
        self.lpupil_branch = SingleBranch(
            feat_size=feat_size,
            alpha=params["alpha"],
            bn_kwargs=params["bn_kwargs"],
            out_channels=params["channels"],
            is_pupil_pts=True,
            name=f"{name}_lpupil",
        )

        self.rpupil_branch = SingleBranch(
            feat_size=feat_size,
            alpha=params["alpha"],
            bn_kwargs=params["bn_kwargs"],
            out_channels=params["channels"],
            is_pupil_pts=True,
            name=f"{name}_rpupil",
        )

        # others
        self.llid_branch = (
            SingleBranch(
                feat_size=feat_size,
                alpha=params["alpha"],
                bn_kwargs=params["bn_kwargs"],
                out_channels=params["channels"],
                is_pupil_pts=False,
                name=f"{name}_llid",
            )
            if not self.use_groupstyle
            else None
        )

        self.liris_branch = (
            SingleBranch(
                feat_size=feat_size,
                alpha=params["alpha"],
                bn_kwargs=params["bn_kwargs"],
                out_channels=params["channels"],
                is_pupil_pts=False,
                name=f"{name}_liris",
            )
            if not self.use_groupstyle
            else None
        )

        self.rlid_branch = (
            SingleBranch(
                feat_size=feat_size,
                alpha=params["alpha"],
                bn_kwargs=params["bn_kwargs"],
                out_channels=params["channels"],
                is_pupil_pts=False,
                name=f"{name}_rlid",
            )
            if not self.use_groupstyle
            else None
        )

        self.riris_branch = (
            SingleBranch(
                feat_size=feat_size,
                alpha=params["alpha"],
                bn_kwargs=params["bn_kwargs"],
                out_channels=params["channels"],
                is_pupil_pts=False,
                name=f"{name}_riris",
            )
            if not self.use_groupstyle
            else None
        )

        self.group_branch = (
            SingleBranch(
                feat_size=feat_size,
                alpha=params["alpha"],
                bn_kwargs=params["bn_kwargs"],
                out_channels=params["channels"],
                is_pupil_pts=False,
                groups=4,
                name=f"{name}_group",
            )
            if self.use_groupstyle
            else None
        )

    def forward(self, x):
        lpupil = self.lpupil_branch(x)
        rpupil = self.rpupil_branch(x)
        if self.use_groupstyle:
            group_lmks = self.group_branch(x)
            return [group_lmks, lpupil, rpupil]
        else:
            llid = self.llid_branch(x)
            liris = self.liris_branch(x)
            rlid = self.rlid_branch(x)
            riris = self.riris_branch(x)
            return [llid, liris, lpupil, rlid, riris, rpupil]

    def fuse_model(self):
        modules = [
            self.llid_branch,
            self.liris_branch,
            self.lpupil_branch,
            self.rlid_branch,
            self.riris_branch,
            self.rpupil_branch,
            self.group_branch,
        ]
        for module in modules:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        for module in [
            self.llid_branch,
            self.liris_branch,
            self.lpupil_branch,
            self.rlid_branch,
            self.riris_branch,
            self.rpupil_branch,
            self.group_branch,
        ]:
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()


class EyeldmkSingleBranch(nn.Module):
    def __init__(
        self,
        name: str,
        feat_size: int,
        out_channels: List,
        alpha=0.25,
        bn_kwargs: Optional[Dict] = None,
        is_pupil_pts=False,
    ):
        super(EyeldmkSingleBranch, self).__init__()
        assert bn_kwargs, "EyeldmkSingleBranch bn_kwargs is None..."

        # pupil is 5 points, iris and lid is 8 points
        _out_channels = copy(out_channels)
        if is_pupil_pts:
            _out_channels[-1] = 10

        in_channel_list = [
            None,
            None,
            None,
            None,
            [int(x * alpha) for x in [64, 64, 64]],
            [int(x * alpha) for x in [48, 32]],
        ]
        out_channel_list = [
            None,
            None,
            None,
            None,
            [int(x * alpha) for x in [64, 64, 48]],
            [int(x * alpha) for x in [32, 32]],
        ]

        self.backbone = HMBackboneModel(
            feat_size=feat_size,
            alpha=alpha,
            bn_kwargs=bn_kwargs,
            start_point=4,
            end_point=6,
            use_bias=False,
            use_dw_relu=True,
            in_channel_list=in_channel_list,
            out_channel_list=out_channel_list,
            name=name,
        )
        self.output = ConvModule2d(
            in_channels=out_channel_list[-1][-1],
            out_channels=_out_channels[-1],
            kernel_size=1,
            stride=2,
            padding=0,
            groups=1,
            bias=True,
            norm_layer=None,
            act_layer=None,
        )
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.backbone(x)
        x = self.output(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        modules = [self.backbone, self.output]
        for module in modules:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        if hasattr(self.backbone, "set_qconfig"):
            self.backbone.set_qconfig()
        self.output.qconfig = qconfig_manager.get_default_qat_out_qconfig()


class ResBlock1x1(BasicResBlock):
    """ResBlock1x1 block for EyeldmkSingleBranchJ5Efficient.

    Args:
        _channels : Channel per group.
        bn_kwargs : BN kwargs.
        groups : Num groups.
        bias : Is use bias.
    """

    def __init__(
        self,
        _channels: int,
        bn_kwargs: dict,
        groups: int = 1,
        bias: bool = True,
    ):
        channels = _channels * groups
        super().__init__(
            channels, channels, bn_kwargs, stride=1, bias=bias, expansion=1
        )
        self.conv = nn.Sequential(
            ConvModule2d(
                channels,
                channels,
                1,
                padding=0,
                stride=1,
                groups=groups,
                bias=bias,
                norm_layer=nn.BatchNorm2d(channels, **bn_kwargs),
                act_layer=nn.ReLU(inplace=True),
            ),
            ConvModule2d(
                channels,
                channels,
                1,
                padding=0,
                stride=1,
                groups=groups,
                bias=bias,
                norm_layer=nn.BatchNorm2d(channels, **bn_kwargs),
            ),
        )


class EyeldmkSingleBranchJ5Efficient(EyeldmkSingleBranch):
    """EyeldmkSingleBranch Efficient for J5.

    Args:
        name : mod name.
        feat_size : Input channels.
        out_channels : Output channels.
        alpha : Network chennels scale.
        bn_kwargs : BN kwargs.
        groups : Num groups.
        is_pupil_pts : Is pupil branch.
    """

    def __init__(
        self,
        name: str,
        feat_size: int,
        out_channels: List,
        alpha: float = 0.25,
        bn_kwargs: dict = None,
        groups: int = 1,
        is_pupil_pts: bool = False,
    ):
        self.groups = groups
        super().__init__(
            name, feat_size, out_channels, alpha, bn_kwargs, is_pupil_pts
        )
        mid_channels = int(alpha * 512)
        assert mid_channels % 16 == 0
        _out_channels = 16
        if is_pupil_pts:
            _out_channels = 10
        self._out_channels = _out_channels

        backbone = []
        # 1024x3x5
        backbone.append(
            ConvModule2d(
                feat_size,
                mid_channels * groups,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                norm_layer=nn.BatchNorm2d(mid_channels * groups, **bn_kwargs),
                act_layer=nn.ReLU(),
            )
        )
        # mid_channelsx3x5
        backbone.append(
            ConvModule2d(
                mid_channels * groups,
                mid_channels * groups,
                kernel_size=3,
                stride=1,
                padding=0,
                groups=groups,
                norm_layer=nn.BatchNorm2d(mid_channels * groups, **bn_kwargs),
                act_layer=nn.ReLU(),
            )
        )

        backbone.append(ResBlock1x1(mid_channels, bn_kwargs, groups=groups))
        # mid_channelsx1x3
        backbone.append(
            ConvModule2d(
                mid_channels * groups,
                mid_channels * groups,
                kernel_size=(1, 3),
                stride=1,
                padding=0,
                groups=groups,
                norm_layer=nn.BatchNorm2d(mid_channels * groups, **bn_kwargs),
                act_layer=nn.ReLU(),
            )
        )
        backbone.append(ResBlock1x1(mid_channels, bn_kwargs, groups=groups))
        self.backbone = ExtSequential(backbone)
        self.output = ConvModule2d(
            mid_channels * groups,
            _out_channels * groups,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=groups,
            norm_layer=None,
            act_layer=None,
        )
