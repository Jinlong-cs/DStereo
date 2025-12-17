# Copyright (c) Horizon Robotics. All rights reserved.
import copy
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import horizon_plugin_pytorch.nn as hnn
import torch
import torch.nn as nn
from horizon_plugin_pytorch.nn.quantized import FloatFunctional
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.losses.lidar_losses import (
    LidarFastFocalLoss,
    LidarRegLoss,
    LidarSmoothL1RegLoss,
)
from hat.models.weight_init import kaiming_init


class SepHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        heads: Dict[str, Tuple[int, int]],
        head_conv: int = 64,
        final_kernel: int = 1,
        bn: bool = False,
        bn_kwargs: Dict[str, Any] = None,
        init_bias: float = -2.19,
        quantize: bool = False,
        upsample_factor: int = 1,
    ):
        """Separate head for different tasks that provids more design freedom.

        Args:
            in_channels (int): input number of channels.
            heads (Dict[str, Tuple[int, int]]): A dict of head configs. Each
                key is the name of the head, and each value is a tuple of
                (num_classes, num_conv_layers).
            head_conv (int, optional): conv channel size. Defaults to 64.
            final_kernel (int, optional): final kernel size. Defaults to 1.
            bn (bool, optional): whether to use batch norm. Defaults to False.
            bn_kwargs (Dict[str, Any], optional): batch norm parameters.
                Defaults to None.
            init_bias (float, optional): initialization bias.
                Defaults to -2.19.
            quantize (bool, optional): whether to quantize this module.
                Defaults to False.
            upsample_factor (int, optional): upsampling factor. Defaults to 1.
        """
        super(SepHead, self).__init__()

        self.dequant = DeQuantStub() if quantize else None
        self.floatmod_addscalar = FloatFunctional()
        self.floatmod_mulscalar = FloatFunctional()
        self.floatmod_mul = FloatFunctional()

        bn_kwargs = {"eps": 1e-3, "momentum": 0.01} if bn else {}

        self.heads = heads
        for head in self.heads:
            classes, num_conv = self.heads[head]
            layers = []
            for _ in range(num_conv - 1):
                layers.append(
                    ConvModule2d(
                        in_channels,
                        head_conv,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        bias=True,
                        norm_layer=nn.BatchNorm2d(head_conv, **bn_kwargs),
                        act_layer=nn.ReLU(inplace=True),
                    )
                )
            if upsample_factor > 1:
                layers.append(
                    hnn.Interpolate(
                        scale_factor=upsample_factor,
                        recompute_scale_factor=True,
                    )
                )
            layers.append(
                nn.Conv2d(
                    head_conv,
                    classes,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    bias=True,
                )
            )
            if "hm" in head:
                layers[-1].bias.data.fill_(init_bias)
            else:
                for m in layers:
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)
            fc = nn.Sequential(*layers)
            self.__setattr__(head, fc)

    def forward(
        self, x: torch.Tensor, grid: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the module."""
        ret_dict = {}
        for head in self.heads:
            ret = self.__getattr__(head)(x)
            if grid is not None:
                sub_grid = self.floatmod_addscalar.add_scalar(grid, -1)
                no_grid = self.floatmod_mulscalar.mul_scalar(sub_grid, -1)
                ret = self.floatmod_mul.mul(no_grid, ret)
            if self.dequant:
                ret = self.dequant(ret)
            ret_dict[head] = ret

        return ret_dict

    def set_qconfig(self, out_qconfig=None, quant_output=True) -> None:
        """Set quantization config."""
        for head in self.heads:
            fc = self.__getattr__(head)
            if quant_output:
                fc[-1].qconfig = out_qconfig

    def fuse_model(self):
        """Fuse model layers."""
        for m in self.modules():
            if isinstance(m, ConvModule2d):
                m.fuse_model()


class LidarBaseHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        bn_kwargs: Optional[Dict[str, Any]] = None,
        tasks: Optional[List[Dict[str, Any]]] = None,
        weight: float = 0.25,
        weight_iou: Optional[List[float]] = None,
        code_weights: Optional[List[float]] = None,
        common_heads: Optional[Dict[str, Tuple[int, int]]] = None,
        init_bias: float = -2.19,
        use_share_conv: bool = True,
        share_conv_channel: int = 64,
        num_hm_conv: int = 2,
        hm_key: str = "hm",
        quantize: bool = False,
        with_upsample: bool = False,
        upsample_factor: int = 1,
    ):
        """Task head base class for all other lidar heads.

        Args:
            in_channels (int): input number of channels.
            bn_kwargs (Optional[Dict[str, Any]], optional): batch norm params.
                Defaults to None.
            tasks (Optional[List[Dict[str, Any]]], optional): task configs.
                e.g. [dict(num_class=2, class_names=["VEHICLE", "CYCLIST])]
                Defaults to None.
            weight (float, optional): weight for localization loss.
                Defaults to 0.25.
            weight_iou (Optional[List[float]], optional): weight for iou
                related loss. Defaults to None.
            code_weights (Optional[List[float]], optional): weight for each
                position in localization loss. Defaults to None.
            common_heads (Optional[Dict[str, Tuple[int, int]]], optional):
                common head configs. Defaults to None.
            init_bias (float, optional): conv layer initialization bias.
                Defaults to -2.19.
            use_share_conv (bool, optional): whether to use shared convolution.
                before head separation. Defaults to True.
            share_conv_channel (int, optional): shared convolution channels.
                Defaults to 64.
            num_hm_conv (int, optional): number of heatmap convolution layers.
                Defaults to 2.
            hm_key (str, optional): string used as heatmap identifier.
                Defaults to "hm".
            quantize (bool, optional): whether to quantize this module.
                Defaults to False.
            with_upsample (bool, optional): whether this module should contain
                upsampling layer. Defaults to False.
            upsample_factor (int, optional): upsamplng ration to use, if
                with_upsample is set to True. Defaults to 1.
        """
        super(LidarBaseHead, self).__init__()

        num_classes = [(t["num_class"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.code_weights = code_weights
        self.weight = weight  # weight for localisation loss
        self.weight_iou = weight_iou

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.hm_key = hm_key
        self.crit = LidarFastFocalLoss()
        self.crit_reg = LidarRegLoss()

        self.crit_iou = LidarSmoothL1RegLoss()
        self.loss_aux = None

        if bn_kwargs is None:
            bn_kwargs = {"eps": 1e-3, "momentum": 0.01}
        self._bn_kwargs = bn_kwargs

        # whether to use shared convolution
        if use_share_conv:
            if isinstance(share_conv_channel, (list, tuple)):
                layers = []
                last_in_channels = in_channels
                for num_filters in share_conv_channel:
                    layers.append(
                        ConvModule2d(
                            last_in_channels,
                            num_filters,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                            norm_layer=nn.BatchNorm2d(
                                num_filters, **bn_kwargs
                            ),
                            act_layer=nn.ReLU(inplace=True),
                        )
                    )
                    last_in_channels = num_filters
                self.shared_conv = nn.Sequential(*layers)
            else:
                assert isinstance(share_conv_channel, int)
                self.shared_conv = ConvModule2d(
                    in_channels,
                    share_conv_channel,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                    norm_layer=nn.BatchNorm2d(share_conv_channel, **bn_kwargs),
                    act_layer=nn.ReLU(inplace=True),
                )
                last_in_channels = share_conv_channel
        else:
            self.shared_conv = None
            last_in_channels = share_conv_channel

        self.tasks = nn.ModuleList()

        if not with_upsample:
            upsample_factor = 1
        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            heads.update({hm_key: (num_cls, num_hm_conv)})
            self.tasks.append(
                SepHead(
                    last_in_channels,
                    heads,
                    bn=True,
                    init_bias=init_bias,
                    final_kernel=3,
                    bn_kwargs=self._bn_kwargs,
                    quantize=quantize,
                    upsample_factor=upsample_factor,
                )
            )

    def forward(self, x: torch.Tensor):
        """Forward pass of the module."""
        ret_dicts = []
        if self.shared_conv:
            x = self.shared_conv(x)
        for task in self.tasks:
            ret_dicts.append(task(x))
        return ret_dicts

    @abstractmethod
    def loss(self, *args, **kwargs):
        raise NotImplementedError("Implement loss in child modules.")

    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError("Implement predict in child modules.")
