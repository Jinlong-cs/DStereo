# Copyright (c) Horizon Robotics. All rights reserved.
import copy
import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY
from .lidar_base_head import SepHead

logger = logging.getLogger(__file__)

__all__ = ["AfdetHead"]


@OBJECT_REGISTRY.register_module
class AfdetHead(nn.Module):
    def __init__(
        self,
        in_channels: List[int] = None,
        bn_kwargs: Dict[str, Any] = None,
        num_class: int = 3,
        common_heads=None,
        init_bias: float = -2.19,
        use_share_conv: bool = True,
        share_conv_channel: int = 64,
        num_hm_conv: int = 2,
        input_idx: int = 0,
        hm_key: str = "hm",
        quantize: bool = False,
        use_stride_feats: bool = False,
        channel_factor: int = 1,
        use_relu6: bool = False,
        us_factor: int = 1,
        split_hm_head: bool = False,
        quant_output: bool = True,
    ):
        """Anchor free detection Head module.

        Args:
            in_channels: A list of to indicates the input channels of the
                block.
            bn_kwargs: Batch norm arguments.
            tasks: head be responsible for tasks.
            common_heads: common heads.
            init_bias: bias init or not.
            use_share_conv: use share conv parameters or not.
            share_conv_channel: share conv channel.
            num_hm_conv: num hm conv.
            input_idx:which input to foward.
            hm_key:str:the name of heatmap.
            use_stride_feats: use middle feature or not.
            channel_factor:factor of channel(z_axis).
            use_relu6:use relu6 or not.
            us_factor:  upsampling factor.
            split_hm_head:split heatmap head or not.
            use_mutil_tasks_feats:if use_mutil_tasks_feats.
                if it is true,foward can support mutil input feature.
            quant_output: whether to do quantization on output.
        """
        super(AfdetHead, self).__init__()
        self.use_stride_feats = use_stride_feats
        self.in_channels = in_channels
        self.num_classes = num_class
        self.use_relu6 = use_relu6
        self.act_layer = (
            nn.ReLU6(inplace=True) if self.use_relu6 else nn.ReLU(inplace=True)
        )

        self.input_idx = input_idx

        self._z_axis_factor = channel_factor
        # change this if your box is different
        self.box_n_dim = 7

        if bn_kwargs is None:
            bn_kwargs = {"eps": 1e-3, "momentum": 0.01}
        self._bn_kwargs = bn_kwargs

        logger.info(f"Detection number class: {num_class}")

        # a shared convolution
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
                    act_layer=self.act_layer,
                )
                last_in_channels = share_conv_channel
        else:
            self.shared_conv = None
            last_in_channels = share_conv_channel

        self.hm_key = hm_key
        self.split_hm_head = split_hm_head and hm_key == "hm"
        heads = copy.deepcopy(common_heads)
        if self.split_hm_head:
            self.hm_keys = []
            for i in range(num_class):
                self.hm_keys.append(f"hm_{i}")
                heads.update({f"hm_{i}": (1, num_hm_conv)})
        else:
            heads.update(
                {hm_key: (num_class * self._z_axis_factor, num_hm_conv)}
            )
        self.head_module = SepHead(
            last_in_channels,
            heads,
            bn=True,
            init_bias=init_bias,
            final_kernel=3,
            bn_kwargs=self._bn_kwargs,
            quantize=quantize,
            upsample_factor=us_factor,
        )

        self.quant_output = quant_output

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        out_qconfig = qconfig_manager.get_default_qat_out_qconfig()

        for m in self.modules():
            if isinstance(m, SepHead):
                m.set_qconfig(out_qconfig, self.quant_output)

    def fuse_model(self):
        if self.shared_conv:
            self.shared_conv.fuse_model()
        for m in self.modules():
            if isinstance(m, SepHead):
                m.fuse_model()

    def forward(
        self,
        input_tensor: Optional[torch.Tensor] = None,
        stride_feats: Optional[List[torch.Tensor]] = None,
        grid: Optional[torch.Tensor] = None,
    ) -> List[Dict]:

        if self.use_stride_feats:
            input_tensor = stride_feats[self.input_idx]
        if self.shared_conv:
            x = self.shared_conv(input_tensor)

        ret = self.head_module(x, grid=grid)
        return ret
