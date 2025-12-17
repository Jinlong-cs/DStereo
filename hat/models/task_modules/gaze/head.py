# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Dict, List, Optional

from torch import nn

from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY
from .eyeldmk_head import EyeldmkHead
from .gaze_head import GazeHead

__all__ = ["GazeEyeldmkHead"]


@OBJECT_REGISTRY.register
class GazeEyeldmkHead(nn.Module):
    def __init__(
        self,
        input_size: List,
        last_channel_out: int,
        shared_feat_size: int,
        gaze_head_params: Optional[Dict] = None,
        eyeldmk_head_params: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        super(GazeEyeldmkHead, self).__init__()

        bn_kwargs = kwargs.get(
            "bn_kwargs",
            {
                "eps": 1e-05,
                "momentum": 0.9,
            },
        )

        self.neck = ConvModule2d(
            last_channel_out,
            shared_feat_size,
            1,
            bias=False,
            norm_layer=nn.BatchNorm2d(shared_feat_size, **bn_kwargs),
            act_layer=nn.ReLU(inplace=True),
        )
        self.downsample = DownSampleBlock(input_size=input_size)

        self.gaze_head = GazeHead(
            channels=gaze_head_params["channels"],
            bn_kwargs=gaze_head_params["bn_kwargs"],
            use_pool=gaze_head_params["use_pool"],
            output_add_bias=gaze_head_params["output_add_bias"],
            dropout_ratio=gaze_head_params["dropout_ratio"],
        )

        self.eyeldmk_head = EyeldmkHead(shared_feat_size, eyeldmk_head_params)

    def forward(self, x):
        head_input = x
        shared_feat = self.downsample(
            self.neck(head_input["last_backbone_feat"])
        )
        extend_input = {"shared_feat": shared_feat}
        gaze_output = self.gaze_head(extend_input)
        eyeldmk_output = self.eyeldmk_head(extend_input)

        head_output = {
            "gaze": gaze_output["gaze"],
            "eye_ldmk": eyeldmk_output,
        }

        return head_output

    def fuse_model(self):
        modules = [self.downsample, self.gaze_head, self.eyeldmk_head]
        self.neck.fuse_model()
        for module in modules:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        for module in [self.downsample, self.gaze_head, self.eyeldmk_head]:
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()


class DownSampleBlock(nn.Module):
    """DownSampleBlock for gaze feature.

    Args:
        input_size : Input data size.
    """

    def __init__(self, input_size, **kwargs):
        super(DownSampleBlock, self).__init__()
        self.input_size = input_size

        self.downsample_flag = False
        if self.input_size == [320, 192]:
            self.down_sample = nn.AvgPool2d(
                kernel_size=(2, 2),
                stride=(2, 2),
                padding=(0, 0),
                ceil_mode=True,
            )
            self.downsample_flag = True
        else:
            self.down_sample = None

    def forward(self, x):
        conv_feat = x
        if self.downsample_flag:
            return self.down_sample(conv_feat)  # (B, 1024, 3, 5)
        return conv_feat

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
