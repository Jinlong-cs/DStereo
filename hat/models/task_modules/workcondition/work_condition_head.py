from typing import List, Optional, Tuple, Union

import torch
from horizon_plugin_pytorch.quantization import FixedScaleObserver, QuantStub
from torch import nn
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "WorkConditionClsHead",
]


@OBJECT_REGISTRY.register
class WorkConditionClsHead(nn.Module):
    """Heads module of workcondition.

    Args:
        output_dim: Output dimension.
        bn_kwargs: BatchNormEx kwargs.
        num_classes: Num classes of workcondition task.
        in_channel: Input channels.
        avg_pool_size: The size of AvgPool2d.
        padding_size: The padding size.
        bias: Whether to use bias in module.
        disable_quanti_input: Whether to distable quanti input.
        final_layer_out_shift: The scale of final layer shift.
        use_gn: Whether to use group normalization in some conv layers.
    """  # noqa

    def __init__(
        self,
        output_dim: int,
        bn_kwargs: dict,
        num_classes: int,
        in_channel: int,
        avg_pool_size: Optional[Union[int, Tuple]] = None,
        padding_size: int = 0,
        bias: bool = True,
        disable_quanti_input: bool = False,
        final_layer_out_shift: int = 4,
        use_gn: bool = False,
    ):
        super(WorkConditionClsHead, self).__init__()
        self.output_dim = output_dim
        self.num_classes = num_classes
        self.disable_quanti_input = disable_quanti_input
        self.final_layer_out_shift = final_layer_out_shift
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        if use_gn:
            norm_layer = nn.GroupNorm(32, output_dim)
        else:
            norm_layer = nn.BatchNorm2d(output_dim, **bn_kwargs)

        self.output = nn.Sequential(
            ConvModule2d(
                in_channel,
                output_dim,
                kernel_size=(3, 3),
                stride=2,
                padding=(1, padding_size),
                norm_layer=norm_layer,
                act_layer=nn.ReLU(inplace=True),
            ),
            nn.AdaptiveAvgPool2d((1, 1))
            if avg_pool_size is None
            else nn.AvgPool2d(avg_pool_size),
            ConvModule2d(
                output_dim,
                num_classes,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=bias,
            ),
        )

    def forward(self, x: List[torch.Tensor]):
        x = x[-1]
        x = x if self.disable_quanti_input else self.quant(x)
        x = self.output(x)
        x = self.dequant(x)
        return x

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        for m in self.output[-1]:
            m.qconfig = qconfig_manager.get_default_qat_out_qconfig()

    def set_calibration_qconfig(self):
        import horizon_plugin_pytorch

        qcfg = horizon_plugin_pytorch.quantization.get_default_calib_qconfig(
            calib_qkwargs={
                "observer": FixedScaleObserver,
                "scale": 1 / 2 ** self.final_layer_out_shift,
            }
        )
        # disable output quantization for last quanti layer.
        for m in self.output[-1]:
            m.qconfig = qcfg

    def fuse_model(self):
        for module in self.output:
            if hasattr(module, "fuse_model"):
                module.fuse_model()
