from horizon_plugin_pytorch.dtype import qinfo
from horizon_plugin_pytorch.quantization import (
    FakeQuantize,
    FixedScaleObserver,
    QuantStub,
    default_weight_8bit_fake_quant,
)
from torch import nn
from torch.quantization import DeQuantStub, QConfig

from hat.models.base_modules.basic_vargnet_module import ExtendVarGNetFeatures
from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "TinyVarGNetV2ClassificationHead",
    "ClassificationExtendHead",
    "ClassificationTrackHead",
]


@OBJECT_REGISTRY.register
class ClassificationExtendHead(nn.Module):
    def __init__(
        self,
        in_channel: int,
        o_channel: int,
        end_feature_idx: int,
        num_units: int,
        group_base: int,
        bn_kwargs: dict,
        disable_quanti_input: bool = False,
    ):
        super(ClassificationExtendHead, self).__init__()
        self.extra_layers = ExtendVarGNetFeatures(
            prev_channel=in_channel,
            channels=o_channel,
            num_units=num_units,
            group_base=group_base,
            bn_kwargs=bn_kwargs,
        )
        self.end_feature_idx = end_feature_idx
        self.disable_quanti_input = disable_quanti_input

    def forward(self, x):
        in_feat = x[: self.end_feature_idx]
        in_feat = self.extra_layers(in_feat)
        return in_feat

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

    def fuse_model(self):
        self.extra_layers.fuse_model()


@OBJECT_REGISTRY.register
class ClassificationTrackHead(nn.Module):
    """Head of classification task for outputing tracking feature."""

    def __init__(
        self,
    ):
        super(ClassificationTrackHead, self).__init__()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.dequant(x[-1])
        return x

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

    def fuse_model(self):
        pass


@OBJECT_REGISTRY.register
class TinyVarGNetV2ClassificationHead(nn.Module):
    """Classification head for TinyVarGNetV2.

    Args:
        input_channels: Input channels of first conv.
        num_classes: Number of classes.
        gc_group_base: Group base of group conv.
        alpha: Float alpha.
        cls_pooling_stride: Stride of first conv.
        cls_pooling_padding: Padding of first conv.
        avg_pool_size: Average pooling size.
        factor: Int factor.
        disable_quanti_input: Whether quanti input.
        final_layer_out_shift: The scale of final layer shift.
        export_model: Whether export model.
        output_by_argmax: Whether argmax output when outputing model.
        is_flatten_output: Whether flatten output.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        gc_group_base: int,
        bn_kwargs: dict,
        alpha: float,
        cls_pooling_stride: int or tuple = 2,
        cls_pooling_padding: int or tuple = 1,
        avg_pool_size: int = None,
        factor: int = 1,
        disable_quanti_input: bool = False,
        final_layer_out_shift: int = 4,
        export_model: bool = False,
        output_by_argmax: bool = True,
        is_flatten_output: bool = True,
    ):
        super(TinyVarGNetV2ClassificationHead, self).__init__()
        c1 = int(input_channels * alpha * factor)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.disable_quanti_input = disable_quanti_input
        self.final_layer_out_shift = final_layer_out_shift
        assert final_layer_out_shift == 4

        self.export_model = export_model
        self.output_by_argmax = output_by_argmax
        self.is_flatten_output = is_flatten_output
        self.conv1 = ConvModule2d(
            input_channels,
            c1,
            kernel_size=(3, 3),
            stride=cls_pooling_stride,
            padding=cls_pooling_padding,
            groups=int(input_channels / gc_group_base),
            bias=False,
            norm_layer=None,
            act_layer=nn.ReLU(inplace=True),
        )
        self.avg_pool_size = avg_pool_size
        if avg_pool_size is not None:
            self.pool = nn.AvgPool2d(avg_pool_size)
        self.conv2 = ConvModule2d(
            c1,
            num_classes,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )

    def forward(self, x):
        x = x[-1]
        x = x if self.disable_quanti_input else self.quant(x)

        x = self.conv1(x)
        if self.avg_pool_size is not None:
            x = self.pool(x)
        x = self.conv2(x)
        if self.export_model:
            if self.output_by_argmax:
                x, _ = x.max(dim=1, keepdim=True)
            x = self.dequant(x)
        else:
            x = self.dequant(x)
            if self.is_flatten_output:
                x = x.flatten(1)
        return x

    def set_calibration_qconfig(self):
        import horizon_plugin_pytorch

        qcfg = horizon_plugin_pytorch.quantization.get_default_calib_qconfig(
            calib_qkwargs={
                "observer": FixedScaleObserver,
                "scale": 1 / 2 ** self.final_layer_out_shift,
            }
        )
        # disable output quantization for last quanti layer.
        for m in self.conv2:
            m.qconfig = qcfg

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        # fix shift for last quanti layer.
        qcfg = QConfig(
            activation=FakeQuantize.with_args(
                observer=FixedScaleObserver,
                quant_min=qinfo("qint8").min,
                quant_max=qinfo("qint8").max,
                dtype="qint8",
                scale=1 / 2 ** self.final_layer_out_shift,
            ),
            weight=default_weight_8bit_fake_quant,
        )
        for m in self.conv2:
            m.qconfig = qcfg

    def fuse_model(self):
        for module in [self.conv1, self.conv2]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()
