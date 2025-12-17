from horizon_plugin_pytorch.quantization import QuantStub
from torch import nn
from torch.quantization import DeQuantStub

from hat.models.base_modules.basic_vargnet_module import ExtendVarGNetFeatures
from hat.registry import OBJECT_REGISTRY

__all__ = ["WorkConditionNeck"]


@OBJECT_REGISTRY.register
class WorkConditionNeck(nn.Module):
    """Control the ExtendVarGNetFeatures from which stage, the neck of the work condition model.

    Args:
        in_channel: Input channels.
        o_channel: Channels of output featuers.
        end_feature_idx: End stage index of the feature gives to ExtendVarGNetFeatures.
        num_units: The number of units of each extend stride.
        group_base: The number of channels per group.
        bn_kwargs: BatchNormEx kwargs.
    """  # noqa

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
        super(WorkConditionNeck, self).__init__()
        self.extra_layers = ExtendVarGNetFeatures(
            prev_channel=in_channel,
            channels=o_channel,
            num_units=num_units,
            group_base=group_base,
            bn_kwargs=bn_kwargs,
        )
        self.end_feature_idx = end_feature_idx
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.disable_quanti_input = disable_quanti_input

    def forward(self, x):
        in_feat = x[: self.end_feature_idx]
        in_feat = [
            i if self.disable_quanti_input else self.quant(i) for i in in_feat
        ]
        in_feat = self.extra_layers(in_feat)
        in_feat = [self.dequant(i) for i in in_feat]
        return in_feat

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

    def fuse_model(self):
        self.extra_layers.fuse_model()
