import torch.nn as nn
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY

__all__ = ["DynGestureHead"]


@OBJECT_REGISTRY.register
class DynGestureHead(nn.Module):
    """Head network of dyn gesture task.

    Args:
        num_classes: number of class.
            Defaults to 59.
        alpha: Alpha for tinyvargnetv2.
            Defaults to 0.5.
        flat_output: whether to view the output tensor.
            Defaults to True.
        use_pool: whether to using the pooling layer.
            Defaults to False.
        pool_size: Pooling size.
            Defaults to 7.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        num_classes: int = 59,
        flat_output: bool = True,
        use_pool: bool = False,
        pool_size: int = 7,
    ):
        super(DynGestureHead, self).__init__()
        self.num_classes = num_classes
        self.dequant = DeQuantStub()
        self.flat_output = flat_output
        self.use_pool = use_pool
        if self.use_pool:
            self.pool_layer = nn.AvgPool2d(pool_size)

        self.output = ConvModule2d(
            int(alpha * 256),
            num_classes,
            1,
            bias=True,
            norm_layer=nn.BatchNorm2d(num_classes),
        )

    def forward(self, x):
        if self.use_pool:
            x = self.pool_layer(x)
        x = self.output(x)
        x = self.dequant(x)
        if self.flat_output:
            x = x.view(-1, self.num_classes)
        return x

    def fuse_model(self):
        self.output.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        self.output.qconfig = qconfig_manager.get_default_qat_out_qconfig()
