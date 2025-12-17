import torch.nn as nn
from torch.quantization import DeQuantStub

from hat.models.backbones.resnet import ResNet18
from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY
from hat.utils import qconfig_manager
from hat.utils.model_helpers import fx_wrap

__all__ = ["ToyResNet18Classifier"]


@OBJECT_REGISTRY.register
class ToyResNet18Classifier(nn.Module):
    def __init__(self, num_classes, loss=None, quanti_head=True):
        super(ToyResNet18Classifier, self).__init__()
        self.backbone = ResNet18(num_classes, bn_kwargs={}, include_top=False)
        self.head = nn.Sequential(
            nn.AvgPool2d(7),
            ConvModule2d(512, num_classes, 1, bias=True),
        )
        self.num_classes = num_classes

        self.quanti_head = quanti_head
        self.loss = loss
        self.dequant = DeQuantStub()

    @fx_wrap()
    def _post_process(self, preds, target):
        preds = preds.view(-1, self.num_classes)
        loss = self.loss(preds, target)
        return preds, loss

    def forward(self, data):
        x = data["img"]
        target = data.get("labels", None)

        x = self.backbone(x)[-1]
        if self.quanti_head:
            x = self.head(x)
            x = self.dequant(x)
            if self.loss is None:
                return x
        else:
            x = self.dequant(x)
            if self.loss is None:
                return x
            x = self.head(x)
        return self._post_process(x, target)

    def fuse_model(self):
        self.backbone.fuse_model()
        if self.quanti_head:
            for module in self.head:
                if hasattr(module, "fuse_model"):
                    module.fuse_model()

    def set_qconfig(self):
        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        if not self.quanti_head:
            self.head.qconfig = None
        if self.loss is not None:
            self.loss.qconfig = None

    def set_calibration_qconfig(self):
        self.calibration_qconfig = (
            qconfig_manager.get_default_calibration_qconfig()
        )
        if not self.quanti_head:
            self.head.qconfig = None
        if self.loss is not None:
            self.loss.qconfig = None
