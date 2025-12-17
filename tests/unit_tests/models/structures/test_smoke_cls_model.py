import pytest
import torch

from hat.models.losses.softmax_ce_loss import SoftmaxCELoss
from hat.models.structures.smoke_cls_model import SmokeClsModel
from hat.registry import build_from_registry


@pytest.mark.parametrize(
    [
        "losses",
    ],
    [pytest.param(SoftmaxCELoss())],
)
def test_SmokeClsModel(losses):

    batch_size = 32
    input_size = 128
    num_classes = 3
    x = torch.randn((batch_size, 3, input_size, input_size))
    data = {"img": x, "labels": torch.zeros([batch_size, num_classes])}

    cfg = dict(
        type="TinyVargNetV2",
        bn_kwargs={},
        num_classes=num_classes,
        alpha=1.0,
        group_base=4,
        flat_output=False,
        include_top=False,
    )
    backbone = build_from_registry(cfg)
    losses = SoftmaxCELoss()
    network = SmokeClsModel(backbone, losses, num_classes)
    preds, losses = network(data)
    assert preds.shape == (batch_size, num_classes)
    assert isinstance(preds, torch.Tensor)
    assert isinstance(losses, torch.Tensor)
