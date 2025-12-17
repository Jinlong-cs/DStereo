import torch

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test


def test_adaptivehead():
    config = dict(
        type="FasClassifier",
        backbone=dict(
            type="VargNetV2",
            num_classes=1000,
            model_type="tinyvargnetv2",
            include_top=False,
            bn_kwargs={},
        ),
        head=dict(
            type="FasAdaptiveHead",
            in_channels=256,
            database_labels=[
                0,
            ],
        ),
    )
    model = build_from_registry(config)
    x = {
        "img": torch.rand(4, 3, 112, 112),
        "fas_label": None,
        "database_labels": None,
    }
    output = model(x)
    assert isinstance(output, list)
    assert len(output) == 1
    assert output[0].shape == (4, 1, 1, 1)
    qat_test(model, x, with_quantized=False)
