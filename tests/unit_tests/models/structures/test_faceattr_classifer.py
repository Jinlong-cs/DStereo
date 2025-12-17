import torch

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test


def test_faceattr_classifier():
    config = dict(
        type="FaceAttrClassifier",
        backbone=dict(
            type="FaceIDLargeVargNet",
            bn_kwargs=dict(eps=1e-3, momentum=0.01),
            bias=False,
            embedding_size=256,
            dropout=0.0,
            use_fp16=False,
            include_all=True,
        ),
        head=dict(
            type="FaceAttrHead",
            in_channels=768,
            age_classes=85,
            gender_classes=1,
            bn_kwargs={},
        ),
    )
    faceattr_model = build_from_registry(config)
    x = {
        "img": torch.rand(4, 3, 112, 112),
        "age": None,
        "gender": None,
        "ord_age": None,
    }
    output = faceattr_model(x)
    assert "pred_faceid" in output
    assert "pred_gender" in output
    assert "pred_age" in output
    qat_test(faceattr_model, x, with_quantized=False)
