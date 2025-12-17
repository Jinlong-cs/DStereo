import pytest
import torch

from hat.registry import build_from_registry


@pytest.mark.parametrize(
    "stage",
    ["train", "test"],
)
def test_image_classifier(stage):
    task_name = "glass"
    if stage == "train":
        loss_func = torch.nn.BCEWithLogitsLoss(reduce=False)
    else:
        loss_func = None
    config = dict(
        type="ImageClassifier",
        backbone=dict(
            type="VargNetV2",
            num_classes=1000,
            model_type="tinyvargnetv2",
            include_top=False,
            bn_kwargs={},
        ),
        head=dict(
            type="FacequalityMultiHead",
            in_channels=256,
            feat_channels=[128, 128],
            output_dim=1,
            task_name=task_name,
            loss=loss_func,
        ),
    )
    model = build_from_registry(config)
    x = {
        "img": torch.rand(4, 3, 128, 128),
        f"gt_{task_name}": torch.Tensor([1, 0, 0, 1]),
    }
    output = model(x)
    pred = output["pred"]
    assert pred.shape == (4,)
