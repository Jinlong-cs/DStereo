import pytest
import torch

from hat.registry import build_from_registry


@pytest.mark.parametrize(
    "stage",
    ["train", "test"],
)
def test_facequality_classifier(stage):
    task_name = "glass"
    if stage == "train":
        loss_func = torch.nn.BCEWithLogitsLoss(reduce=False)
    else:
        loss_func = None
    config = dict(
        type="FacequalityMultiHead",
        in_channels=256,
        feat_channels=[128, 128],
        output_dim=1,
        task_name=task_name,
        loss=loss_func,
    )
    model = build_from_registry(config)
    feat = torch.rand(4, 256, 4, 4)
    data = {
        "img": torch.rand(4, 3, 128, 128),
        f"gt_{task_name}": torch.Tensor([1, 0, 0, 1]),
    }
    output = model(feat, data)
    pred = output["pred"]
    assert pred.shape == (4,)
    if loss_func is not None:
        loss = output["loss"]
        loss_expect = loss_func(pred, torch.Tensor([1, 0, 0, 1]))
        assert torch.sum(loss - loss_expect) == 0
