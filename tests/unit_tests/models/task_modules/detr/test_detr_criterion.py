import pytest
import torch

from hat.models.task_modules.detr import DetrCriterion


@pytest.mark.parametrize(
    ["num_classes", "dec_layers", "aux_loss"],
    [
        pytest.param(80, 6, True),
        pytest.param(80, 6, False),
    ],
)
def test_detr_criterion(num_classes, dec_layers, aux_loss):
    losses = ("labels", "boxes", "cardinality")
    targets = dict(
        gt_classes=[torch.tensor([16.0, 29.0]), torch.tensor([1.0, 0.0])],
        gt_bboxes=[
            torch.tensor([[10.0, 10.0, 30.0, 35.0], [10.0, 10.0, 30.0, 35.0]]),
            torch.tensor([[10.0, 10.0, 30.0, 35.0], [10.0, 10.0, 30.0, 35.0]]),
        ],
        img_shape=[[3, 420, 620], [3, 420, 620]],
    )
    outputs_class = torch.rand(6, 2, 3, 81)
    outputs_box = torch.rand(6, 2, 3, 4)

    module = DetrCriterion(
        num_classes=num_classes,
        dec_layers=dec_layers,
        aux_loss=aux_loss,
        losses=losses,
    )
    results = module(
        outs=(outputs_class, outputs_box),
        targets=targets,
    )
    assert len(results.keys()) >= len(losses) + 2
