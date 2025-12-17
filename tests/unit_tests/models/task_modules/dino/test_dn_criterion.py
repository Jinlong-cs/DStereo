import pytest
import torch

from hat.models.task_modules.detr.matcher import HungarianMatcher
from hat.models.task_modules.dino import DINOCriterion


@pytest.mark.parametrize(
    [
        "num_classes",
    ],
    [
        pytest.param(
            80,
        ),
    ],
)
def test_dn_criterion(num_classes):
    targets = dict(
        gt_classes=[torch.tensor([6, 29]).long(), torch.tensor([1, 0]).long()],
        gt_bboxes=[
            torch.tensor([[10.0, 10.0, 30.0, 35.0], [10.0, 10.0, 30.0, 35.0]]),
            torch.tensor([[10.0, 10.0, 30.0, 35.0], [10.0, 10.0, 30.0, 35.0]]),
        ],
        img_shape=[[3, 420, 620], [3, 420, 620]],
    )
    outputs_class = torch.rand(2, 3, 81)
    outputs_box = torch.rand(2, 3, 4)
    outputs = {
        "pred_logits": outputs_class,
        "pred_boxes": outputs_box,
        "aux_outputs": [
            {
                "pred_logits": torch.rand(2, 3, 81),
                "pred_boxes": torch.rand(2, 3, 4),
            }
            for i in range(2)
        ],
    }
    module = DINOCriterion(
        num_classes=num_classes,
        matcher=HungarianMatcher(
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            use_focal=True,
            alpha=0.25,
            gamma=2.0,
        ),
        weight_dict={
            "loss_class": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
        },
        loss_class_type="focal_loss",
        alpha=0.25,
        gamma=2.0,
    )
    results = module(
        outputs=outputs,
        targets=targets,
    )
    assert len(results.keys()) == 18
