import pytest
import torch

from hat.models.task_modules.detr import HungarianMatcher
from hat.utils.package_helper import check_packages_available


@pytest.mark.skipif(
    not check_packages_available("torchvision", raise_exception=False),
    reason="need torchvision",
)
@pytest.mark.parametrize(
    ["cost_class", "cost_bbox", "cost_giou"],
    [
        pytest.param(1, 5, 2),
    ],
)
def test_detr_matcher(cost_class, cost_bbox, cost_giou):
    targets = dict(
        gt_classes=[torch.tensor([16.0, 29.0]), torch.tensor([1.0, 0.0])],
        gt_bboxes=[torch.rand(2, 4), torch.rand(2, 4)],
    )
    outputs = {
        "pred_logits": torch.rand(2, 100, 81),
        "pred_boxes": torch.rand(2, 100, 4),
    }

    module = HungarianMatcher(
        cost_class=cost_class,
        cost_bbox=cost_bbox,
        cost_giou=cost_giou,
    )
    indices = module(
        outputs=outputs,
        data=targets,
    )
    assert len(indices) == len(targets["gt_bboxes"])
