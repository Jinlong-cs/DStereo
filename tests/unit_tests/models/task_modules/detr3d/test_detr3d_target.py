import torch

from hat.core.match_costs.match_cost import BBox3DL1Cost, FocalLossCost
from hat.models.task_modules.detr3d.target import Detr3dTarget


def test_detr3d_target():
    bev_range = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
    ref_p = torch.randn(2, 4, 256, 3)
    target = Detr3dTarget(
        cls_cost=FocalLossCost(
            alpha=0.25,
            gamma=2.0,
            weight=2.0,
        ),
        reg_cost=BBox3DL1Cost(
            weight=0.25,
        ),
        bev_range=bev_range,
    )

    labels = [torch.randint(1, 10, (10, 10)), torch.randint(1, 10, (10, 10))]
    cls_preds = torch.rand(2, 10, 4, 256)
    reg_preds = torch.rand(2, 10, 4, 256)
    results = target(labels, cls_preds, reg_preds, ref_p)

    for result in results:
        assert "target" in result
        assert "pred" in result
