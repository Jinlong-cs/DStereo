import torch

from hat.models.losses.focal_loss import GaussianFocalLoss
from hat.models.losses.l1_loss import L1Loss
from hat.models.task_modules.centerpoint.loss import CenterPointLoss


def test_centerpoint_loss():
    loss = CenterPointLoss(
        loss_cls=GaussianFocalLoss(loss_weight=1.0),
        loss_bbox=L1Loss(
            reduction="mean",
            loss_weight=0.25,
        ),
        with_velocity=True,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
    )

    preds = list()
    data = {}
    data["reg"] = torch.rand(1, 2, 32, 32)
    data["height"] = torch.rand(1, 1, 32, 32)
    data["dim"] = torch.rand(1, 3, 32, 32)
    data["vel"] = torch.rand(1, 2, 32, 32)
    data["rot"] = torch.rand(1, 2, 32, 32)
    data["heatmap"] = torch.rand(1, 1, 32, 32)
    preds.append(data)
    heatmaps = [torch.rand(1, 1, 32, 32)]
    anno_boxes = [torch.rand(1, 20, 10)]
    inds = [
        torch.randint(
            0,
            500,
            (
                1,
                20,
            ),
        )
    ]
    masks = [
        torch.randint(
            0,
            2,
            (
                1,
                20,
            ),
        )
    ]

    losses = loss(
        heatmaps,
        anno_boxes,
        inds,
        masks,
        preds,
    )
    assert "task0.loss_heatmap" in losses
    assert "task0.loss_bbox" in losses
