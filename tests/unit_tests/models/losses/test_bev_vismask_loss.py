import torch

from hat.models.losses.bev_vismask_loss import ANCBevVismaskLoss
from hat.models.losses.cross_entropy_loss import CrossEntropyLoss


def get_fake_vismask_data():
    pred = {
        "pred_bev_vismask_frame0": 0.8 * torch.ones(1, 2, 512, 512),
    }
    target = {
        "gt_bev_elevation_vismask": {
            "vismask": torch.ones(1, 512, 512).int(),
            "agent": torch.ones(1, 512, 512).int(),
        }
    }
    return pred, target


def test_vismask_loss():
    torch.manual_seed(0)
    cls_loss_cfg = CrossEntropyLoss(
        reduction="mean",
        loss_weight=1.0,
        ignore_index=255,
    )
    cls_loss_name = "loss_bev_vismask"
    pred, target = get_fake_vismask_data()

    bev_vismask_loss = ANCBevVismaskLoss(
        vismask_loss_cfg=cls_loss_cfg,
    )
    loss = bev_vismask_loss(pred, target)

    assert cls_loss_name in loss
    assert torch.abs(loss[cls_loss_name] - 0.7) < 0.1
