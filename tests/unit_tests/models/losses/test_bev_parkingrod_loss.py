import torch

from hat.models.losses.bev_parkingrod_loss import ANCBEVParkingRodLoss
from hat.models.losses.focal_loss import FocalLossV2
from hat.models.losses.smooth_l1_loss import SmoothL1Loss
from hat.registry import build_from_registry


def test_bevparkingrod_loss():
    pred = (
        0.98 * torch.ones(1, 1, 96, 64),
        0.88 * torch.ones(1, 4, 96, 64),
    )
    target = {
        "gt_bev_parkingrod_obj": {
            "classification_obj": 1.00 * torch.ones(1, 1, 96, 64),
            "endpoint_offset_obj": 0.98 * torch.ones(1, 4, 96, 64),
            "classification_weight_mask": 1.00 * torch.ones(1, 1, 96, 64),
            "endpoint_offset_weight_mask": 0.98 * torch.ones(1, 4, 96, 64),
        }
    }
    torch.manual_seed(0)
    loss_weight_dict = {
        "cls_loss": 1.0,
        "endpoint_offset_loss": 1.0,
    }
    classification_loss = FocalLossV2(reduction="none")
    endpoint_offset_loss = SmoothL1Loss(beta=10.0, reduction="none")

    parkingrod_loss = ANCBEVParkingRodLoss(
        classification_loss=classification_loss,
        endpoint_offset_loss=endpoint_offset_loss,
        loss_weight_dict=loss_weight_dict,
    )
    parkingrod_loss = build_from_registry(parkingrod_loss)
    result = parkingrod_loss(preds=pred, targets=target)
    assert "classification_loss" in result.keys()
    assert "endpoint_offset_loss" in result.keys()
