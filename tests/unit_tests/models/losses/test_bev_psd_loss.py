import torch

from hat.models.losses.bev_psd_loss import (
    ANCBEVPSDGlobalLoss,
    ANCBEVPSDLocalLoss,
    ANCBEVPSDLoss,
)
from hat.models.losses.focal_loss import FocalLossV2, GaussianFocalLoss
from hat.models.losses.smooth_l1_loss import SmoothL1Loss
from hat.registry import build_from_registry


def get_fake_bevpsd_data():
    pred = (
        0.98 * torch.ones(1, 1, 48, 32),
        0.88 * torch.ones(1, 8, 48, 32),
        0.76 * torch.ones(1, 1, 48, 32),
        0.50 * torch.ones(1, 3, 48, 32),
        0.05 * torch.ones(1, 2, 48, 32),
        0.98 * torch.ones(1, 4, 192, 128),
        0.88 * torch.ones(1, 8, 192, 128),
        0.76 * torch.ones(1, 8, 192, 128),
        0.55 * torch.ones(1, 4, 192, 128),
    )
    target = {
        "gt_bev_psd_obj": {
            "global_classification_obj": 1.00 * torch.ones(1, 1, 48, 32),
            "global_offset_obj": 0.98 * torch.ones(1, 8, 48, 32),
            "global_occupancy_obj": 0.98 * torch.ones(1, 1, 48, 32),
            "global_slot_type_obj": 0.98 * torch.ones(1, 3, 48, 32),
            "global_direction_obj": 0.98 * torch.ones(1, 2, 48, 32),
            "global_classification_grad": 1.00 * torch.ones(1, 1, 48, 32),
            "global_offset_grad": 0.98 * torch.ones(1, 8, 48, 32),
            "global_occupancy_grad": 0.98 * torch.ones(1, 1, 48, 32),
            "global_slot_type_grad": 0.98 * torch.ones(1, 3, 48, 32),
            "global_direction_grad": 0.98 * torch.ones(1, 2, 48, 32),
            "local_classification_obj": 1.00 * torch.ones(1, 4, 192, 128),
            "local_offset_obj": 0.98 * torch.ones(1, 8, 192, 128),
            "local_sline_angle_obj": 0.98 * torch.ones(1, 8, 192, 128),
            "local_point_type_obj": 0.98 * torch.ones(1, 4, 192, 128),
            "local_classification_grad": 1.00 * torch.ones(1, 4, 192, 128),
            "local_offset_grad": 0.98 * torch.ones(1, 8, 192, 128),
            "local_sline_angle_grad": 0.98 * torch.ones(1, 8, 192, 128),
            "local_point_type_grad": 0.98 * torch.ones(1, 4, 192, 128),
        }
    }
    return pred, target


def test_bevpsd_loss():
    torch.manual_seed(0)
    local_loss_weights = [1, 1, 1, 1]
    local_classification_loss = GaussianFocalLoss(alpha=2.0, gamma=4.0)
    local_offset_loss = SmoothL1Loss(reduction="none", beta=1.0)
    local_sline_angle_loss = torch.nn.MSELoss(reduction="none")
    local_point_type_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
    local_offset_weight = [4.0, 1.0]
    local_loss = ANCBEVPSDLocalLoss(
        classification_loss=local_classification_loss,
        offset_loss=local_offset_loss,
        sline_angle_loss=local_sline_angle_loss,
        point_type_loss=local_point_type_loss,
        loss_weights=local_loss_weights,
        offset_weight=local_offset_weight,
    )

    global_loss_weights = [1, 1, 1, 1, 1]
    global_classification_loss = FocalLossV2(reduction="none")
    global_offset_loss = SmoothL1Loss(beta=10.0, reduction="none")
    global_occupancy_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
    global_slot_type_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
    global_direction_loss = torch.nn.MSELoss(reduction="none")
    global_loss = ANCBEVPSDGlobalLoss(
        classification_loss=global_classification_loss,
        offset_loss=global_offset_loss,
        occupancy_loss=global_occupancy_loss,
        slot_type_loss=global_slot_type_loss,
        direction_loss=global_direction_loss,
        loss_weights=global_loss_weights,
    )

    bevpsd_loss = ANCBEVPSDLoss(local_loss=local_loss, global_loss=global_loss)
    bevpsd_loss = build_from_registry(bevpsd_loss)
    pred, target = get_fake_bevpsd_data()
    result = bevpsd_loss(preds=pred, targets=target)
    assert "global_classification_loss" in result.keys()
    assert "global_offset_loss" in result.keys()
    assert "global_occupancy_loss" in result.keys()
    assert "global_slot_type_loss" in result.keys()
    assert "global_direction_loss" in result.keys()
    assert "local_classification_loss" in result.keys()
    assert "local_offset_loss" in result.keys()
    assert "local_sline_angle_loss" in result.keys()
    assert "local_point_type_loss" in result.keys()
