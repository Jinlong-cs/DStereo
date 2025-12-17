import pytest
import torch

from hat.models.losses.landmark import HumanPoseLoss, LdmkLoss
from hat.models.losses.smooth_l1_loss import SmoothL1Loss


@pytest.mark.parametrize(["loss_type"], [["l1"], ["l2"]])
def test_ldmk_loss(loss_type):
    label = torch.randn((8, 68, 32, 32))
    pred = torch.randn(label.shape)
    weight = torch.ones_like(label)
    loss_func = LdmkLoss(loss_type)
    loss = loss_func(label, pred, weight)
    assert loss > 0


def test_human_pose_loss_module():
    loss = HumanPoseLoss(15, 32, 32, SmoothL1Loss(), SmoothL1Loss())
    preds = {
        "ldmk_pred": torch.randn((8, 45, 32, 32)),
    }
    labels = {
        "ldmk_cls_label": torch.randn((8, 15, 32, 32)),
        "ldmk_cls_label_weight": torch.randn((8, 15, 32, 32)),
        "ldmk_reg_label": torch.randn((8, 30, 32, 32)),
        "ldmk_reg_label_weight": torch.randn((8, 30, 32, 32)),
    }
    r = loss(preds, labels)
    assert len(r) == 2
    assert "ldmk_cls_loss" in r
    assert "ldmk_reg_loss" in r
