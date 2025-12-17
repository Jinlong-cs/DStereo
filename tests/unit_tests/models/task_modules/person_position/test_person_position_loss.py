import torch

from hat.models.losses.cross_entropy_loss import CrossEntropyLoss
from hat.models.task_modules.person_position.person_position_loss import (
    PersonPositionLoss,
)


def test_person_position_loss():
    pred = {
        "pred_oms": torch.randn((32, 4, 1, 1)),
        "pred_dms": torch.randn((32, 4, 1, 1)),
    }
    gt = {
        "dms_cls_label": torch.randint(low=0, high=3, size=(2, 16)),
        "oms_cls_label": torch.randint(low=0, high=3, size=(2, 16)),
        "dms_cls_label_weight": torch.randn((2, 16)),
        "oms_cls_label_weight": torch.randn((2, 16)),
    }
    loss = PersonPositionLoss(
        oms_loss=CrossEntropyLoss(reduction="sum"),
        dms_loss=CrossEntropyLoss(reduction="sum"),
    )
    output = loss(pred, gt)

    assert "oms_acc" in output
    assert "dms_acc" in output
    assert "oms_loss" in output
    assert "dms_loss" in output
