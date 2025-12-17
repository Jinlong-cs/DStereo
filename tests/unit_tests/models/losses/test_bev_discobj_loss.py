import torch

from hat.models.losses.bev_loss import (
    ANCBEVDiscObjRelativeLoss,
    ANCBEVDiscreteObjectLoss,
    ANCBEVDiscreteObjectWithClsLoss,
)
from hat.models.losses.real3d_losses import sigmoid_and_clip


def test_bev_discrete_obj_loss():
    torch.manual_seed(0)
    loss_weights = {
        "bev_discobj_hm": 2.0,
        "bev_discobj_wh": 2.0,
        "bev_discobj_rot": 2.0,
    }
    target_hm_loss = 0.04321
    loss = ANCBEVDiscreteObjectLoss(
        loss_weights=loss_weights,
        use_focal_hm_loss=True,
    )
    loss_name = list(loss_weights.keys())

    pred = {
        "pred_bev_discobj_hm": 0.98 * torch.ones(1, 3, 256, 256),
        "pred_bev_discobj_wh": 0.88 * torch.ones(1, 3, 256, 256),
        "pred_bev_discobj_rot": 0.76 * torch.ones(1, 2, 256, 256),
    }
    target = {
        "bev_discrete_obj": {
            "bev_discobj_hm": 0.98 * torch.ones(1, 3, 256, 256),
            "bev_discobj_wh": 0.88 * torch.ones(1, 3, 256, 256),
            "bev_discobj_rot": 0.75 * torch.ones(1, 2, 256, 256),
            "bev_discobj_weight_hm": torch.rand(1, 1, 256, 256),
            "bev_discobj_ignore": torch.rand(1, 1, 256, 256),
        }
    }

    result = loss(pred, target)
    for i, _ in enumerate(loss_name):
        assert "loss_" + loss_name[i] in result
    assert torch.abs(result["loss_bev_discobj_hm"] - target_hm_loss) < 1e-4


def test_bev_discrete_obj_withclsloss():
    torch.manual_seed(0)
    loss_weights = {
        "bev_discobj_hm": 2.0,
        "bev_discobj_hm_cls": 2.0,
        "bev_discobj_hm_cls_aux": 2.0,
        "bev_discobj_wh": 2.0,
        "bev_discobj_rot": 2.0,
    }
    target_hm_loss = 0.04321
    target_cls_loss = 2.2467
    loss = ANCBEVDiscreteObjectWithClsLoss(
        loss_weights=loss_weights,
        use_focal_hm_loss=True,
        use_focal_cls_loss=False,
        use_softmax_focal_aux_loss=True,
    )
    loss_name = list(loss_weights.keys())

    pred = {
        "pred_bev_discobj_hm": 0.98 * torch.ones(1, 3, 256, 256),
        "pred_bev_discobj_hm_cls": 0.95 * torch.ones(1, 3, 256, 256),
        "pred_bev_discobj_wh": 0.88 * torch.ones(1, 3, 256, 256),
        "pred_bev_discobj_rot": 0.76 * torch.ones(1, 2, 256, 256),
    }
    target = {
        "bev_discrete_obj": {
            "bev_discobj_hm": 0.98 * torch.ones(1, 3, 256, 256),
            "bev_discobj_hm_cls": 0.95 * torch.ones(1, 3, 256, 256),
            "bev_discobj_wh": 0.88 * torch.ones(1, 3, 256, 256),
            "bev_discobj_rot": 0.75 * torch.ones(1, 2, 256, 256),
            "bev_discobj_weight_hm": torch.rand(1, 1, 256, 256),
            "bev_discobj_ignore": torch.rand(1, 1, 256, 256),
        }
    }

    result = loss(pred, target)
    for i, _ in enumerate(loss_name):
        assert "loss_" + loss_name[i] in result, loss_name[i]
    assert torch.abs(result["loss_bev_discobj_hm"] - target_hm_loss) < 1e-4
    assert (
        torch.abs(result["loss_bev_discobj_hm_cls"]) - target_cls_loss < 1e-4
    )


def test_bev_discrete_obj_relative_loss():
    torch.manual_seed(0)
    loss_weights = {
        "bev_discobj_rel_loc": 0.2,
        "bev_discobj_rel_rot": 0.5,
    }
    target_relative_loss = 5.8876
    loss = ANCBEVDiscObjRelativeLoss(
        loss_weights=loss_weights,
    )

    pred = {
        "pred_bev_discobj_hm": 0.98 * torch.ones(1, 3, 256, 256),
        "pred_bev_discobj_wh": 0.88 * torch.ones(1, 3, 256, 256),
        "pred_bev_discobj_rot": 0.76 * torch.ones(1, 2, 256, 256),
        "pred_bev_discobj_ct_offset": 0.76 * torch.ones(1, 2, 256, 256),
    }
    bev_discobj_instances = torch.zeros(1, 3, 256, 256)
    bev_discobj_instances[:, 0, 40:60, 60:80] = 1
    bev_discobj_instances[:, 0, 100:160, 160:180] = 3
    bev_discobj_instances[:, 1, 40:60, 60:80] = 2
    bev_discobj_instances[:, 1, 100:160, 160:180] = 4
    target = {
        "bev_discrete_obj": {
            "bev_discobj_hm": torch.ones(1, 3, 256, 256),
            "bev_discobj_wh": 0.8 * torch.ones(1, 3, 256, 256),
            "bev_discobj_rot": 0.7 * torch.ones(1, 2, 256, 256),
            "bev_discobj_weight_hm": torch.rand(1, 1, 256, 256),
            "bev_discobj_ct_offset": torch.ones(1, 2, 256, 256),
            "bev_discobj_instances": bev_discobj_instances,
        }
    }
    pred_hm = sigmoid_and_clip(pred["pred_bev_discobj_hm"])
    result = loss(pred, target, pred_hm)
    assert (
        torch.abs(result["loss_bev_discobj_group_rel"] - target_relative_loss)
        < 1e-4
    )
