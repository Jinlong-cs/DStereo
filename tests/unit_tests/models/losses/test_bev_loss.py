import torch

from hat.models.losses.bev_loss import ANCBEV3DLoss, ANCBEVSegLoss
from hat.models.losses.cross_entropy_loss import CrossEntropyLoss


def get_fake_bev3d_data():
    pred = {
        "bev3d_hm": 0.98 * torch.ones(1, 3, 256, 256),
        "bev3d_dim": 0.88 * torch.ones(1, 3, 256, 256),
        "bev3d_rot": 0.76 * torch.ones(1, 3, 256, 256),
        "bev3d_ct_offset": 0.5 * torch.ones(1, 2, 256, 256),
        "bev3d_loc_z": 0.05 * torch.ones(1, 1, 256, 256),
    }
    target = {
        "gt_bev_3d": {
            "bev3d_hm": 0.98 * torch.ones(1, 3, 256, 256),
            "bev3d_dim": 0.88 * torch.ones(1, 3, 256, 256),
            "bev3d_rot": 0.75 * torch.ones(1, 3, 256, 256),
            "bev3d_ct_offset": 0.5 * torch.ones(1, 2, 256, 256),
            "bev3d_loc_z": 0.05 * torch.ones(1, 1, 256, 256),
            "bev3d_weight_hm": torch.rand(1, 1, 256, 256),
            "bev3d_point_pos_mask": torch.ones(1, 1, 256, 256),
            "bev3d_background_weight": torch.ones(1, 1, 256, 256),
            "bev3d_roi_weight": torch.ones(1, 1, 256, 256),
        }
    }
    return pred, target


def test_bev3d_loss():
    torch.manual_seed(0)
    loss_weights = {
        "bev3d_hm": 2.0,
        "bev3d_dim": 2.0,
        "bev3d_rot": 2.0,
        "bev3d_ct_offset": 2.0,
        "bev3d_loc_z": 2.0,
    }
    target_bev3d_hm_loss = 0.04321
    loss = ANCBEV3DLoss(loss_weights=loss_weights)
    loss_name = list(loss_weights.keys())
    pred, target = get_fake_bev3d_data()
    result = loss(pred, target)
    assert loss_name[0] + "_loss" in result
    assert loss_name[1] + "_loss" in result
    assert loss_name[2] + "_loss" in result
    assert loss_name[3] + "_loss" in result

    assert torch.abs(result["bev3d_hm_loss"] - target_bev3d_hm_loss) < 1e-4

    loss.use_rot_wing_loss = True
    target_bev3d_rot_wing_loss = 1.2164
    result = loss(pred, target)
    assert (
        torch.abs(result["bev3d_rot_loss"] - target_bev3d_rot_wing_loss) < 1e-4
    )


def test_bev_seg_loss():
    torch.manual_seed(0)
    pred_names = [
        "pred_bev_segs_frame0",
        "pred_bev_conf_frame0",
        "pred_bev_occlusion_frame0",
    ]
    loss_names = ["loss_bev_seg", "loss_bev_conf", "loss_bev_occlusion"]
    conf_loss_weight = 1.0
    num_classes = 6
    target_bev_seg_loss = 1.79175
    target_bev_conf_loss = 0.71333
    target_bev_occlusion_loss = 0.6931

    pred = {
        "pred_bev_segs_frame0": [0.98 * torch.ones(1, num_classes, 256, 256)],
        "pred_bev_conf_frame0": [0.88 * torch.ones(1, 1, 256, 256)],
        "pred_bev_occlusion_frame0": [0.98 * torch.ones(1, 2, 256, 256)],
    }
    target = {
        "gt_bev_seg": torch.ones(1, 256, 256),
        "occlusion": torch.ones(1, 256, 256),
    }

    seg_loss_criterion = CrossEntropyLoss(
        reduction="mean",
    )
    occlusion_loss_criterion = CrossEntropyLoss(
        reduction="mean",
    )

    bev_seg_loss = ANCBEVSegLoss(
        pred_names=pred_names,
        loss_names=loss_names,
        loss_seg_cfg=seg_loss_criterion,
        loss_occlusion_cfg=occlusion_loss_criterion,
        conf_loss_weight=conf_loss_weight,
    )

    result = bev_seg_loss(pred, target)

    for loss_name in loss_names:
        assert loss_name in result

    assert torch.abs(result["loss_bev_seg"] - target_bev_seg_loss) < 1e-4
    assert torch.abs(result["loss_bev_conf"] - target_bev_conf_loss) < 1e-4
    assert (
        torch.abs(result["loss_bev_occlusion"] - target_bev_occlusion_loss)
        < 1e-4
    )
