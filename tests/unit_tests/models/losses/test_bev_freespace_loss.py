import torch

from hat.models.losses.bev_freespace_loss import ANCBEVFreespaceLoss
from hat.models.losses.cross_entropy_loss import CrossEntropyLoss


def get_fake_freespace_data(
    output_extra_obstacle=False,
    output_lidardet=False,
    output_bev_weight_map=False,
):
    pred = {
        "pred_bev_freespace_frame0": 0.8 * torch.ones(1, 2, 512, 512),
    }
    label = torch.ones(1, 512, 512).long()
    label[:, :, 100:300] = 0
    target = {
        "gt_bev_freespace": {
            "gt_bev_freespace": label,
        }
    }
    if output_extra_obstacle:
        target["gt_bev_freespace"].update(
            gt_bev_extra_obstacle=torch.randint(2, size=(1, 512, 512))
        )
    if output_lidardet:
        target["gt_bev_freespace"].update(
            gt_lidardet_veh_mask=torch.randint(2, size=(1, 512, 512))
        )
    if output_bev_weight_map:
        target["gt_bev_freespace"].update(
            bev_weight_map=torch.randint(2, size=(1, 512, 512))
        )
    return pred, target


def test_bevfreespace_loss():
    torch.manual_seed(0)
    use_sigmoid = True
    cls_loss_cfg = CrossEntropyLoss(
        reduction="mean",
        loss_weight=1.0,
        ignore_index=255,
        use_sigmoid=use_sigmoid,
    )
    cls_loss_name = "loss_bev_freespace"
    pred, target = get_fake_freespace_data()

    boundary_loss_cfg = {
        "use_boundary_loss": True,
        "boundary_winsize_half": 3,
        "boundary_loss_weight": 1.0,
    }
    bev_freespace_loss = ANCBEVFreespaceLoss(
        cls_loss_name=cls_loss_name,
        cls_loss_cfg=cls_loss_cfg,
        use_pixelwise_weight=True,
        pixelwise_weight_size=17,
        boundary_loss_cfg=boundary_loss_cfg,
    )
    loss = bev_freespace_loss(pred, target)

    assert cls_loss_name in loss
    assert torch.abs(loss[cls_loss_name] - 1.2584) < 1e-4
    if boundary_loss_cfg["use_boundary_loss"]:
        assert torch.abs(loss["loss_freespace_boundary"] - 0.7711) < 1e-4


def test_bevfreespace_reweight_loss():
    torch.manual_seed(0)
    use_sigmoid = True
    cls_loss_cfg = CrossEntropyLoss(
        reduction="mean",
        loss_weight=1.0,
        ignore_index=255,
        use_sigmoid=use_sigmoid,
    )
    cls_loss_name = "loss_bev_freespace"
    pred, target = get_fake_freespace_data(
        output_extra_obstacle=True,
        output_lidardet=True,
        output_bev_weight_map=True,
    )

    boundary_loss_cfg = {
        "use_boundary_loss": True,
        "boundary_winsize_half": 3,
        "boundary_loss_weight": 1.0,
    }
    bev_freespace_loss = ANCBEVFreespaceLoss(
        cls_loss_name=cls_loss_name,
        cls_loss_cfg=cls_loss_cfg,
        use_pixelwise_weight=True,
        pixelwise_weight_size=17,
        boundary_loss_cfg=boundary_loss_cfg,
    )
    loss = bev_freespace_loss(pred, target)

    assert cls_loss_name in loss
