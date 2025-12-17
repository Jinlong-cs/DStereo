import torch

from hat.registry import build_from_registry


def test_ganet_decoder():
    kpts_hm = torch.randn((1, 1, 40, 100))
    pts_offset = torch.randn((1, 2, 40, 100))
    int_offset = torch.randn((1, 2, 40, 100))

    loss_config = dict(
        type="GaNetLoss",
        loss_kpts_cls=dict(
            type="LaneFastFocalLoss",
            loss_weight=1.0,
        ),
        loss_pts_offset_reg=dict(
            type="L1Loss",
            loss_weight=0.5,
        ),
        loss_int_offset_reg=dict(
            type="L1Loss",
            loss_weight=1.0,
        ),
    )
    loss_module = build_from_registry(loss_config)

    target = dict(
        gt_kpts_hm=torch.rand(1, 1, 40, 100),
        pts_offset_mask=torch.rand(1, 2, 40, 100),
        pts_offset=torch.rand(1, 2, 40, 100),
        int_offset_mask=torch.rand(1, 1, 40, 100),
        int_offset=torch.rand(1, 2, 40, 100),
    )

    out_loss = loss_module(kpts_hm, pts_offset, int_offset, target)

    assert "kpts_cls_loss" in out_loss
    assert "offset_reg_loss" in out_loss
    assert "int_offset_reg_loss" in out_loss
