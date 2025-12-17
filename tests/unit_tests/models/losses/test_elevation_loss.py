import torch

from hat.models.losses.elevation_loss import (
    ElevationLoss,
    GammaLoss,
    GroundLoss,
)
from tests.utils import gen_fake_torch_randint_data


def test_gamma_loss():
    torch.manual_seed(0)
    gamma_scale = 1000.0
    gamma_low = int(-0.06 * gamma_scale)
    gamma_high = int(0.31 * gamma_scale)
    target_gamma_loss = 122.9829
    target_ground_loss = 0.8193

    gamma_loss = GammaLoss(
        low=gamma_low,
        high=gamma_high,
        loss_weight=1.0,
    )
    ground_loss = GroundLoss(
        loss_weight=1.0,
        loss_type="cosin",
    )
    elevation_loss = ElevationLoss(
        gamma_loss=gamma_loss,
        ground_loss=ground_loss,
        pred_gammas_name="pred_gammas_frame0",
        gt_gamma_name="gt_gamma",
        pred_ground_name="pred_ground",
        gt_ground_name="ground_norm",
    )
    pred_gammas = gen_fake_torch_randint_data(
        (1, 1, 512, 960), low=gamma_low, high=gamma_high
    ).float()
    gt_gamma = gen_fake_torch_randint_data(
        (1, 1, 512, 960), low=gamma_low, high=gamma_high
    ).float()
    pred_ground = gen_fake_torch_randint_data((1, 3, 1, 1)).float()
    gt_ground = gen_fake_torch_randint_data((1, 3, 1)).float()

    pred = dict(
        pred_gammas_frame0=[pred_gammas],
        pred_ground=[pred_ground],
    )
    gt = dict(gt_gamma=[gt_gamma], ground_norm=[gt_ground])
    loss = elevation_loss(pred, gt)

    assert torch.abs(loss["gamma_loss"] - target_gamma_loss) < 1e-4
    assert torch.abs(loss["ground_loss"] - target_ground_loss) < 1e-4
