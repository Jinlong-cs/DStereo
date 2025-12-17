import torch

from hat.models.losses.hand_bmcloss import LP_PATH, H3DBMCLoss


def test_hand3d_bmc_loss_module():
    bmc_loss = H3DBMCLoss(
        lambda_bl=1,
        lambda_rb=1,
        lambda_a=1,
        path_bmc_lp=LP_PATH,
    )
    preds = {
        "pred_ldmk3d": torch.rand((16, 21, 3)),
    }

    _, bmc_loss_v = bmc_loss.compute_loss(preds["pred_ldmk3d"])

    assert "l_bmc_bl" in bmc_loss_v and bmc_loss_v["l_bmc_bl"] >= 0
    assert "l_bmc_rb" in bmc_loss_v and bmc_loss_v["l_bmc_rb"] >= 0
    assert "l_bmc_a" in bmc_loss_v and bmc_loss_v["l_bmc_a"] >= 0
