import pytest
import torch

from hat.models.task_modules.hand3d.losses.loss_structure import (
    H3DLossStucture,
)


def test_hand3d_bmc_loss_module():
    loss = H3DLossStucture(
        loss_class_init_list=[],
        loss_forward_list=[
            dict(
                loss_name="l_mano_ldmk3d",
                forward_func="smoothl1",
                weight=1,
                forward_params=dict(
                    pred="output_decoder['pred_ldmk3d']",
                    target="data['gt_ldmk3d']",
                    normlier="data['ldmk3d_vis']",
                    sigma=2.5,
                ),
            ),
            dict(
                loss_name="l_ldmk2d_gt",
                forward_func="smoothl1",
                weight=1,
                forward_params=dict(
                    pred="output_decoder['pred_ldmk2d']",
                    target="data['gt_ldmk']",
                    normlier="data['ldmk_vis']",
                    sigma=2.5,
                ),
            ),
        ],
    )
    batch_size = 16
    output_decoder = {
        "pred_ldmk2d": torch.rand((batch_size, 21, 2)),
        "pred_ldmk3d": torch.rand((batch_size, 21, 3)),
    }
    data = {
        "gt_ldmk": torch.rand((batch_size, 21, 2)),
        "gt_ldmk3d": torch.rand((batch_size, 21, 3)),
        "ldmk_vis": torch.ones((batch_size, 21)),
        "ldmk3d_vis": torch.ones((batch_size,)),
    }

    losses = loss(output_decoder, data)

    assert "l_mano_ldmk3d" in losses and losses["l_mano_ldmk3d"] >= 0
    assert "l_ldmk2d_gt" in losses and losses["l_ldmk2d_gt"] >= 0


if __name__ == "__main__":
    pytest.main(["-s", __file__])
