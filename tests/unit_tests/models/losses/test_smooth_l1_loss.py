import pytest
import torch

from hat.models.losses.smooth_l1_loss import SmoothL1Loss


@pytest.mark.parametrize(
    ["loss_weight", "hard_neg_mining_cfg"],
    [
        pytest.param(
            0.5,
            dict(  # noqa B006
                keep_pos=True,
                neg_ratio=0.5,
                hard_ratio=0.5,
            ),
        ),
        pytest.param(None, None),
    ],
)
def test_smooth_l1_loss(loss_weight, hard_neg_mining_cfg):
    pred = torch.Tensor(
        [
            [
                [0.9140, 0.3939, 0.8436, 0.8709, 0.3387],
                [0.7222, 0.1991, 0.7227, 0.0187, 0.8135],
                [0.5644, 0.7513, 0.0774, 0.3974, 0.2366],
            ],
            [
                [0.8459, 0.2419, 0.1675, 0.3813, 0.7049],
                [0.2227, 0.8864, 0.8276, 0.9101, 0.4410],
                [0.6678, 0.2443, 0.5320, 0.6253, 0.0654],
            ],
        ]
    )
    target = torch.Tensor(
        [
            [
                [0.4135, 0.6855, 0.8427, 0.3257, 0.9094],
                [0.1176, 0.4234, 0.7032, 0.2256, 0.4893],
                [0.4502, 0.8952, 0.6433, 0.0331, 0.3308],
            ],
            [
                [0.9196, 0.7138, 0.6535, 0.2618, 0.0073],
                [0.3535, 0.3894, 0.0118, 0.0079, 0.0777],
                [0.0233, 0.7960, 0.5167, 0.4476, 0.1777],
            ],
        ]
    )
    weight = torch.Tensor(
        [
            [
                [True, True, True, True, True],
                [True, True, True, True, True],
                [False, False, False, False, False],
            ],
            [
                [True, True, True, True, True],
                [False, False, False, False, False],
                [True, True, True, True, True],
            ],
        ]
    )

    if hard_neg_mining_cfg:
        # (N, 1, H, W)
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)
        weight = weight.unsqueeze(1)
        assert pred.shape[1] == target.shape[1] == weight.shape[1] == 1
        target_loss = torch.FloatTensor([2.8337]).squeeze()
    else:
        target_loss = torch.FloatTensor([0.1889]).squeeze()

    if loss_weight:
        target_loss *= loss_weight

    smooth_l1_loss = SmoothL1Loss(
        beta=1 / 9.0,
        reduction="mean",
        loss_weight=loss_weight,
        hard_neg_mining_cfg=hard_neg_mining_cfg,
    )
    loss = smooth_l1_loss(pred, target, weight)

    assert torch.abs(target_loss - loss) < 1e-4
