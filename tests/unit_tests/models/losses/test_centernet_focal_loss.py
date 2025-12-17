import pytest
import torch

from hat.models.task_modules.centernet import CenterNetFocalLoss
from tests.utils import gen_fake_torch_randn_data

BS, NC, FS = 3, 5, 32


@pytest.mark.parametrize(
    ["pred", "target"],
    [
        pytest.param(
            gen_fake_torch_randn_data((BS, NC, FS, FS)),
            gen_fake_torch_randn_data((BS, NC, FS, FS)),
        ),
    ],
)
def test_centernet_focal_loss(pred, target):
    loss_fn = CenterNetFocalLoss(
        loss_name="test_loss",
        alpha=2,
        gamma=4,
        loss_weight=1,
    )
    target = torch.clip(target, 0, 1)

    valid_classes_list = [[0, 1], [1, 2, 3], [3, 4]]
    loss = loss_fn(
        pred,
        target,
        valid_classes_list=valid_classes_list,
    )
    assert loss["test_loss"] > 0
