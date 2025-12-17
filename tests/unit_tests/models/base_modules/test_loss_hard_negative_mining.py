import numpy as np
import pytest
import torch

from hat.models.base_modules.loss_hard_neg_mining import LossHardNegativeMining


@pytest.mark.parametrize(
    ["keep_pos", "neg_ratio", "hard_ratio", "per_channel"],
    [
        pytest.param(True, 0.5, 0.5, True),
        pytest.param(False, 0.3, 0.4, True),
        pytest.param(True, 0.5, 0.5, False),
        pytest.param(False, 0.3, 0.4, False),
    ],
)
def test_LossHardNegativeMining(keep_pos, neg_ratio, hard_ratio, per_channel):

    lhnm = LossHardNegativeMining(
        keep_pos=keep_pos,
        neg_ratio=neg_ratio,
        hard_ratio=hard_ratio,
        per_channel=per_channel,
    )

    # generate fake data
    data_shape = np.random.randint(5, 10, size=4)
    fake_data = torch.as_tensor(
        np.random.random(size=data_shape),
        dtype=torch.float,
    )
    fake_tm = torch.as_tensor(np.random.randint(-1, 2, size=data_shape))

    fake_data_before = fake_data.clone()
    # get rid of randomness
    torch.manual_seed(0)
    loss_mask = lhnm(fake_data, fake_tm)

    # make sure loss value remains the same after lhnm
    assert torch.all(fake_data_before == fake_data)

    # 0. check if all igs are actually ignored
    assert torch.all(loss_mask[fake_tm == lhnm.IGNORE] == 0)

    # 1. check the keep_pos behavior
    if keep_pos:
        assert torch.all(loss_mask[fake_tm == lhnm.POSITIVE] == 1)

        if per_channel:
            for i in range(loss_mask.shape[1]):
                fake_tm_channel = fake_tm[:, i]
                loss_mask_channel = loss_mask[:, i]
                _neg_ratio = (
                    fake_tm_channel[loss_mask_channel] == lhnm.NEGATIVE
                ).sum() / loss_mask_channel.sum()
                _label_neg_ratio = (fake_tm[:, i] == lhnm.NEGATIVE).sum() / (
                    fake_tm[:, i] != lhnm.IGNORE
                ).sum()
                assert (abs(_neg_ratio - neg_ratio) <= 0.02) or (
                    abs(_neg_ratio - _label_neg_ratio) <= 0.02
                )
        else:
            _neg_ratio = (
                fake_tm[loss_mask] == lhnm.NEGATIVE
            ).sum() / loss_mask.sum()
            _label_neg_ratio = (fake_tm == lhnm.NEGATIVE).sum() / (
                fake_tm != lhnm.IGNORE
            ).sum()
            assert (abs(_neg_ratio - neg_ratio) <= 0.01) or (
                abs(_neg_ratio - _label_neg_ratio) <= 0.01
            )

    # 2. check cpu/gpu consistency
    fake_data_d = fake_data.to(torch.device(0))
    fake_tm_d = fake_tm.to(torch.device(0))

    torch.manual_seed(0)
    loss_mask_2 = lhnm(fake_data_d, fake_tm_d).cpu()

    assert torch.all(loss_mask == loss_mask_2)
