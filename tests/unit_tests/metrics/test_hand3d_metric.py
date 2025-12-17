import pytest
import torch

from hat.metrics.hand3d_metric import Hand3dJPE, Hand3dNME

torch.manual_seed(0)
batch_size = 16
label = {
    "gt_ldmk": torch.rand(batch_size, 21, 2),
    "ldmk3d_relat": torch.rand(batch_size, 21, 3) * 0.25,
    "gt_ldmk3d": torch.rand(batch_size, 21, 3) * 2.5,
    "ldmk3d_vis": torch.randint(0, 1, [batch_size, 1]),
}

data = {
    "pred_ldmk3d_relat": torch.rand(batch_size, 21, 3) * 0.25,
    "pred_ldmk3d": torch.rand(batch_size, 21, 3) * 2.5,
    "pred_ldmk": torch.rand(batch_size, 21, 2),
    "pred_ldmk_proj": torch.rand(batch_size, 21, 2),
}


@pytest.mark.parametrize(
    ["label", "data", "mode"],
    [
        pytest.param(label, data, "reproj"),
        pytest.param(label, data, "coords"),
    ],
)
def test_Hand3dNME(label, data, mode):

    hand3d_nme = Hand3dNME(mode=mode, name="hand3d_nme")
    hand3d_nme.update(label, data)

    num_inst = hand3d_nme.num_inst
    sum_metric = hand3d_nme.sum_metric
    assert sum_metric / (num_inst + 1) is not None


@pytest.mark.parametrize(
    ["label", "data", "mode"],
    [
        pytest.param(label, data, "root"),
        pytest.param(label, data, "cam"),
    ],
)
def test_Hand3dJPE(label, data, mode):

    hand3d_jpe = Hand3dJPE(mode=mode, name="hand3d_jpe")
    hand3d_jpe.update(label, data)

    num_inst = hand3d_jpe.num_inst
    sum_metric = hand3d_jpe.sum_metric
    assert sum_metric / (num_inst + 1) is not None
