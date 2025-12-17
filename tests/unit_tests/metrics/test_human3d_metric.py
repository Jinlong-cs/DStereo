import pytest
import torch

from hat.metrics.human3d_metric import Human3dMetric

try:
    import smplx
except ImportError:
    smplx = None

labels = {
    "gt_smpl_pose": torch.randn((1, 72)),
    "gt_smpl_betas": torch.randn((1, 11)),
    "img": torch.randn((1, 3, 224, 224)),
}

preds = {
    "pred_pose": torch.randn((1, 144)),
    "pred_betas": torch.randn((1, 11)),
}


@pytest.mark.skipif(smplx is None, reason="need smplx")
@pytest.mark.parametrize(
    ["use_pa"],
    [
        [True],
        [False],
    ],
)
def test_human3d_metric(use_pa):
    metric = Human3dMetric(
        "test_metric", "test", "./tmp_orig_data/human3d/spin_params", use_pa
    )
    metric.update(labels, preds)
    _, value = metric.get()
    assert value > 0
