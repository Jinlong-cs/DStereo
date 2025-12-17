import torch

from hat.metrics.metric_elevation import ElevationMetric
from tests.utils import gen_fake_torch_randn_data


def test_metric_elevation():
    torch.manual_seed(0)
    gamma_scale = 1000.0
    metric = ElevationMetric(
        metrics=["depth"],
        gamma_scale=gamma_scale,
    )
    pred_gamma = gen_fake_torch_randn_data((1, 1, 512, 960)).float()
    gt_gamma = gen_fake_torch_randn_data((1, 1, 512, 960)).float()
    pred_depth = gen_fake_torch_randn_data((1, 1, 512, 960)).float()
    gt_depth = gen_fake_torch_randn_data((1, 1, 512, 960)).float()
    pred_heigth = gen_fake_torch_randn_data((1, 1, 512, 960)).float()
    gt_height = gen_fake_torch_randn_data((1, 1, 512, 960)).float()
    timestamp = torch.tensor(123456789).float()

    metric.update(
        gamma_gt=gt_gamma,
        depth_gt=gt_depth,
        height_gt=gt_height,
        gamma_pred=pred_gamma,
        depth_pred=pred_depth,
        height_pred=pred_heigth,
        timestamp=timestamp,
    )
    _, result = metric.get()
    assert result is not None
