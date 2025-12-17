import numpy.testing as npt
import torch

from hat.metrics.metric_3dv import RMSE, AbsRel, ConfRMSE
from tests.utils import gen_fake_torch_randint_data


def test_absrel():
    torch.manual_seed(0)
    target_value = [2.303, 22.5321]
    target_value_std = [0.0092, 0.1400]
    target = [target_value, target_value_std]
    metric = AbsRel(range_list=[[0, 150], [0, 10]], name="ABREL")
    batch_size = 4
    pred_depth = gen_fake_torch_randint_data(
        (batch_size, 1, 512, 960), low=0, high=150
    ).float()
    gt_depth = gen_fake_torch_randint_data(
        (batch_size, 1, 512, 960), low=0, high=150
    ).float()
    for i in range(batch_size):
        metric.update(gt_depth[i : i + 1], pred_depth[i : i + 1])
    _, result = metric.get()
    for i, val in enumerate(result):
        npt.assert_almost_equal(val, target[i], decimal=2)


def test_rmse():
    torch.manual_seed(0)
    target_value = [61.0565, 81.4261]
    target_value_std = [0.0413, 0.2421]
    target = [target_value, target_value_std]
    metric = RMSE(range_list=[[0, 150], [0, 10]], name="RMSE")
    batch_size = 4
    pred_depth = gen_fake_torch_randint_data(
        (batch_size, 1, 512, 960), low=0, high=150
    ).float()
    gt_depth = gen_fake_torch_randint_data(
        (batch_size, 1, 512, 960), low=0, high=150
    ).float()
    for i in range(batch_size):
        metric.update(gt_depth[i : i + 1], pred_depth[i : i + 1])
    _, result = metric.get()
    for i, val in enumerate(result):
        npt.assert_almost_equal(val, target[i], decimal=2)


def test_conf_rmse():
    torch.manual_seed(0)
    target_value = [1.5424, 1.9330]
    target_value_std = [0.0000, 0.001]
    target = [target_value, target_value_std]
    high = 150
    metric = ConfRMSE(range_list=[[0, high], [0, 10]], name="RMSEConf")
    batch_size = 4
    scale = 2
    pred_depth = gen_fake_torch_randint_data(
        (batch_size, 1, 512, 960), low=0, high=high
    ).float()
    gt_depth = gen_fake_torch_randint_data(
        (batch_size, 1, 512, 960), low=0, high=high
    ).float()
    gt_conf = (pred_depth - gt_depth).abs()
    gt_conf = torch.exp(-gt_conf / (gt_depth + 1e-6))
    pred_conf = gt_depth / high * scale
    for i in range(batch_size):
        metric.update(gt_depth[i], gt_conf[i], pred_conf)
    _, result = metric.get()
    for i, val in enumerate(result):
        npt.assert_almost_equal(val, target[i], decimal=2)
