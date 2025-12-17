import torch

from hat.registry import build_from_registry


def test_loss_show():
    metric = build_from_registry(dict(type="LossShow"))
    for i in range(10):
        output = torch.tensor(i)
        metric.update(output)
    _, value = metric.get()
    assert value == [4.5]
