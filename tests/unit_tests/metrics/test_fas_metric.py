import torch

from hat.registry import build_from_registry


def test_fas_acc():
    fas_metric = build_from_registry(
        dict(type="FasSigmoidAccuracy", thr=0.5, name="fas_accuracy")
    )

    fas_label = torch.tensor([1, 0])
    car_cls = torch.tensor([0, 1])

    for _ in range(10):
        preds = [torch.rand(2, 1, 1, 1) + 1e-5 for i in range(3)]
        fas_metric.update(preds, fas_label, car_cls)
        name, val = fas_metric.get()
        assert abs(val - 0.5) < 1e-4


def test_tartrr():
    fas_metric = build_from_registry(
        dict(type="FasTARTRR", database_names=["a", "b"])
    )
    fas_label = torch.tensor([1, 0])
    car_cls = torch.tensor([0, 1])
    preds = [torch.tensor([-1, 100]), torch.tensor([-100, -10])]
    fas_metric.update(preds, fas_label, car_cls)
    names, vals = fas_metric.get()
    assert len(names) == len(vals) == 2
    for i in range(2):
        if names[i] == "a_acc":
            assert abs(vals[i] - 0) < 1e-4
        elif names[i] == "b_acc":
            assert abs(vals[1] - 100) < 1e-4
        else:
            assert 0
