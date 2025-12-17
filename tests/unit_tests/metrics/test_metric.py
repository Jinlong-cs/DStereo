import logging

from hat.metrics.metric import EvalMetric


class ToyMetric1(EvalMetric):
    def __init__(self, name, warn_without_compute):
        super().__init__(name=name, warn_without_compute=warn_without_compute)

    def update(self, pred, label) -> None:
        self.num_inst += 1
        self.sum_metric += 1

    def get(self):
        return self.name, self.sum_metric / self.num_inst


class ToyMetric2(EvalMetric):
    def __init__(self, name):
        super().__init__(name=name)

    def update(self, pred, label) -> None:
        self.num_inst += 1
        self.sum_metric += 1


def test_eval_metric(caplog):
    """Test EvalMetric base class.

    Make sure users will get notification while using metrics without
    DDP support.
    """
    m1 = ToyMetric2("toy1")
    m2 = ToyMetric1("toy2", warn_without_compute=False)
    m3 = ToyMetric1("toy3", warn_without_compute=True)

    with caplog.at_level(logging.WARNING):
        m1.update(1, 2)
        m1.get()
    assert not caplog.text

    with caplog.at_level(logging.WARNING):
        m2.update(1, 2)
        m2.get()
    assert not caplog.text

    with caplog.at_level(logging.WARNING):
        m3.update(1, 2)
        m3.get()

    assert "not ready for distributed environment" in caplog.text
