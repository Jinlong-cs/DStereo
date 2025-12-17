import pytest

from hat.callbacks.exception_monitor import (
    ExceptionMonitor,
    default_monitor_func,
)
from hat.callbacks.metric_updater import MetricUpdater
from hat.metrics.metric import EvalMetric
from tests.unit_tests.base import BoringTrainer
from tests.utils import init_test_logger

try:
    import aidisdk
except ImportError:
    aidisdk = None


class TestMetric(EvalMetric):
    def __init__(self, name, **kwargs):
        super(TestMetric, self).__init__(name=name, **kwargs)
        self.cnt = 0

    def update(self, model_outs):
        self.cnt += 1

    def get(self):
        return self.name, self.cnt


def update_metric(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


@pytest.mark.skipif(aidisdk is None, reason="need aidisdk")
def test_exception_monitor():
    init_test_logger(0)

    train_metrics = [TestMetric("loss_metric"), TestMetric("val_metric")]
    metric_updater = MetricUpdater(
        metric_update_func=update_metric,
        step_log_freq=1,
        epoch_log_freq=1,
        log_prefix="ToyTask",
        step_storage_freq=1,
        epoch_storage_freq=1,
        storage_key="monitor_obj",
    )

    exception_monitor = ExceptionMonitor(
        step_monitor_freq=1,
        epoch_monitor_freq=1,
        monitor_prefix="ToyTask",
        monitor_func=default_monitor_func,
        monitor_key="monitor_obj",
    )

    trainer = BoringTrainer(
        callbacks=[metric_updater, exception_monitor],
        train_metrics=train_metrics,
    )
    trainer.fit()
