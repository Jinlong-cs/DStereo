import pytest

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.metrics.metric import EvalMetric
from hat.registry import OBJECT_REGISTRY, build_from_registry
from tests.unit_tests.base import BoringTrainer
from tests.utils import init_test_logger


@OBJECT_REGISTRY.register
class ToyMetric(EvalMetric):
    def __init__(self, name, **kwargs):
        super(ToyMetric, self).__init__(name=name, **kwargs)
        self.cnt = 0

    def update(self, labels, preds):
        self.cnt += 1

    def get(self):
        return self.name, self.cnt


@OBJECT_REGISTRY.register
class ToyLossMetric(EvalMetric):
    def __init__(self, name, **kwargs):
        super(ToyLossMetric, self).__init__(name=name, **kwargs)
        self.cnt = 0

    def update(self, preds):
        self.cnt += 1

    def get(self):
        return self.name, self.cnt


def test_update_metric_using_regex():
    init_test_logger(0)
    metrics = [
        dict(type="ToyLossMetric", name="loss_metric"),
        dict(type="ToyMetric", name="pred_metric"),
    ]
    train_metrics = build_from_registry(metrics)

    metric_updater = dict(
        type="MetricUpdater",
        metric_update_func=update_metric_using_regex(
            per_metric_patterns=[  # corresponding to metrics
                dict(label_pattern=None, pred_pattern="^.*loss.*"),
                dict(label_pattern="^.*label.*", pred_pattern="^.*predict.*"),
            ]
        ),
        step_log_freq=1,
        epoch_log_freq=1,
        log_prefix="ToyTask",
    )
    metric_updater = build_from_registry(metric_updater)

    trainer = BoringTrainer(
        callbacks=metric_updater, train_metrics=train_metrics
    )
    trainer.fit()


# TODO(linkai.liang, ?): test case #
def test_update_metric_using_index():
    pass


if __name__ == "__main__":
    pytest.main(["-s", __file__])
