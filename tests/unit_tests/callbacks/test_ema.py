import pytest

from hat.callbacks.exponential_moving_average import ExponentialMovingAverage
from tests.unit_tests.base import BoringTraceModel, BoringTrainer


@pytest.fixture
def boring_model():
    model = BoringTraceModel()
    return model


@pytest.fixture
def boring_Trainer():
    trainer = BoringTrainer()
    return trainer


@pytest.fixture
def ema():
    return ExponentialMovingAverage(decay=0.9999)


class TestEMA(object):
    def test_on_loop_begin(
        self,
        ema,
        boring_Trainer,
        boring_model,
    ):
        ema.on_loop_begin(loop=boring_Trainer, model=boring_model)

    def test_on_step_end(self, ema, boring_Trainer, boring_model):
        ema.on_step_end(
            0,
            model=boring_Trainer.model,
            infer_model=boring_Trainer.ema_model,
        )
