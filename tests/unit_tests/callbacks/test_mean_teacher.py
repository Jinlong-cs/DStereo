import pytest

from hat.callbacks.mean_teacher import MeanTeacher
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
def mean_teacher():
    return MeanTeacher(decay=0.9999)


class TestMeanTeacher(object):
    def test_on_loop_begin(
        self,
        mean_teacher,
        boring_Trainer,
        boring_model,
    ):
        mean_teacher.on_loop_begin(loop=boring_Trainer, model=boring_model)

    def test_on_step_end(self, mean_teacher, boring_Trainer, boring_model):
        mean_teacher.on_step_end(
            0,
            model=boring_Trainer.model,
            infer_model=boring_Trainer.ema_model,
        )
