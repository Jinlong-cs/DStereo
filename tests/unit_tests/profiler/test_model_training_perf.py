from hat.profiler.model_training_perf import ModelTrainingPerf
from tests.unit_tests.base import BoringTrainer


def test_model_training_perf():
    trainer = BoringTrainer()
    model_perf = ModelTrainingPerf(trainer, iter_nums=100, frequent=10)
    model_perf.run()
