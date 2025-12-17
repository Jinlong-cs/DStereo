from distutils.version import LooseVersion

import horizon_plugin_pytorch as horizon
import pytest

from hat.callbacks.pruner import Pruner
from tests.unit_tests.base import BoringTrainer

skip_test = not LooseVersion(horizon.__version__) >= LooseVersion("1.10.3")


@pytest.mark.skipif(skip_test, reason="need to update horizon-plugin-pytorch")
class TestPruner(object):
    def test_unstructed_pruner(self):
        trainer = BoringTrainer()

        def eval_func():
            for data in trainer.data_loader:
                output = trainer.model(data)["pred_bboxes"].sum()
            return output

        pruner = Pruner("unstructed", eval_func=eval_func)
        pruner.on_loop_begin(trainer.model, trainer.optimizer)

    def test_semistructed_pruner(self):
        trainer = BoringTrainer()

        def eval_func():
            for data in trainer.data_loader:
                output = trainer.model(data)["pred_bboxes"].sum()
            return output

        pruner = Pruner("semistructed", eval_func=eval_func)
        pruner.on_loop_begin(trainer.model, trainer.optimizer)
        pruner.on_epoch_begin()
