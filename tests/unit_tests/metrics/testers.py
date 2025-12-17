import os

import pytest
import torch

from hat.engine.ddp_trainer import launch
from hat.utils.seed import seed_everything


class MetricTester:
    def setup_class(self):
        """Setup the metric class and inputs"""
        self.args = []

    @pytest.mark.serial_task
    def test_ddp(self):
        seed_everything(os.getpid())
        launch(
            self.func_test,
            device_ids=[0, 1],
            args=self.args,
        )

    def func_test(self):
        pass

    @staticmethod
    def check_states_values(metric):
        values = {}
        for attr in metric._defaults:
            val = getattr(metric, attr)
            if isinstance(val, torch.Tensor):
                val = val.clone()
            elif isinstance(val, list):
                val = [v.clone() for v in val]
            else:
                raise TypeError(type(val))

            values[attr] = val

        name, value = metric.get()

        for attr in metric._defaults:
            assert torch.all(values[attr] == getattr(metric, attr))

        return name, value
