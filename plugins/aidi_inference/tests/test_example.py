import os

import pytest
import torch
from base import BaseTestModel


class TestModelExample(BaseTestModel):
    def setup_class(self):
        os.environ[
            "HAT_AIDI_INFERENCE_CONFIG"
        ] = "plugins/aidi_inference/configs/aidi_config.py"
        super().setup_class(self)

    def results_compare(self, src, dst):
        diff = src - dst
        return diff, bool(torch.mean(diff) < 1e-2)


if __name__ == "__main__":
    pytest.main(["-s", "-x", f"{__file__}"])
