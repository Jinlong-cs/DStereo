import os
import shutil

import pytest
import torch

from hat.callbacks.save_traced import DEPLOY_FILE_FORMAT, SaveTraced
from tests.unit_tests.base import BoringTraceModel

model = BoringTraceModel()


@pytest.fixture()
def save_traced(tmpdir):
    checkpoint = SaveTraced(
        save_dir=tmpdir,
        trace_inputs=torch.randn(1),
    )
    return checkpoint


class TestSaveTraced(object):
    def test_on_loop_end(self, save_traced):
        save_traced.on_loop_end(model=model)
        save_dir = save_traced.save_dir
        name_prefix = save_traced.name_prefix
        assert os.path.exists(
            os.path.join(save_dir, DEPLOY_FILE_FORMAT % (name_prefix, "last"))
        )
        shutil.rmtree(save_dir)
