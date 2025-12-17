import os
from distutils.version import LooseVersion

import pytest
import torch

# isort: off
from hat.utils.saved_tensor import (
    checkpoint_convbn_with_saved_tensor,
    checkpoint_with_saved_tensor,
    support_saved_tensor,
)

# isort: on


@pytest.mark.serial_task
def test_support_saved_tensor():
    support_ver = LooseVersion(torch.__version__) >= LooseVersion("1.10.2")
    os.environ["HAT_USE_SAVEDTENSOR"] = "0"
    assert support_saved_tensor() is False
    os.environ["HAT_USE_SAVEDTENSOR"] = "1"
    assert support_saved_tensor() == support_ver
    os.environ["HAT_USE_SAVEDTENSOR"] = "0"


class TestModel(torch.nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv = torch.nn.Conv2d(3, 10, kernel_size=3)
        self.bn = torch.nn.BatchNorm2d(10)
        self.act = torch.nn.ReLU()

    def forward(self, data):
        output = self.act(self.bn(self.conv(data)))
        return output

    def saved_tensor(self):
        print("checkpoint with saved tensor")


@pytest.mark.serial_task
def test_checkpoint_convbn_with_saved_tensor():
    os.environ["HAT_USE_SAVEDTENSOR"] = "1"
    mod = TestModel()
    fake_data = torch.rand(1, 3, 10, 10)
    if support_saved_tensor():
        fn = checkpoint_convbn_with_saved_tensor(mod.conv, mod.bn, mod.act)
        mod.eval()
        o1 = fn(fake_data)
        o2 = mod(fake_data)
        torch.allclose(o1, o2)
        mod.train()
        fn(fake_data)
    os.environ["HAT_USE_SAVEDTENSOR"] = "0"


def test_checkpoint_with_saved_tensor():
    mod = TestModel()
    checkpoint_with_saved_tensor(mod)
