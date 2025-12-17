import os

import horizon_plugin_pytorch as horizon
import numpy as np
import pytest
import torch
import torch.nn as nn

from hat.models.base_modules.conv_module import (
    ConvModule2d,
    ConvTransposeModule2d,
    ConvUpsample2d,
)
from tests.unit_tests.models.base import qtensor_test


def test_ConvModule2d():
    data_shape = np.random.randint(10, 20, size=4)
    fake_data = torch.tensor(
        np.random.random(size=data_shape) * 2 - 1,
        dtype=torch.float,
    )
    # norm_type = random.choice([None, nn.BatchNorm2d])
    # act_type = random.choice([None, nn.ReLU])
    for cp in ["1", "0"]:
        os.environ["HAT_USE_CHECKPOINT"] = cp
        conv_module = nn.Sequential(
            ConvModule2d(
                fake_data.shape[1],
                20,
                (3, 3),
                norm_layer=nn.BatchNorm2d(20),
                act_layer=nn.ReLU(),
            ),
            ConvModule2d(
                20,
                20,
                (3, 3),
                norm_layer=nn.BatchNorm2d(20),
                act_layer=nn.ReLU(),
            ),
        )
        output = conv_module(fake_data)
        assert output is not None

        # qat checkpoint test
        for mod in conv_module:
            mod.fuse_model()
        conv_module.qconfig = horizon.quantization.get_default_qat_qconfig()
        horizon.quantization.prepare_qat(conv_module, inplace=True)
        qat_fake_data = qtensor_test(fake_data)
        output = conv_module(qat_fake_data)
        assert output is not None


def test_ConvTransposeModule2d():
    # Out​=(In​−1)×stride−2×pad+dilation×(k_size−1)+output_pad+1
    fake_data = torch.randn(1, 3, 64, 64)
    conv_module = ConvTransposeModule2d(
        in_channels=fake_data.shape[1],
        out_channels=20,
        kernel_size=3,
        stride=2,
        padding=1,
        output_padding=1,
        norm_layer=nn.BatchNorm2d(20),
        act_layer=nn.ReLU(),
    )
    output = conv_module(fake_data)
    assert output.shape == (1, 20, 128, 128)


def test_ConvUpsample2d():
    fake_data = torch.randn(1, 3, 64, 64)
    conv_module = ConvUpsample2d(
        in_channels=fake_data.shape[1],
        out_channels=20,
        kernel_size=3,
        stride=2,
        padding=1,
        norm_layer=nn.BatchNorm2d(20),
        act_layer=nn.ReLU(),
    )
    output = conv_module(fake_data)
    assert output.shape == (1, 20, 128, 128)


@pytest.mark.serial_task
@pytest.mark.parametrize(
    ["training"],
    [
        pytest.param(True),
        pytest.param(False),
    ],
)
def test_ConvModule2dSavedtensor(training):
    """
    Test compute result for savedtensor and non savedtensor mode
    """
    data_shape = np.random.randint(10, 20, size=4)
    label_shape = [data_shape[0]]
    fake_data = torch.tensor(
        np.random.random(size=data_shape) * 2 - 1,
        dtype=torch.float,
    )

    # create module with savedtensor mode
    class TestModule(torch.nn.Module):
        def __init__(self):
            super(TestModule, self).__init__()
            self.conv_module = nn.Sequential(
                ConvModule2d(
                    fake_data.shape[1],
                    20,
                    (3, 3),
                    padding=1,
                    bias=False,
                    norm_layer=nn.BatchNorm2d(20),
                    act_layer=nn.ReLU(inplace=False),
                ),
                ConvModule2d(
                    20,
                    20,
                    (3, 3),
                    padding=1,
                    norm_layer=nn.BatchNorm2d(20),
                    # act_layer=nn.ReLU(inplace=True),
                ),
                ConvModule2d(
                    20,
                    20,
                    (3, 3),
                    padding=1,
                    norm_layer=nn.BatchNorm2d(20),
                    act_layer=None,
                ),
                ConvModule2d(
                    20,
                    20,
                    (3, 3),
                    padding=1,
                    bias=False,
                    norm_layer=nn.BatchNorm2d(20),
                    act_layer=None,
                ),
                ConvModule2d(
                    20,
                    20,
                    (3, 3),
                    padding=1,
                    bias=False,
                    norm_layer=nn.BatchNorm2d(20),
                    act_layer=nn.ReLU(inplace=False),
                ),
                ConvModule2d(
                    20,
                    20,
                    (3, 3),
                    padding=1,
                    bias=False,
                    norm_layer=nn.BatchNorm2d(20),
                    act_layer=nn.ReLU(inplace=True),
                ),
            )
            self.conv2 = ConvModule2d(
                fake_data.shape[1],
                20,
                (3, 3),
                padding=1,
                norm_layer=nn.BatchNorm2d(20),
                act_layer=nn.ReLU(inplace=False),
            )
            self.flat = torch.nn.Flatten()

        def forward(self, x):
            out1 = self.conv_module(x)
            out2 = self.conv2(x)
            res = self.flat(out1 + out2)
            return res

    # use deterministic
    # because the conv result is not always same for default non-deterministic
    torch.use_deterministic_algorithms(True)

    os.environ["HAT_USE_SAVEDTENSOR"] = "1"
    ckp_module = TestModule()
    ckp_module.cuda()

    os.environ["HAT_USE_SAVEDTENSOR"] = "0"
    non_ckp_module = TestModule()
    non_ckp_module.cuda()

    if training:
        ckp_module.train()
        non_ckp_module.train()
    else:
        ckp_module.eval()
        non_ckp_module.eval()

    # ensure same params weight and bias
    for p1, p2 in zip(ckp_module.parameters(), non_ckp_module.parameters()):
        p2.data.copy_(p1)

    for b1, b2 in zip(ckp_module.buffers(), non_ckp_module.buffers()):
        b2.data.copy_(b1)

    def reset_grad(m1, m2):
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            p1.grad = None
            p2.grad = None

    criterion1 = nn.CrossEntropyLoss().cuda()
    criterion2 = nn.CrossEntropyLoss().cuda()

    for _ in range(10):
        fake_data = torch.tensor(
            np.random.random(size=data_shape) * 2 - 1,
            dtype=torch.float,
        ).cuda()
        label_data = torch.randint(low=0, high=1000, size=label_shape).cuda()

        reset_grad(ckp_module, non_ckp_module)

        ckp_out = ckp_module(fake_data)
        non_ckp_out = non_ckp_module(fake_data)

        assert ckp_out is not None
        assert non_ckp_out is not None

        # check output are same
        assert torch.allclose(ckp_out, non_ckp_out)

        if training:
            l1 = criterion1(ckp_out, label_data)
            l2 = criterion2(non_ckp_out, label_data)

            assert l1 == l2

            l1.backward()
            l2.backward()

            # check buffers are same
            for b1, b2 in zip(ckp_module.buffers(), non_ckp_module.buffers()):
                assert torch.allclose(b1, b2)

            # check all grads are same
            for p1, p2 in zip(
                ckp_module.parameters(), non_ckp_module.parameters()
            ):
                assert torch.allclose(p1, p2)
                assert p1.grad is not None and p2.grad is not None
                assert torch.allclose(p1.grad, p2.grad)

    # reset to default
    torch.use_deterministic_algorithms(False)
