import horizon_plugin_pytorch as horizon
import numpy as np
import pytest
import torch
import torch.nn as nn

from hat.models.base_modules.conv_factory import get_conv_module
from tests.unit_tests.models.base import qtensor_test


@pytest.mark.parametrize(
    ["conv_method"],
    [
        pytest.param("varg_conv"),
        pytest.param("sep_conv"),
        pytest.param("conv"),
    ],
)
def test_ConvModule2d(conv_method):
    data_shape = np.random.randint(10, 20, size=4)
    fake_data = torch.tensor(
        np.random.random(size=data_shape) * 2 - 1,
        dtype=torch.float,
    )
    conv_module = get_conv_module(
        in_channels=data_shape[1],
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        conv_method=conv_method,
        dw_activation=None,
        pw_activation=nn.ReLU,
        dw_norm_method=None,
        pw_norm_method=nn.BatchNorm2d,
    )
    # qat checkpoint test
    conv_module.fuse_model()
    conv_module.qconfig = horizon.quantization.get_default_qat_qconfig()
    horizon.quantization.prepare_qat(conv_module, inplace=True)
    qat_fake_data = qtensor_test(fake_data)
    output = conv_module(qat_fake_data)
    assert output is not None
