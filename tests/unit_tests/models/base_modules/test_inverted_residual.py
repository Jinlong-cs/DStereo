import numpy as np
import torch

from hat.models.base_modules.inverted_residual import InvertedResidual


def test_InvertedResidual():
    data_shape = np.random.randint(10, 20, size=4)
    fake_data = torch.tensor(
        np.random.random(size=data_shape) * 2 - 1,
        dtype=torch.float,
    )
    ir_module = InvertedResidual(
        fake_data.shape[1],
        20,
        1,
        2.0,
        bn_kwargs={},
    )
    output = ir_module(fake_data)
    assert output is not None
