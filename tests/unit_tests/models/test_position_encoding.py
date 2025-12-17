import pytest
import torch

from hat.models.base_modules.activation import DynamicWeight, Scale


@pytest.mark.parametrize(
    ["scale"],
    [
        pytest.param(0.5),
        pytest.param(2.0),
    ],
)
def test_scale(scale):
    m = Scale(scale=scale)
    x = torch.tensor(1.0)
    y = m(x)
    assert y == x * scale


def test_dynamic_weight():
    x = torch.tensor(1.0)
    w = DynamicWeight()
    y = w(x)
    assert y == torch.exp(-w.scale) * x
    w = DynamicWeight(5.5)
    y = w(x)
    assert y == torch.exp(-w.scale) * x
    assert w.scale == 5.5


if __name__ == "__main__":
    pytest.main(["-s", __file__])
