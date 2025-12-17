import horizon_plugin_pytorch.nn as hnn
import pytest
import torch
import torch.nn as nn

from hat.utils.deterministic import (
    NonDeterministicOpsTensor,
    cast_model_to_deterministic,
    get_hooked_modules,
    get_hooked_ops,
    maybe_cast_batch_for_deterministic,
    reset_deterministic_level,
    set_deterministic_level,
)


class TestNet(nn.Module):
    def __init__(self) -> None:
        super(TestNet, self).__init__()
        self.conv = nn.Conv2d(3, 3, 1, 1)
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.up = hnn.Interpolate(
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
            recompute_scale_factor=True,
        )

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = torch.nn.functional.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        x = self.pool(x)
        x = self.up(x)
        return x


@pytest.mark.parametrize(
    ["add_hook"],
    [
        pytest.param(True),
        pytest.param(False),
    ],
)
def test_maybe_cast_batch_for_deterministic(add_hook):
    if add_hook:
        set_deterministic_level(level=2)

    batch = {"data": torch.randn((1, 3, 9, 9))}
    batch = maybe_cast_batch_for_deterministic(batch)
    if add_hook:
        assert isinstance(batch["data"], NonDeterministicOpsTensor)
        reset_deterministic_level()
    else:
        assert isinstance(batch["data"], torch.Tensor)


def test_deterministic_hooks():
    model = TestNet()
    model.cuda()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)
    set_deterministic_level(level=2)
    cast_model_to_deterministic(model)

    for _ in range(1):
        optimizer.zero_grad()
        batch = torch.randn(2, 3, 4, 4).cuda()
        batch = maybe_cast_batch_for_deterministic(batch)
        output = model(batch)
        output.sum().backward()
        optimizer.step()

    hooked_modules = get_hooked_modules()
    hooked_ops = get_hooked_ops()
    assert (
        "pool",
        "torch.nn.modules.pooling.AdaptiveAvgPool2d",
    ) in hooked_modules
    assert (
        "horizon_plugin_pytorch.nn.interpolate.autocasted_interpolate_outer"
        in hooked_ops
        or "torch.nn.functional.interpolate" in hooked_ops
    )
    reset_deterministic_level()
