import pytest
import torch

from hat.registry import build_from_registry


@pytest.mark.parametrize(
    [
        "channels_list",
        "repeats",
        "group_list",
        "factor_list",
        "flat_output",
        "include_top",
        "deep_stem",
    ],
    [
        pytest.param(
            [4, 8, 16, 32, 64],
            [2, 2, 2, 2],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            False,
            False,
            True,
        ),
        pytest.param(
            [4, 8, 16, 32, 64],
            [3, 3, 3, 3],
            [1, 2, 4, 8],
            [1, 1, 1, 1],
            False,
            True,
            False,
        ),
        pytest.param(
            [4, 4, 4, 4, 4],
            [2, 3, 4, 5],
            [2, 2, 2, 2],
            [1, 1, 2, 2],
            True,
            True,
            False,
        ),
    ],
)
def test_vargconvnet(
    channels_list,
    repeats,
    group_list,
    factor_list,
    flat_output,
    include_top,
    deep_stem,
):
    num_classes = 10
    cfg = dict(
        type="VargConvNet",
        channels_list=channels_list,
        repeats=repeats,
        group_list=group_list,
        factor_list=factor_list,
        flat_output=flat_output,
        include_top=include_top,
        num_classes=num_classes,
        bn_kwargs={},
    )
    model = build_from_registry(cfg)
    input_size = 480
    x = torch.randn((1, 3, input_size, input_size))
    y = model(x)

    if not include_top:
        assert isinstance(y, list) and len(y) == 5
        assert y[0].shape[-1] == input_size / 2
        assert y[1].shape[-1] == input_size / 4
        assert y[2].shape[-1] == input_size / 8
        assert y[3].shape[-1] == input_size / 16
        assert y[4].shape[-1] == input_size / 32
    else:
        assert isinstance(y, torch.Tensor)
        assert y.shape[1] == num_classes
