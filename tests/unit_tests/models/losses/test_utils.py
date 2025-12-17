import pytest
import torch

from hat.models.losses.utils import (
    gather_feat,
    transpose_and_gather_feat,
    weight_reduce_loss,
)


@pytest.mark.parametrize(
    ["loss", "weight", "reduction", "inplace"],
    [
        pytest.param(torch.randn((2, 4, 8, 8)), None, None, False),
        pytest.param(
            torch.randn((2, 4, 8, 8)), torch.randn((2, 4, 8, 8)), None, False
        ),
        pytest.param(
            torch.randn((2, 4, 8, 8)), torch.randn((1,)), None, False
        ),
        pytest.param(
            torch.randn((2, 4, 8, 8)), torch.randn((2, 4, 8, 8)), "mean", False
        ),
        pytest.param(torch.randn((2, 4, 8, 8)), torch.randn((1,)), None, True),
    ],
)
def test_weight_reduce_loss(loss, weight, reduction, inplace):

    _ = weight_reduce_loss(loss, weight, reduction=reduction, inplace=inplace)


def test_gather_feat():
    input_ = torch.arange(0, 9).reshape(1, 3, 3)
    ind = torch.zeros(1, 3).type(torch.int64)
    out = gather_feat(input_, ind)

    expected = torch.tensor([[[0, 1, 2], [0, 1, 2], [0, 1, 2]]])

    assert torch.all(out == expected)


def test_transpose_and_gather_feat():
    input_ = torch.arange(0, 18).reshape(1, 2, 3, 3)
    ind = torch.zeros(1, 3).type(torch.int64)
    out = transpose_and_gather_feat(input_, ind)
    expected = torch.tensor([[[0, 9], [0, 9], [0, 9]]])
    assert torch.all(out == expected)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
