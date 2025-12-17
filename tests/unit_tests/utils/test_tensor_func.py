import numpy as np
import pytest
import torch

from hat.utils.tensor_func import (
    divide_no_nan,
    insert_row,
    mean_with_mask,
    select_sample,
    take_row,
    tensor_mean,
)


@pytest.mark.parametrize(
    ["B", "N", "rest_dims", "M"],
    [
        pytest.param(3, 10, [5], 6),
        pytest.param(5, 20, [4, 6], 10),
        pytest.param(3, 5, [2, 5, 7], 3),
    ],
)
def test_take_row(B, N, rest_dims, M):
    target = torch.randn([B, N] + rest_dims)
    index = torch.stack([torch.randperm(N)[:M] for _ in range(B)], dim=0)

    loop_free = take_row(target, index)
    loop_res = torch.stack([t[i] for t, i in zip(target, index)], dim=0)

    assert torch.all(loop_free == loop_res)


@pytest.mark.parametrize(
    ["B", "N", "rest_dims", "M"],
    [
        pytest.param(3, 10, [5], 6),
        pytest.param(5, 20, [4, 6], 10),
        pytest.param(3, 5, [2, 5, 7], 3),
    ],
)
def test_insert_row(B, N, rest_dims, M):
    in_tensor = torch.randn([B, N] + rest_dims)
    index = torch.stack([torch.randperm(N)[:M] for _ in range(B)], dim=0)
    target_tensor = torch.randn([B, M] + rest_dims)

    insert_row(in_tensor, index, target_tensor)
    assert torch.all(take_row(in_tensor, index) == target_tensor)


def test_select_sample():
    x = torch.tensor([1, 2, 3, 4])
    index = [True, False, True, False]
    x_sampled = select_sample(x, index)
    assert torch.all(torch.eq(x_sampled, torch.tensor([1, 3])))

    xx_sampled = select_sample([x, x], index)  # noqa: F841
    xx_sampled = select_sample((x, x), index)  # noqa: F841
    xs_sampled = select_sample({"data": x}, index)  # noqa: F841
    with pytest.raises(TypeError):
        error_sampled = select_sample(100, index)  # noqa: F841


@pytest.mark.parametrize(
    ["x", "mask", "gt"],
    [
        pytest.param(
            np.array([1, 3, 4, 11, 23]),
            np.array([True, False, False, False, True]),
            12,
        ),
        pytest.param(
            np.ones([100, 20, 2]),
            np.array([True, False]),
            np.ones([100, 20]),
        ),
        pytest.param(
            np.zeros([100, 20, 2]),
            np.array([True, False]),
            np.zeros([100, 20]),
        ),
        pytest.param(
            torch.ones([100, 20, 2]),
            torch.Tensor([True, False]),
            torch.ones([100, 20]),
        ),
    ],
)
def test_mean_with_mask(x, mask, gt):
    ret = mean_with_mask(x, mask)
    if isinstance(ret, torch.Tensor):
        assert np.all((torch.abs(ret - gt) < 1e-8).numpy())
    elif isinstance(ret, np.ndarray):
        assert np.all(np.abs(ret - gt) < 1e-8)
    else:
        assert abs(ret - gt) < 1e-8


@pytest.mark.parametrize(
    ["x", "y"],
    [
        pytest.param(
            torch.Tensor([5, -5]),
            torch.Tensor([0]),
        ),
        pytest.param(
            torch.Tensor([float("nan")]),
            torch.Tensor([2]),
        ),
    ],
)
def test_divide_no_nan(x, y):
    div = divide_no_nan(x, y)
    assert not torch.any(torch.isnan(div))
    assert not torch.any(torch.isinf(div))


@pytest.mark.parametrize(
    ["inputs", "target"],
    [
        pytest.param(
            [1, 2, 3, 4, 5],
            3,
        ),
        pytest.param(
            [2, 0, 3, 11],
            4,
        ),
    ],
)
def test_tensor_mean(inputs, target):
    tensor_list = []
    for input in inputs:
        tensor_list.append(torch.Tensor([input])[0])
    target = torch.Tensor([target])
    result = tensor_mean(tensor_list)
    assert abs(result.item() - target.item()) < 1e-8
