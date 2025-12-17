import numpy as np
import pytest
import torch

from hat.utils.apply_func import (
    _as_list,
    _as_numpy,
    _get_keys_from_dict,
    _is_increasing_sequence,
    convert_numpy,
    multi_apply,
    regroup,
    to_cuda,
    to_flat_ordered_dict,
)


def test_as_list():
    assert _as_list(1) == [1]
    assert _as_list([1]) == [1]
    assert _as_list("str") == ["str"]
    t = torch.randn((1, 3, 100, 100))
    assert _as_list(t) == [t]


def test_as_numpy():
    y = _as_numpy([1, 2, 3])
    assert isinstance(y, np.ndarray)

    x = np.array([1])
    y = _as_numpy([x, x, x])
    assert isinstance(y, np.ndarray)


def test_convert_numpy():
    x = torch.randn((1, 3, 100, 100))
    assert isinstance(convert_numpy(x), np.ndarray)
    assert isinstance(convert_numpy(x, to_list=True), list)
    assert convert_numpy(x, dtype="int").dtype == np.int64
    assert convert_numpy(x, dtype=np.float32).dtype == np.float32


def test_get_keys_from_dict():
    x = dict(
        type="typename",
        data=dict(
            name="innername",
        ),
    )
    assert _get_keys_from_dict(x, "type") == ["typename"]
    assert _get_keys_from_dict(x, "name") == ["innername"]
    assert _get_keys_from_dict(x, "not_exist") == []


def test_is_increasing_sequence():
    assert not _is_increasing_sequence([1, 1, 3, 4], strict=True)
    assert _is_increasing_sequence([1, 1, 3, 4], strict=False)
    assert _is_increasing_sequence([1], strict=False)


def test_multi_apply():
    def dummy_func(x, b, k):
        y = x * k + b
        return (y,)

    x = [1, 2, 3, 4]
    b = [1, 2, 3, 4]

    results = multi_apply(dummy_func, x, b, k=2)
    assert results == ([3, 6, 9, 12],)

    def dummy_func_no_tuple_return(x, b, k):
        y = x * k + b
        return y

    results = multi_apply(dummy_func_no_tuple_return, x, b, k=2)
    assert results == (3, 6, 9, 12)


def test_to_flat_ordered_dict():
    obj = dict(a=[dict(c=1)], d=(2, 3))
    flat_dict = to_flat_ordered_dict(obj, key_prefix="test")
    assert len(flat_dict) == 3, len(flat_dict)
    assert flat_dict["test_a_0_c"] == 1
    assert flat_dict["test_d_0"] == 2
    assert flat_dict["test_d_1"] == 3

    flat_dict = to_flat_ordered_dict(
        obj,
        key_prefix="test",
        flat_condition=lambda k, v: not isinstance(v, tuple),
    )
    assert len(flat_dict) == 2, len(flat_dict)
    assert flat_dict["test_a_0_c"] == 1
    assert flat_dict["test_d"] == (2, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="require gpu")
def test_to_cuda():
    x = torch.randn(10)
    assert not x.is_cuda

    x_gpu = to_cuda(x)
    assert x_gpu.is_cuda

    with pytest.raises(NotImplementedError):
        x_gpu = to_cuda(x, inplace=True)

    x_gpu = to_cuda([x, x])
    assert x_gpu[0].is_cuda and x_gpu[1].is_cuda

    x_gpu = to_cuda(dict(data=x))
    assert x_gpu["data"].is_cuda

    test_mod = torch.nn.Conv2d(3, 8, (1, 1))
    adam = torch.optim.Adam(test_mod.parameters(), lr=0.01)
    # add fake state to optimizer
    adam.state["state_name"] = dict(x=x)

    to_cuda(adam, inplace=True)
    assert adam.state["state_name"]["x"].is_cuda

    with pytest.raises(AssertionError):
        to_cuda(adam, inplace=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="require gpu")
def test_to_cuda_with_dup_tensor():
    x = torch.randn(10)
    assert not x.is_cuda

    dup_dict = {
        "x": x,
        "y": x,
    }

    dup_dict_gpu = to_cuda(dup_dict)
    assert dup_dict_gpu["x"].is_cuda
    assert dup_dict_gpu["y"].is_cuda
    assert id(dup_dict_gpu["x"]) == id(dup_dict_gpu["y"])


def test_regroup():
    flats = [0, 1, 2]
    fmt = (
        dict,
        (
            ("a", (list, ((dict, (("c", object),)),))),
            ("d", (tuple, (object, object))),
        ),
    )
    flats = list(flats)

    obj, obj_idx = regroup(tuple(flats), fmt)
    assert len(tuple(flats)) == obj_idx, obj_idx

    assert obj["a"][0]["c"] == 0
    assert obj["d"][0] == 1
    assert obj["d"][1] == 2
