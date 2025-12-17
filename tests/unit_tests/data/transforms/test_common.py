import copy

import numpy as np
import pytest
import torch
from PIL import Image
from torch import Tensor

from hat.data.transforms.common import BgrToYuv444V2
from hat.registry import build_from_registry
from hat.utils.apply_func import _as_list
from hat.utils.package_helper import check_packages_available


def test_list_to_dict():
    cfg = dict(
        type="ListToDict",
        keys=["image", "target"],
    )
    ts = build_from_registry(cfg)
    list_data = [torch.tensor(1.0), torch.tensor(2.0)]
    res = ts(list_data)
    assert res["image"] == list_data[0]
    assert res["target"] == list_data[1]


def test_delete_keys():
    data = dict(img=1, shape=2)
    t = build_from_registry(
        dict(
            type="DeleteKeys",
            keys=[
                "shape",
            ],
        )
    )
    result = t(data)
    assert "img" in result
    assert "shape" not in result


def test_rename_keys():
    data = dict(img=1, shape=2)
    t = build_from_registry(
        dict(
            type="RenameKeys",
            keys=["img|imgs", "shape|size"],
            split="|",
        )
    )
    result = t(data)
    assert "img" not in result
    assert "imgs" in result
    assert "size" in result


def test_repeat_keys():
    data = dict(
        imgs=[1],
        cam_intrinsic=np.zeros((1, 3, 3)),
    )
    repeat_times = 3
    t = build_from_registry(
        dict(
            type="RepeatKeys",
            keys=["imgs", "cam_intrinsic"],
            repeat_times=repeat_times,
        )
    )
    result = t(data)
    assert len(result["imgs"]) == repeat_times
    assert result["cam_intrinsic"].shape == (repeat_times, 3, 3)


@pytest.mark.skipif(
    not check_packages_available("torchvision", raise_exception=False),
    reason="need torchvision",
)
def test_pil_to_tensor():
    data = dict(img=Image.fromarray(np.random.rand(6, 6, 3).astype(np.uint8)))
    t = build_from_registry(dict(type="PILToTensor"))

    ret = t(data)
    assert isinstance(ret["img"], Tensor)


def test_tensor_to_numpy():
    data = dict(img=Tensor(np.random.rand(6, 6, 3).astype(np.uint8)))
    t = build_from_registry(dict(type="TensorToNumpy"))

    ret = t(data)
    assert isinstance(ret["img"], np.ndarray)


def test_to_cuda():
    data = dict(img=torch.rand(2, 3, 3))
    t = build_from_registry(dict(type="ToCUDA"))

    ret = t(data)
    assert ret["img"].is_cuda


def test_add_keys():
    data = dict(img=None, size=None)
    t = build_from_registry(dict(type="AddKeys", kv={"side_img": ""}))
    _ = t(data)

    t = build_from_registry(dict(type="AddKeys", kv={"size": ""}))
    with pytest.raises(AssertionError):
        _ = t(data)

    with pytest.raises(AssertionError):
        t = build_from_registry(dict(type="AddKeys", kv=()))
        _ = t(data)


def test_copy_keys():
    data = dict(img=torch.randn((1, 3, 10, 10)))
    t = build_from_registry(dict(type="CopyKeys", keys=["img|side_img"]))
    data_transformed = t(data)
    assert "img" in data_transformed
    assert "side_img" in data_transformed
    assert id(data_transformed["img"]) != id(data_transformed["side_img"])


def test_random_select_one():
    data = dict(
        img=np.array([[[2, 3, 4], [2, 3, 4], [2, 3, 4]]]).astype(np.uint8)
    )
    ori_data = copy.deepcopy(data)
    cfg = dict(
        type="RandomSelectOne",
        transforms=[
            dict(
                type="RGBShift",
                r_shift_limit=(1, 1),
                g_shift_limit=(1, 1),
                b_shift_limit=(1, 1),
                p=1.0,
            ),
            dict(
                type="RGBShift",
                r_shift_limit=(1, 1),
                g_shift_limit=(1, 1),
                b_shift_limit=(1, 1),
                p=1.0,
            ),
        ],
        p=1.0,
    )
    t = build_from_registry(cfg)
    result = t(data)

    assert (ori_data["img"][..., 0] + 1 == result["img"][..., 0]).all()
    assert (ori_data["img"][..., 1] + 1 == result["img"][..., 1]).all()
    assert (ori_data["img"][..., 2] + 1 == result["img"][..., 2]).all()


def test_multi_task_anno_wrapper():
    cfg = dict(
        type="MultiTaskAnnoWrapper",
        sub_transforms=dict(
            keep_target=[
                dict(
                    type="ListToDict",
                    keys=["image", "target", "pack"],
                )
            ],
            keep_ignore=[
                dict(
                    type="ListToDict",
                    keys=["image", "ignore", "pack"],
                )
            ],
        ),
        unikeys=("target", "ignore"),
        repkeys=("pack",),
    )
    ts = build_from_registry(cfg)
    list_data = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
    res = ts(list_data)
    assert res["keep_target"]["target"] == res["keep_ignore"]["ignore"]
    assert res["image"] == list_data[0]
    assert res["keep_target"]["pack"] == res["pack"]


def test_convert_data_type():
    data = dict(
        numpydata=np.array([[[2, 3, 4], [2, 3, 4], [2, 3, 4]]]).astype(
            np.uint8
        ),
        tensordata=torch.randn((1, 3, 10, 10)).to(torch.float64),
    )
    cfg = dict(
        type="ConvertDataType",
        convert_map={"numpydata": "float32", "tensordata": torch.int64},
    )
    t = build_from_registry(cfg)
    result = t(data)

    assert result["numpydata"].dtype == np.float32
    assert result["tensordata"].dtype == torch.int64


@pytest.mark.parametrize(
    ["dims", "input_tensor"],
    [
        pytest.param(0, True),
        pytest.param(0, False),
        pytest.param([2, 3], False),
        pytest.param([3, 2, 1], False),
        pytest.param([3, 2, 1], True),
    ],
)
def test_fix_length_pad(dims, input_tensor):
    length = 10
    dims = _as_list(dims)
    keys = [f"test_{i}" for i in range(len(dims))]
    cfg = dict(type="FixLengthPad", keys=keys, lengths=length, dims=dims)
    t = build_from_registry(cfg)

    for _ in range(10):
        data = dict()
        for i, key in enumerate(keys):
            ndim = int(np.random.random()) * 2 + dims[i] + 1
            shape = []
            for _ in range(ndim):
                shape.append(int(np.random.random() * 15) + 1)
            data[key] = np.zeros(shape=shape)
            if input_tensor:
                data[key] = torch.from_numpy(data[key])

        result = t(data)

        for dim, key in zip(dims, keys):
            assert result[key].shape[dim] == length


@pytest.mark.parametrize(
    "swing",
    ["studio", "full"],
)
@pytest.mark.parametrize(
    "device",
    ["cpu", "cuda"],
)
def test_bgr2yuv444_v2(swing, device):
    fake_data = torch.randint(
        0, 255, (3, 224, 224), dtype=torch.uint8, device=device
    )
    bgr2yuv = BgrToYuv444V2(swing=swing)
    yuv_img = bgr2yuv(fake_data)
    assert yuv_img.shape == (3, 224, 224)
