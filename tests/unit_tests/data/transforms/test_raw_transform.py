# Copyright (c) Horizon Robotics. All rights reserved.
from copy import deepcopy

import numpy as np
import pytest
import torch

from hat.data.transforms.raw_transform import (
    BLC,
    DEC,
    DGain,
    NNLogLut,
    RawPack,
    RawPad,
)
from tests.utils import gen_fake_transforms_data


@pytest.mark.parametrize("resize_gt", [True, False])
@pytest.mark.parametrize("layout", ["hwc", "chw"])
def test_raw_pack(resize_gt, layout):
    data = gen_fake_transforms_data(300, 400, layout, c=1)
    src_data = data.copy()
    raw_packer = RawPack("pack", resize_gt)
    packed_data = raw_packer(data).copy()
    raw_unpacker = RawPack("unpack", resize_gt)
    unpacked_data = raw_unpacker(packed_data).copy()

    assert (unpacked_data["img"] == src_data["img"]).all()


def test_inverse_transform():
    data = gen_fake_transforms_data(128, 64, layout="hwc", c=1)
    raw_packer = RawPack(method="pack", resize_gt=True)
    for _ in range(100):
        for op in [raw_packer]:
            gt = op(deepcopy(data))
            gt_bboxes = op.inverse_transform(
                torch.tensor(gt["gt_bboxes"]), "detection", gt
            )
            assert (gt_bboxes.long().numpy() == data["gt_bboxes"]).all()


@pytest.mark.parametrize("layout", ["hwc", "chw"])
@pytest.mark.parametrize("method", ["padself", "padzero"])
def test_raw_pad(layout, method):
    data = gen_fake_transforms_data(300, 400, layout, c=1, fill_value=0)
    raw_pader = RawPad(method)
    pad_data = raw_pader(data).copy()
    tar_data = gen_fake_transforms_data(300, 400, layout, c=3, fill_value=0)

    assert (pad_data["img"] == tar_data["img"]).all()


@pytest.mark.parametrize(
    ["sensor", "input_bit", "output_bit"],
    [
        pytest.param("ovx8b", 12, 20),
        pytest.param("ovx8b", 12, 24),
        pytest.param("ar0820", 12, 20),
    ],
)
def test_dec(sensor, input_bit, output_bit):
    data = {"img": np.random.randint(0, 2 ** input_bit, size=(300, 400, 1))}
    dec = DEC(sensor, input_bit, output_bit)
    dec_data = dec(data).copy()
    assert (
        dec_data["img"].dtype == np.uint32
        and dec_data["img"].min() >= 0
        and dec_data["img"].max() <= 2 ** output_bit - 1
    )


@pytest.mark.parametrize(
    ["output_bit"],
    [
        pytest.param(20),
        pytest.param(24),
    ],
)
def test_dgain(output_bit):
    data = {
        "img": np.random.randint(0, 2 ** 20, size=(300, 400, 1)).astype(
            np.uint32
        ),
        "dgain": np.random.uniform(1.0, 2.0),
    }
    dgain = DGain(output_bit)
    dgain_data = dgain(data).copy()
    assert (
        dgain_data["img"].min() >= 0
        and dgain_data["img"].max() <= 2 ** output_bit - 1
    )


@pytest.mark.parametrize(
    ["sensor", "output_bit"],
    [
        pytest.param("ovx8b", 20),
        pytest.param("ar0820", 20),
    ],
)
def test_blc(sensor, output_bit):
    data = {
        "img": np.random.randint(0, 2 ** 20, size=(300, 400, 1)).astype(
            np.uint32
        ),
    }
    blc = BLC(sensor, output_bit)
    blc_data = blc(data).copy()
    assert (
        blc_data["img"].min() >= 0
        and blc_data["img"].max() <= 2 ** output_bit - 1
    )


@pytest.mark.parametrize(
    ["input_bit", "output_bit", "pregamma"],
    [
        pytest.param(20, 16, None),
        pytest.param(20, 16, 1 / 3.0),
    ],
)
def test_nnloglut(input_bit, output_bit, pregamma):
    data = {
        "img": np.random.randint(0, 2 ** input_bit, size=(300, 400, 1)),
    }
    lut_x = np.linspace(0, 1, 64)
    lut_y = np.random.uniform(0, 1, size=64)
    lut_y = np.cumsum(lut_y) / np.sum(lut_y)
    nnloglut = NNLogLut(input_bit, output_bit, lut_x, lut_y, pregamma)
    nnloglut_data = nnloglut(data).copy()
    assert (
        nnloglut_data["img"].min() >= 0
        and nnloglut_data["img"].max() <= 2 ** output_bit - 1
    )


if __name__ == "__main__":
    pytest.main(["-s", __file__])
