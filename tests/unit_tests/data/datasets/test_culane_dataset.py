import math
import os

import pytest
import torchvision
from torch.utils.data import DataLoader

from hat.core.adapter import TorchVisionAdapter
from hat.data.collates.collates import collate_2d
from hat.data.datasets.culane_dataset import (
    CuLaneDataset,
    CuLaneFromImage,
    CuLanePacker,
)
from hat.data.transforms.classification import BgrToYuv444
from hat.data.transforms.detection import ToTensor

transforms = torchvision.transforms.Compose(
    [
        ToTensor(
            to_yuv=False,
        ),
        BgrToYuv444(
            rgb_input=True,
        ),
        TorchVisionAdapter(
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ]
)


@pytest.mark.parametrize(
    "transforms, batch_size, pack_type",
    [(transforms, 1, "lmdb"), (None, 4, None)],
)
def test_culane_dataset(transforms, batch_size, pack_type):
    dataset = CuLaneDataset(
        pack_type=pack_type,
        data_path="./tmp_data/CULane/test_lmdb",
        transforms=transforms,
    )

    assert len(dataset) == 34680

    for ind, data in enumerate(dataset):
        image, gt_lines = data["img"], data["gt_lines"]
        print(image.shape, len(gt_lines))
        if ind > 10:
            break

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_2d,
    )

    assert len(dataloader) == math.ceil(len(dataset) / batch_size)
    for ind, batch in enumerate(dataloader):
        image, gt_lines = batch["img"], batch["gt_lines"]
        assert image.shape[0] == batch_size
        assert len(gt_lines) == batch_size
        print(image.shape, len(gt_lines))
        if ind > 10:
            break


data_path = os.path.join("./tmp_orig_data/CULane/data")

PATH_EXIST = False
if os.path.exists(data_path):
    PATH_EXIST = True


@pytest.mark.skipif(not PATH_EXIST, reason="path does not exist")
def test_culane_fromimage():
    dataset_train = CuLaneFromImage(data_path, train_flag=True)
    for ind, data in enumerate(dataset_train):
        image, gt_lines = (data["img"], data["gt_lines"])
        print(image.shape, len(gt_lines))
        if ind > 10:
            break
    assert len(dataset_train) == 88880

    dataset_test = CuLaneFromImage(data_path, train_flag=False)
    for ind, data in enumerate(dataset_test):
        image, gt_lines = (data["img"], data["gt_lines"])
        print(image.shape, len(gt_lines))
        if ind > 10:
            break
    assert len(dataset_test) == 34680


@pytest.mark.skipif(not PATH_EXIST, reason="path does not exist")
def test_culanedataset_packer(tmpdir):
    tmp_dir = os.path.join(tmpdir, "Culane")

    packer = CuLanePacker(
        data_path, tmp_dir, "test", 1, "lmdb", num_samples=50
    )
    packer()
    assert os.path.exists(os.path.join(tmp_dir, "data.mdb"))


if __name__ == "__main__":
    pytest.main(["-s", __file__])
