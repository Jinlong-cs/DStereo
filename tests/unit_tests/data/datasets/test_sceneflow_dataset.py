import math
import os

import pytest
import torchvision
from torch.utils.data import DataLoader

from hat.core.adapter import TorchVisionAdapter
from hat.data.collates.collates import collate_2d
from hat.data.datasets.sceneflow_dataset import (
    SceneFlow,
    SceneFlowFromImage,
    SceneFlowPacker,
)
from hat.data.transforms.classification import BgrToYuv444
from hat.data.transforms.detection import RandomCrop, ToTensor

transforms = torchvision.transforms.Compose(
    [
        RandomCrop(size=(256, 512)),
        ToTensor(to_yuv=False, use_yuv_v2=False),
        BgrToYuv444(rgb_input=True),
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
def test_sceneflow_dataset(transforms, batch_size, pack_type):
    dataset = SceneFlow(
        pack_type=pack_type,
        data_path="./tmp_data/SceneFlow/train_lmdb",
        transforms=transforms,
    )

    assert len(dataset) == 35454

    for ind, data in enumerate(dataset):
        image = data["img"]
        print(image.shape)
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
        assert len(batch["img"]) == batch_size
        if ind > 10:
            break


data_path = "./tmp_orig_data/SceneFlow"

PATH_EXIST = False
if os.path.exists(data_path):
    PATH_EXIST = True


@pytest.mark.skipif(not PATH_EXIST, reason="path does not exist")
def test_sceneflow_fromimage():
    dataset_train = SceneFlowFromImage(
        data_path=data_path,
        data_list=os.path.join(data_path, "SceneFlow_finalpass_train.txt"),
    )
    for ind, data in enumerate(dataset_train):
        image = data["img"]
        print(image.shape)
        if ind > 10:
            break
    assert len(dataset_train) == 35454

    dataset_test = SceneFlowFromImage(
        data_path=data_path,
        data_list=os.path.join(data_path, "SceneFlow_finalpass_test.txt"),
    )
    for ind, data in enumerate(dataset_test):
        image = data["img"]
        print(image.shape)
        if ind > 10:
            break
    assert len(dataset_test) == 4370


@pytest.mark.skipif(not PATH_EXIST, reason="path does not exist")
def test_sceneflow_packer(tmpdir):
    tmp_dir = os.path.join(tmpdir, "SceneFlow")

    packer = SceneFlowPacker(
        data_path, tmp_dir, "test", 1, "lmdb", num_samples=50
    )
    packer()
    assert os.path.exists(os.path.join(tmp_dir, "data.mdb"))


if __name__ == "__main__":
    pytest.main(["-s", __file__])
