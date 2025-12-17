# Copyright (c) Horizon Robotics. All rights reserved.

import pytest
import torchvision
from torch.utils.data import DataLoader

from hat.data.datasets.iris_dataset import RecDataset
from hat.data.transforms.detection import ToTensor
from hat.registry import build_from_registry

TRANSFORMS = torchvision.transforms.Compose([ToTensor()])


@pytest.mark.parametrize(
    "visible_index, invisible_index", [([0, 2], [1, 3, 5])]
)
def test_rec_dataset(visible_index, invisible_index):

    rec_path = "./tmp_orig_data/face/iris/train_iris_cd569_20210527_cls_2.rec"
    idx_path = rec_path.replace(".rec", ".idx")
    batch_size = 16
    dataset = RecDataset(
        rec_path,
        idx_path,
        TRANSFORMS,
        visible_index=visible_index,
        invisible_index=invisible_index,
    )

    for idx, batch in enumerate(dataset):
        img = batch["img"]
        label = batch["labels"]
        if idx > 10:
            break

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    for idx, batch in enumerate(dataloader):
        print(idx)
        img = batch["img"]
        label = batch["labels"]
        assert img.shape == (batch_size, 3, 120, 200)
        assert len(label) == 7
        if idx > 5:
            break


@pytest.mark.parametrize("transforms, batch_size", [(TRANSFORMS, 256)])
def test_iris_dataset(transforms, batch_size):

    data_cfg = dict(
        type="IrisDataset",
        rec_list=[
            "./tmp_orig_data/face/iris/" "train_iris_cd569_20210527_cls_2.rec",
            "./tmp_orig_data/face/iris/" "train_iris_20211116.rec",
        ],
        transforms=transforms,
        visible_index=[0, 2],
        invisible_index=[1, 3, 4],
        shuffle=True,
    )
    dataset = build_from_registry(data_cfg)

    for idx, batch in enumerate(dataset):
        img = batch["img"]
        label = batch["labels"]
        if idx > 10:
            break

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    assert int(len(dataloader)) == round(len(dataset) / batch_size)
    for idx, batch in enumerate(dataloader):
        print(idx)
        img = batch["img"]
        label = batch["labels"]
        assert img.size()[0] == batch_size
        assert img.size()[1] == 3
        assert img.size()[2] == 120
        assert img.size()[3] == 200
        assert len(label) == 7
        if idx > 5:
            break


if __name__ == "__main__":
    pytest.main(["-s", __file__])
