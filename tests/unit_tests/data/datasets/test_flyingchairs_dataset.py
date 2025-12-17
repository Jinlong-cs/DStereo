import os

import pytest
import torchvision
from torchvision.transforms import InterpolationMode

from hat.core.adapter import TorchVisionAdapter
from hat.data.datasets.flyingchairs_dataset import (
    FlyingChairs,
    FlyingChairsFromImage,
)
from hat.data.transforms.classification import BgrToYuv444
from hat.data.transforms.detection import RandomCrop, RandomFlip, ToTensor
from hat.data.transforms.segmentation import (
    FlowRandomAffineScale,
    SegRandomAffine,
)

transforms = torchvision.transforms.Compose(
    [
        RandomCrop(
            size=(256, 448),
        ),
        RandomFlip(
            px=0.5,
            py=0.5,
        ),
        ToTensor(
            to_yuv=False,
        ),
        SegRandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            interpolation=InterpolationMode.BILINEAR,
            label_fill_value=0,
            translate_p=0.5,
            scale_p=0.5,
        ),
        FlowRandomAffineScale(
            scale_p=0.5,
            scale_r=0.05,
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


data_path = os.path.join("./tmp_orig_data/FlyingChairs")

PATH_EXIST = False
if os.path.exists(data_path):
    PATH_EXIST = True
    with open(data_path + "/FlyingChairs_train_val.txt", "r") as f:
        train_val_IDs = f.readlines()
    train_val_IDs = list(map(int, train_val_IDs))
    num_img_train = len(
        [idx for idx, value in enumerate(train_val_IDs) if value == 1]
    )

    num_img_test = len(
        [idx for idx, value in enumerate(train_val_IDs) if value == 2]
    )


@pytest.mark.parametrize(
    "transforms, batch_size, pack_type",
    [(transforms, 1, "lmdb"), (None, 50, None)],
)
def test_flying_chairs(transforms, batch_size, pack_type):
    dataset = FlyingChairs(
        pack_type=pack_type,
        data_path="./tmp_data/FlyingChairs/val_lmdb/",
        transforms=transforms,
    )
    state = dataset.__getstate__()
    dataset.__setstate__(state)
    assert len(dataset) == num_img_test

    for ind, data in enumerate(dataset):
        image, gt_flow = data["img"], data["gt_flow"]
        print(image.shape, gt_flow.shape)
        if ind > 10:
            break


@pytest.mark.skipif(not PATH_EXIST, reason="path does not exist")
def test_flyingchairs_fromimage():
    dataset_train = FlyingChairsFromImage(data_path, train_flag=True)
    for ind, data in enumerate(dataset_train):
        image, gt_flow = (data["img"], data["gt_flow"])
        print(image.shape, gt_flow.shape)
        if ind > 10:
            break
    assert len(dataset_train) == num_img_train

    dataset_test = FlyingChairsFromImage(data_path, train_flag=False)
    for ind, data in enumerate(dataset_test):
        image, gt_flow = (data["img"], data["gt_flow"])
        print(image.shape, gt_flow.shape)
        if ind > 10:
            break
    assert len(dataset_test) == num_img_test


if __name__ == "__main__":
    pytest.main(["-s", __file__])
