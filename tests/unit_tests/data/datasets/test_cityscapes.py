# Copyright (c) Horizon Robotics. All rights reserved.
import pytest

from hat.core.compose_transform import Compose
from hat.data.datasets.cityscapes import Cityscapes
from hat.data.transforms.classification import BgrToYuv444
from hat.data.transforms.common import PILToTensor
from hat.data.transforms.detection import Normalize

transforms = Compose(
    [
        PILToTensor(),
        BgrToYuv444(rgb_input=False),
        Normalize(mean=128.0, std=128.0),
    ]
)


@pytest.mark.parametrize(
    "transforms, batch_size, pack_type",
    [(transforms, 10, None), (transforms, 10, "lmdb")],
)
def test_cityscapes_dataset(transforms, batch_size, pack_type):
    dataset = Cityscapes(
        pack_type=pack_type,
        data_path="./tmp_data/cityscapes/val_lmdb/",
        transforms=transforms,
    )

    assert len(dataset) == 500
    for ind, data in enumerate(dataset):
        image, target = data["img"], data["gt_seg"]
        assert image.shape[0] == 3
        assert target.shape[0] == 1
        if ind > 10:
            break
