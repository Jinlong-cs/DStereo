import pytest
import torch

from hat.core.adapter import TorchVisionAdapter
from hat.core.compose_transform import Compose
from hat.data.datasets.imagenet import ImageNet, ImageNetFromImage
from hat.utils.package_helper import check_packages_available

transforms = Compose(
    [
        TorchVisionAdapter(
            interface="RandomResizedCrop",
            size=224,
            scale=(0.08, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
        ),
        TorchVisionAdapter(
            interface="RandomHorizontalFlip",
        ),
        TorchVisionAdapter(
            interface="ConvertImageDtype",
            dtype=torch.float32,
        ),
        TorchVisionAdapter(
            interface="Normalize",
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


@pytest.mark.parametrize(
    "transforms, batch_size, pack_type",
    [(None, 1, None), (transforms, 50, "lmdb")],
)
def test_imagenet(transforms, batch_size, pack_type):
    dataset = ImageNet(
        pack_type=pack_type,
        data_path="./tmp_data/imagenet/val_lmdb/",
        transforms=transforms,
    )

    assert len(dataset) == 50000

    for ind, data in enumerate(dataset):
        image, target = data["img"], data["labels"]
        print(image.shape, target)
        if ind > 10:
            break


@pytest.mark.skipif(
    not check_packages_available("torchvision", raise_exception=False),
    reason="need torchvision",
)
@pytest.mark.skip(
    "The testcast depends fragmentary images on bucket, very slow."
)
def test_imagenet_from_image():
    dataset = ImageNetFromImage(
        root="./tmp_orig_data/imagenet/val/",
        split="val",
    )

    assert len(dataset) == 50000

    for index, data in enumerate(dataset):
        image, target = data["img"], data["labels"]
        print(image, target)
        if index > 10:
            break
