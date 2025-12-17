import pytest

from hat.core.compose_transform import Compose
from hat.data.datasets.voc import PascalVOC, VOCFromImage
from hat.data.transforms.detection import (
    MinIoURandomCrop,
    Normalize,
    RandomExpand,
    RandomFlip,
    Resize,
    ToTensor,
)
from hat.utils.package_helper import check_packages_available

transforms = Compose(
    [
        RandomExpand(ratio_range=(1, 4)),
        MinIoURandomCrop(min_ious=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)),
        RandomFlip(),
        Resize(img_scale=[(416, 416)], keep_ratio=False),
        ToTensor(to_yuv=True),
        Normalize(mean=128.0, std=128.0),
    ]
)


@pytest.mark.parametrize(
    "transforms, batch_size, pack_type",
    [(transforms, 16, None), (transforms, 16, "lmdb")],
)
def test_voc(transforms, batch_size, pack_type):
    dataset = PascalVOC(
        pack_type=pack_type,
        data_path="./tmp_data/voc/trainval_lmdb/",
        transforms=transforms,
    )

    assert len(dataset) == 16551
    for ind, data in enumerate(dataset):
        img, gt_bboxes, gt_classes = (
            data["img"],
            data["gt_bboxes"],
            data["gt_classes"],
        )
        print(img.shape, gt_bboxes.shape, gt_classes.shape)
        if ind > 10:
            break


@pytest.mark.skipif(
    not check_packages_available("torchvision", raise_exception=False),
    reason="need torchvision",
)
@pytest.mark.skip(
    "The testcast depends fragmentary images on bucket, very slow."
)
def test_voc_from_image():
    dataset = VOCFromImage(
        root="./tmp_orig_data/voc/",
        year="2007",
        image_set="test",
    )
    for ind, data in enumerate(dataset):
        img, gt_bboxes, gt_classes = (
            data["img"],
            data["gt_bboxes"],
            data["gt_classes"],
        )
        print(img, gt_bboxes, gt_classes)
        if ind > 10:
            break
