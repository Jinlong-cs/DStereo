import pytest

from hat.core.compose_transform import Compose
from hat.data.datasets.mscoco import Coco, CocoFromImage
from hat.data.transforms.detection import Normalize, Pad, Resize, ToTensor
from hat.utils.package_helper import check_packages_available

transforms = Compose(
    [
        Resize(img_scale=(800, 1024), keep_ratio=True),
        Pad(size=(1024, 1024)),
        ToTensor(to_yuv=True),
        Normalize(mean=128.0, std=128.0),
    ]
)


@pytest.mark.skipif(
    not check_packages_available("pycocotools", raise_exception=False),
    reason="need pycocotools",
)
@pytest.mark.skipif(
    not check_packages_available("torchvision", raise_exception=False),
    reason="need torchvision",
)
@pytest.mark.parametrize(
    "transforms, batch_size, pack_type",
    [(transforms, 2, None), (transforms, 2, "lmdb")],
)
def test_mscoco(transforms, batch_size, pack_type):
    dataset = Coco(
        pack_type=pack_type,
        data_path="./tmp_data/mscoco/val_lmdb/",
        transforms=transforms,
    )

    assert len(dataset) == 5000
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
    not check_packages_available("pycocotools", raise_exception=False),
    reason="need pycocotools",
)
@pytest.mark.skipif(
    not check_packages_available("torchvision", raise_exception=False),
    reason="need torchvision",
)
@pytest.mark.skip(
    "The testcast depends fragmentary images on bucket, very slow."
)
def test_coco_from_image():
    dataset = CocoFromImage(
        root="./tmp_orig_data/mscoco/val2017",
        annFile="./tmp_orig_data/mscoco/annotations/instances_val2017.json",
    )

    assert len(dataset) == 5000
    for index, data in enumerate(dataset):
        img, gt_bboxes, gt_classes = (
            data["img"],
            data["gt_bboxes"],
            data["gt_classes"],
        )
        print(img.shape, gt_bboxes.shape, gt_classes.shape)
        if index > 10:
            break
