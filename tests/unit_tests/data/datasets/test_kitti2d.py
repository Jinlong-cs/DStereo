import pytest
import torchvision

from hat.data.datasets.kitti2d import Kitti2D
from hat.data.transforms.detection import Normalize, Pad, ToTensor

transforms = torchvision.transforms.Compose(
    [
        Pad(size=(384, 1248)),
        ToTensor(to_yuv=True),
        Normalize(mean=128.0, std=128.0),
    ]
)


@pytest.mark.parametrize(
    "transforms, batch_size, pack_type",
    [(transforms, 2, None), (transforms, 10, "lmdb")],
)
def test_kitti2d(transforms, batch_size, pack_type):
    dataset = Kitti2D(
        pack_type=pack_type,
        data_path="./tmp_data/kitti2d/kitti_train_lmdb/",
        transforms=transforms,
    )

    assert len(dataset) == 7113
    for ind, data in enumerate(dataset):
        img, gt_bboxes, gt_classes = (
            data["img"],
            data["gt_bboxes"],
            data["gt_classes"],
        )
        print(img.shape, gt_bboxes.shape, gt_classes.shape)
        if ind > 10:
            break
