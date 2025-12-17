import pytest

from hat.core.adapter import TorchVisionAdapter
from hat.data.datasets.batch_transform_dataset import BatchTransformDataset
from hat.data.datasets.mscoco import Coco
from hat.data.datasets.voc import PascalVOC
from hat.data.transforms.detection import ToTensor


@pytest.mark.parametrize("dataset", [Coco, PascalVOC])
@pytest.mark.parametrize("epoch_steps", [1, 2])
def test_batch_transform_dataset(dataset, epoch_steps):
    if dataset == Coco:
        dataset = dataset(data_path="./tmp_data/mscoco/val_lmdb/")
    else:
        dataset = dataset(data_path="./tmp_data/voc/test_lmdb")
    batch_transform_dataset = BatchTransformDataset(
        dataset,
        [
            [
                ToTensor(to_yuv=True),
            ],
            [
                ToTensor(to_yuv=True),
                TorchVisionAdapter(
                    interface="Normalize",
                    mean=128.0,
                    std=128.0,
                ),
            ],
        ],
        [epoch_steps],
    )
    assert len(batch_transform_dataset) == 5000 if dataset == Coco else 16551
    for ind, data in enumerate(batch_transform_dataset):
        img, gt_bboxes, gt_classes = (
            data["img"],
            data["gt_bboxes"],
            data["gt_classes"],
        )
        print(img.shape, gt_bboxes.shape, gt_classes.shape)
        if ind > 10:
            break
    assert len(batch_transform_dataset.current_transforms) == 1
    batch_transform_dataset.set_epoch(epoch_steps)
    assert len(batch_transform_dataset.current_transforms) == 2


if __name__ == "__main__":
    pytest.main(["-s", __file__])
