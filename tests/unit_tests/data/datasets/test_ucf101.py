import math

import pytest
import torchvision
from torch.utils.data import DataLoader

try:
    import av  # noqa: F401
except ImportError:
    AV_AVAILABLE = False
else:
    AV_AVAILABLE = True
from hat.data.datasets.ucf101 import UCF101
from hat.data.transforms.video_classification import (
    JitterScaleVideo,
    NormalizeVideo,
    RandomCropVideo,
)

transforms = torchvision.transforms.Compose(
    [
        NormalizeVideo(
            mean=(0.45, 0.45, 0.45),
            std=(0.225, 0.225, 0.225),
            tensor_shape=(1, 1, 1, 3),
        ),
        JitterScaleVideo(min_size=256, max_size=256),
        RandomCropVideo(target_size=224),
    ]
)


@pytest.mark.skipif(not AV_AVAILABLE, reason="av is required")
@pytest.mark.parametrize(
    "transforms, batch_size", [(None, 1), (transforms, 50)]
)
def test_ucf101(transforms, batch_size):
    dataset = UCF101(
        root="./tmp_orig_data/ucf101",
        file_path="./tmp_orig_data/ucf101/ucf101_val_list.txt",
        mode="test",
        transforms=transforms,
    )
    print(len(dataset))

    for ind, data in enumerate(dataset):
        image, target = data["img"], data["labels"]
        print(image.shape, target)
        if ind > 10:
            break

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    assert len(dataloader) == int(math.ceil(len(dataset) / batch_size))
    for ind, batch in enumerate(dataloader):
        image, target = batch["img"], batch["labels"]
        assert image.shape[0] == batch_size
        assert target.shape[0] == batch_size
        print(image.shape, target)
        if ind > 10:
            break
