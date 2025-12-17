import math
import os

import pytest
import torchvision
from torch.utils.data import DataLoader

from hat.data.collates.collates import collate_2d
from hat.data.datasets.hpp_dataset import HppDataset
from hat.data.transforms.detection import Normalize, ToTensor
from tests import MONO_BUCKET_PATH

MONO_BUCKET_EXISTS = os.path.exists(MONO_BUCKET_PATH)

transforms = torchvision.transforms.Compose(
    [ToTensor(to_yuv=True), Normalize(mean=128.0, std=128.0)]
)

img_path = "data/yingqian.zhao/HPP/data/train_data/virtual_10_70_distcoeffs/rec/trajs_10_70m_wide20_3721.rec"  # noqa
anno_path = "data/yingqian.zhao/HPP/data/train_data/virtual_10_70_distcoeffs/hpp_json/trajs_10_70m_wide20_3721.json"  # noqa


@pytest.mark.skipif(not MONO_BUCKET_EXISTS, reason="need mono bucket")
@pytest.mark.parametrize(
    ["transforms", "batchsize", "read_mode"],
    [pytest.param(transforms, 2, "rec")],
)
def test_hpp_dataset(transforms, batchsize, read_mode):
    dataset = HppDataset(
        img_path=os.path.join(MONO_BUCKET_PATH, img_path),
        anno_path=os.path.join(MONO_BUCKET_PATH, anno_path),
        read_mode=read_mode,
    )
    data = dataset[0]
    assert "img" in data
    assert "img_path" in data
    assert "points" in data

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_2d,
    )

    assert len(dataloader) == math.ceil(len(dataset) / batchsize)
    for ind, batch in enumerate(dataloader):
        image, gt_points = batch["img"], batch["points"]
        assert image.shape[0] == batchsize
        assert len(gt_points) == batchsize
        print(image.shape, len(gt_points))
        if ind > 10:
            break


if __name__ == "__main__":
    pytest.main(["-s", __file__])
