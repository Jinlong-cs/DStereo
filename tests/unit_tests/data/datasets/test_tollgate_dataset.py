import math
import os

import pytest
from torch.utils.data import DataLoader

from hat.data.collates.collates import collate_2d
from hat.data.datasets.toll_kps import TollgateDataset
from tests import MONO_BUCKET_PATH

MONO_BUCKET_EXISTS = os.path.exists(MONO_BUCKET_PATH)

train_json = "ziyang02.wang/dataset/rec/Toll_pole/super_drive/202303/202303_v1.json"  # noqa
train_rec = "ziyang02.wang/dataset/rec/Toll_pole/super_drive/202303/202303_v1.rec"  # noqa


@pytest.mark.skipif(not MONO_BUCKET_EXISTS, reason="need mono bucket")
def test_tollgate_dataset(batchsize=2):
    dataset = TollgateDataset(
        rec_path=os.path.join(MONO_BUCKET_PATH, train_rec),
        anno_path=os.path.join(MONO_BUCKET_PATH, train_json),
    )
    data = dataset[0]
    tags = [
        "img_name",
        "gt_lines",
        "img_height",
        "img_width",
        "img_id",
        "img",
        "color_space",
        "layout",
        "img_shape",
    ]
    for tag in tags:
        assert tag in data
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
        image, gt_lines = batch["img"], batch["gt_lines"]
        assert image.shape[0] == batchsize
        assert len(gt_lines) == batchsize
        if ind > 10:
            break


if __name__ == "__main__":
    pytest.main(["-s", __file__])
