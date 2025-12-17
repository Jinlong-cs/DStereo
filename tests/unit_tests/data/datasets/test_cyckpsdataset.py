import os

import pytest

from hat.data.datasets.cyckps_dataset import CycKpsDataset
from tests import MONO_BUCKET_PATH

MONO_BUCKET_EXISTS = os.path.exists(MONO_BUCKET_PATH)

img_rec = [
    "data/lele.liu/kps/cyc_kps/mini/data.train.pb_rec",
]
anno_rec = [
    "data/lele.liu/kps/cyc_kps/mini/label.train.pb_rec",
]


@pytest.mark.skipif(not MONO_BUCKET_EXISTS, reason="need mono bucket")
def test_cyckps_dataset():
    img_rec_path = list(
        map(lambda p: os.path.join(MONO_BUCKET_PATH, p), img_rec)
    )
    anno_rec_path = list(
        map(lambda p: os.path.join(MONO_BUCKET_PATH, p), anno_rec)
    )
    dataset = CycKpsDataset(img_rec_path, anno_rec_path)
    test_nums = min(len(dataset), max(1, int(len(dataset) * 0.1)))
    for i in range(test_nums):
        data = dataset[i]
        assert "img" in data
        assert "boxes" in data
        assert "keypoints" in data
