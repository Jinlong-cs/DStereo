import math
import os

import pytest
import torchvision
from torch.utils.data import DataLoader

from hat.data.collates.collates import collate_mot_seq
from hat.data.datasets.mot17_dataset import (
    Mot17Dataset,
    Mot17FromImage,
    Mot17Packer,
)
from hat.data.transforms.seq_transform import (
    SeqBgrToYuv444,
    SeqNormalize,
    SeqRandomFlip,
    SeqRandomSizeCrop,
    SeqResize,
    SeqToTensor,
)

transforms = torchvision.transforms.Compose(
    [
        SeqRandomFlip(
            px=0.5,
            py=0,
        ),
        SeqResize(
            img_scale=[
                (400, 9999999),
                (500, 9999999),
                (600, 9999999),
            ],
            multiscale_mode="value",
            keep_ratio=True,
            rm_neg_coords=False,
            divisor=2,
        ),
        SeqRandomSizeCrop(
            min_size=384,
            max_size=600,
            filter_area=False,
            rm_neg_coords=False,
        ),
        SeqResize(
            img_scale=[
                (608, 1536),
                (640, 1536),
                (672, 1536),
                (704, 1536),
                (736, 1536),
                (768, 1536),
                (800, 1536),
                (832, 1536),
                (864, 1536),
                (896, 1536),
                (928, 1536),
                (960, 1536),
                (992, 1536),
            ],
            multiscale_mode="value",
            keep_ratio=True,
            rm_neg_coords=False,
            divisor=2,
        ),
        SeqToTensor(to_yuv=False),
        SeqBgrToYuv444(rgb_input=True),
        SeqNormalize(mean=128.0, std=128.0),
    ]
)


@pytest.mark.parametrize(
    "transforms, batch_size, pack_type",
    [(transforms, 1, "lmdb"), (None, 4, None)],
)
def test_mot17_dataset(transforms, batch_size, pack_type):
    dataset = Mot17Dataset(
        pack_type=pack_type,
        data_path="./tmp_data/mot17/test_lmdb",
        transforms=transforms,
    )

    assert len(dataset) == 2659

    for ind, data in enumerate(dataset):
        image = data["frame_data_list"][0]["img"]
        print(image.shape)
        if ind > 10:
            break

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_mot_seq,
    )

    assert len(dataloader) == math.ceil(len(dataset) / batch_size)
    for ind, batch in enumerate(dataloader):
        assert len(batch["frame_data_list"]) == batch_size
        if ind > 10:
            break


data_path = os.path.join("./tmp_orig_data/mot17/split_data")

PATH_EXIST = False
if os.path.exists(data_path):
    PATH_EXIST = True


@pytest.mark.skipif(not PATH_EXIST, reason="path does not exist")
def test_mot17_fromimage():
    dataset_train = Mot17FromImage(os.path.join(data_path, "train"))
    for ind, data in enumerate(dataset_train):
        image = data["frame_data_list"][0]["img"]
        print(image.shape)
        if ind > 10:
            break
    assert len(dataset_train) == 2657

    dataset_test = Mot17FromImage(os.path.join(data_path, "test"))
    for ind, data in enumerate(dataset_test):
        image = data["frame_data_list"][0]["img"]
        print(image.shape)
        if ind > 10:
            break
    assert len(dataset_test) == 2659


@pytest.mark.skipif(not PATH_EXIST, reason="path does not exist")
def test_mot17dataset_packer(tmpdir):
    tmp_dir = os.path.join(tmpdir, "Mot17")

    packer = Mot17Packer(data_path, tmp_dir, "test", 1, "lmdb", num_samples=50)
    packer()
    assert os.path.exists(os.path.join(tmp_dir, "data.mdb"))


if __name__ == "__main__":
    pytest.main(["-s", __file__])
