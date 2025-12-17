# Copyright (c) Horizon Robotics. All rights reserved.

import pytest
import torchvision
from torch.utils.data import DataLoader

from hat.data.datasets.heartrate_dataset import HeartRateDataset

transforms = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()]
)
test_list = [
    "p89",
    "p39",
    "p3",
    "p60",
    "p18",
    "p14",
    "p69",
    "p19",
    "p93",
    "p16",
    "p86",
    "p4",
    "p52",
    "p11",
    "p33",
    "p42",
    "p96",
    "p30",
    "p54",
    "p36",
    "p80",
    "p32",
]


@pytest.mark.parametrize("transforms, batch_size", [(transforms, 1)])
def test_heartrate_dataset(transforms, batch_size):
    dataset = HeartRateDataset(
        st_maps_path="./tmp_orig_data/vipl_MSTmaps_RGB_psv/",
        transform=transforms,
        test_list=test_list,
    )
    assert len(dataset) == 1
    for _, batch in enumerate(dataset):
        st_maps = batch["st_maps"]
        fps = batch["fps"]
        heart_beats_num = batch["heart_beats_num"]
        gt_HR = batch["gt_HR"]
        break

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    assert int(len(dataloader)) == round(len(dataset) / batch_size)
    for idx, batch in enumerate(dataloader):
        print(idx)
        st_maps = batch["st_maps"]
        fps = batch["fps"]
        heart_beats_num = batch["heart_beats_num"]
        gt_HR = batch["gt_HR"]
        assert st_maps.size()[0] == batch_size
        assert st_maps.size()[1] == 3
        assert st_maps.size()[2] == 63
        assert st_maps.size()[3] == 300
        assert fps.size()[0] == batch_size
        assert heart_beats_num.size()[0] == batch_size
        assert gt_HR.size()[0] == batch_size

        break


if __name__ == "__main__":
    pytest.main(["-s", __file__])
