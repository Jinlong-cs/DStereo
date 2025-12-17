import os

import pytest
import torch

from hat.engine.ddp_trainer import launch
from hat.registry import RegistryContext, build_from_registry
from hat.utils.seed import seed_everything

dataset = dict(
    type="ConcatDataset",
    datasets=[
        dict(
            type="SimpleDataset",
            length=8,
            start=0,
        ),
        dict(
            type="SimpleDataset",
            length=8,
            start=10,
        ),
    ],
)


data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dataset,
    sampler=dict(type="DistConcatSampler"),
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=False,
    drop_last=True,
)


def data_func(local_rank):
    with RegistryContext():
        dataloader = build_from_registry(data_loader)

    for i, data in enumerate(dataloader):
        if i < 4:
            # first dataset
            assert data < 10
        else:
            # second dataset
            assert data >= 10


@pytest.mark.serial_task
def test_dist_concat_sampler():
    seed_everything(os.getpid())
    launch(
        data_func,
        device_ids=[0, 1],
        args=(),
    )


if __name__ == "__main__":
    pytest.main(["-s", __file__])
