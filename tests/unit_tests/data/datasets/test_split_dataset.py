import os

import pytest

from hat.engine.ddp_trainer import launch
from hat.registry import RegistryContext, build_from_registry
from hat.utils.global_var import set_value
from hat.utils.seed import seed_everything

dataset = dict(
    type="RankSplitDataset",
    datasets=[
        dict(
            type="SimpleDataset",
            length=4,
            flag=1,
            start=0,
            __lazy_build__=True,
        ),
        dict(
            type="SimpleDataset",
            length=5,
            flag=2,
            start=4,
            __lazy_build__=True,
        ),
    ],
)

data_loader = dict(
    type="RankSplitDataLoader",
    dataset=dataset,
    # use StatefulDistributedSampler to support resume
    sampler=dict(type="StatefulDistributedSampler", drop_last=True),
    batch_size=1,
    shuffle=True,
    num_workers=2,
    pin_memory=False,
    drop_last=True,
)


def data_func(local_rank):
    with RegistryContext():
        dataloader = build_from_registry(data_loader)

    data_list = []
    for _, data in enumerate(dataloader):  # noqa
        data_list.append(data)

        pass

    # test resume
    set_value("dataloader_batch_size", dataloader.batch_size)
    set_value("dataloader_start_iter", 2)
    resumed_data_list = []
    for _, data in enumerate(dataloader):
        resumed_data_list.append(data)

    assert resumed_data_list == data_list[2:]


@pytest.mark.serial_task
def test_worker_split_dataset():
    seed_everything(os.getpid())
    launch(
        data_func,
        device_ids=[0, 1],
        args=(),
    )


if __name__ == "__main__":
    pytest.main(["-s", __file__])
