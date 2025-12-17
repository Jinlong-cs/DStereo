import os

import pytest
from torch.utils.data import DataLoader

from hat.data.samplers.dist_bucket_sampler import (
    DistOrderBucketBatchSampler,
    DistOrderBucketSampler,
    DistRandomBucketBatchSampler,
    DistRandomBucketSampler,
)
from hat.engine.ddp_trainer import launch
from hat.registry import RegistryContext, build_from_registry
from hat.utils.seed import seed_everything

dataset_config = dict(
    type="SimpleDataset",
    length=100,
    start=0,
)


def data_func(local_rank):
    with RegistryContext():
        dataset = build_from_registry(dataset_config)
    dataset.info_list = [{"length": i + 1} for i in range(len(dataset))]

    def is_overlap(list, sort=True):
        sorted_list = sorted(list, key=lambda x: x[0])
        for i in range(len(sorted_list) - 1):
            if sorted_list[i][1] > sorted_list[i + 1][0]:
                return True

    # test DistOrderBucketSampler
    dataloader = DataLoader(
        batch_size=10,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        dataset=dataset,
        sampler=DistOrderBucketSampler(
            dataset=dataset,
            shuffle=True,
            bucket_size=10,
        ),
    )

    last_max = -1
    for batch in dataloader:
        assert min(batch) >= last_max, f"minimum of {batch} < {last_max}"
        last_max = max(batch)

    # test DistRandomBucketSampler
    dataloader = DataLoader(
        batch_size=10,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        dataset=dataset,
        sampler=DistRandomBucketSampler(
            dataset=dataset,
            shuffle=True,
            bucket_size=10,
            batch_size=10,
        ),
    )
    length_list = []
    for batch in dataloader:
        length_list.append((min(batch).item(), max(batch).item()))
    assert not is_overlap(length_list), f"overlap in {length_list}"

    # test DistRandomBucketBatchSampler
    dataloader = DataLoader(
        num_workers=0,
        pin_memory=False,
        dataset=dataset,
        batch_sampler=DistRandomBucketBatchSampler(
            dataset=dataset,
            max_times_in_batch=100,
            shuffle=True,
            bucket_size=10,
            drop_last=False,
        ),
    )
    for batch in dataloader:
        assert sum(batch) <= 100, f"length of {batch} > 50"

    # test DistOrderBucketBatchSampler
    dataloader = DataLoader(
        num_workers=0,
        pin_memory=False,
        dataset=dataset,
        batch_sampler=DistOrderBucketBatchSampler(
            dataset=dataset,
            max_times_in_batch=100,
            sampler=DistOrderBucketSampler(
                dataset=dataset,
                shuffle=True,
                drop_last=False,
                bucket_size=10,
            ),
        ),
    )
    for batch in dataloader:
        assert sum(batch) <= 100, f"length of {batch} > 50"


@pytest.mark.serial_task
def test_dist_bucket_sampler():
    seed_everything(os.getpid())
    launch(
        data_func,
        device_ids=[0, 1],
        args=(),
    )


if __name__ == "__main__":
    pytest.main(["-s", __file__])
