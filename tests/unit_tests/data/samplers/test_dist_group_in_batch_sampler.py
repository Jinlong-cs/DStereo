import random

import numpy as np
import pytest

from hat.data.datasets.rand_dataset import RandDataset
from hat.data.samplers.dist_group_in_batch_sampler import (
    DistributedGroupInBatchSampler,
)


@pytest.mark.parametrize(
    "batch_size, groups",
    [(4, 10), (5, 10), (6, 10), (20, 20)],
)
def test_dist_group_in_batch_sampler(batch_size, groups):
    flag = []
    random.seed(batch_size + groups)
    for i in range(groups):
        flag.extend([i] * (random.choice(list(range(100))) + 1))
    length = len(flag)
    dataset = RandDataset(
        length=length,
        example=1,
    )
    dataset.flag = np.array(flag, dtype=np.int64)
    sampler = DistributedGroupInBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        world_size=1,
        rank=0,
    )
    sampler_iter = iter(sampler)
    batch_index = [None] * length
    while None in batch_index:
        batch = next(sampler_iter)
        for i, data_index in enumerate(batch):
            if batch_index[data_index] is None:
                batch_index[data_index] = i

    group_batch_index_map = {}
    for f, index in zip(flag, batch_index):
        if f not in group_batch_index_map:
            group_batch_index_map[f] = index
        else:
            assert group_batch_index_map[f] == index
