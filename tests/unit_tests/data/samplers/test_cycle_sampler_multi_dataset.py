import math

import pytest

from hat.data.datasets.dataset_wrappers import ComposeDataset
from hat.registry import build_from_registry


def test_DistributedCycleMultiDatasetSampler():
    length_list = [100, 200]
    dataset_a = dict(
        type="RandDataset",
        length=length_list[0],
        example=0,
    )
    dataset_b = dict(
        type="RandDataset",
        length=length_list[0],
        example=1,
    )
    datasets = [dataset_a, dataset_b]
    batchsize_list = [2, 3]
    compose_dataset = ComposeDataset(
        datasets=datasets, batchsize_list=batchsize_list
    )

    sampler = dict(
        type="DistributedCycleMultiDatasetSampler",
        batchsize_list=batchsize_list,
        dataset=compose_dataset,
    )

    # TODO (enci.zhou, 0.3): develop ut #
    with pytest.raises(RuntimeError):
        _ = build_from_registry(sampler)
        max_iter_time = max(
            [
                math.ceil(length // bs)
                for length, bs in zip(length_list, batchsize_list)
            ]
        )
        sampler_iter = iter(sampler)
        for _ in range(max_iter_time):
            _ = sampler_iter.__next__()
