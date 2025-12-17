import numpy as np

from hat.data.datasets.rand_dataset import RandDataset
from hat.data.samplers.dist_group_sampler import DistributedGroupSampler


def test_DistributedGroupSampler():
    length = 100
    batch = 8
    dataset = RandDataset(
        length=length,
        example=1,
    )
    dataset.flag = np.zeros(len(dataset), dtype=np.uint8)
    for ind in range(0, len(dataset), 2):
        dataset.flag[ind] = 1

    # test with no rank set
    sampler = DistributedGroupSampler(
        dataset=dataset,
        samples_per_gpu=batch,
    )

    assert sampler.num_replicas == 1
    assert sampler.rank == 0

    num_samples = 0
    for size in sampler.group_sizes:
        num_samples += (
            int(np.math.ceil(size * 1.0 / batch / sampler.num_replicas))
            * batch
        )

    assert len(sampler) == num_samples

    sampler_iter = iter(sampler)
    for _ in range(len(sampler) // batch):
        group_flags = []
        for _ in range(batch):
            ind = sampler_iter.__next__()
            flag = dataset.flag[ind]
            group_flags.append(flag)
        assert len(np.unique(group_flags)) == 1

    # test with setting rank
    sampler = DistributedGroupSampler(
        dataset=dataset,
        samples_per_gpu=batch,
        rank=1,
        num_replicas=2,
    )

    assert sampler.num_replicas == 2
    assert sampler.rank == 1

    num_samples = 0
    for size in sampler.group_sizes:
        num_samples += (
            int(np.math.ceil(size * 1.0 / batch / sampler.num_replicas))
            * batch
        )

    assert len(sampler) == num_samples

    sampler_iter = iter(sampler)
    for _ in range(len(sampler) // batch // sampler.num_replicas):
        group_flags = []
        for _ in range(batch):
            ind = sampler_iter.__next__()
            flag = dataset.flag[ind]
            group_flags.append(flag)
        assert len(np.unique(group_flags)) == 1
