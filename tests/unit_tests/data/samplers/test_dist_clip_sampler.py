from hat.data.datasets.dataset_wrappers import ComposeDataset, ResampleDataset
from hat.data.datasets.rand_dataset import RandDataset
from hat.data.samplers.dist_clip_group_sampler import (
    ANCDistributedClipValSampler,
    ANCDistributedGroupClipSampler,
)


def test_DistributedClipSampler():
    length = 100
    batch = 2
    dataset = RandDataset(
        length=length,
        example=1,
    )
    # test with no rank set
    sampler = ANCDistributedGroupClipSampler(
        dataset=dataset,
        samples_per_gpu=batch,
        sub_clip_num=2,
    )

    assert sampler.num_replicas == 1
    assert sampler.rank == 0

    sampler_iter = iter(sampler)
    his_indices_list = None
    for i in range(len(sampler) // batch):
        indices_list = []
        for _ in range(batch):
            indices = sampler_iter.__next__()
            indices_list.append(indices)
        if i % 2 == 1:
            assert (indices_list[0] - his_indices_list[0]) == 1
            assert (indices_list[1] - his_indices_list[1]) == 1
        his_indices_list = indices_list

    # test with no rank set
    sampler = ANCDistributedGroupClipSampler(
        dataset=dataset,
        samples_per_gpu=batch,
        sub_clip_num=2,
        rank=1,
        num_replicas=2,
    )
    assert sampler.num_replicas == 2
    assert sampler.rank == 1

    sampler_iter = iter(sampler)
    his_indices_list = None
    for i in range(len(sampler) // batch):
        indices_list = []
        for _ in range(batch):
            indices = sampler_iter.__next__()
            indices_list.append(indices)
        if i % 2 == 1:
            assert (indices_list[0] - his_indices_list[0]) == 1
            assert (indices_list[1] - his_indices_list[1]) == 1
        his_indices_list = indices_list


def test_DistributedClipValSampler():

    length = [100, 200]
    dataset_a = RandDataset(
        length=length[0],
        example=1,
    )

    dataset_a.sync_info = [([{"pack_dir": "aa/aa"}] * 2, 0)] * length[0]
    dataset_a = ResampleDataset(dataset_a)

    dataset_b = RandDataset(
        length=length[1],
        example=1,
    )

    dataset_b.sync_info = [([{"pack_dir": "bb/bb"}] * 2, 0)] * length[1]
    dataset_b = ResampleDataset(dataset_b)

    datasets = [dataset_a, dataset_b]
    compose_dataset = ComposeDataset(datasets=datasets, batchsize_list=[1, 1])

    sampler = ANCDistributedClipValSampler(
        compose_dataset, num_replicas=2, rank=1
    )

    assert len(sampler) == 200

    sampler_iter = iter(sampler)
    history_indice = None
    for _ in range(len(sampler)):
        indice = sampler_iter.__next__()
        if history_indice is not None:
            assert (indice - history_indice) == 1
        history_indice = indice
