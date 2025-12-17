import torch

from hat.registry import build_from_registry


def test_DistSamplerHook():
    dataset = dict(
        type="RandDataset",
        length=100,
        example=0,
    )

    # test with no rank set
    sampler = dict(
        type=torch.utils.data.DistributedSampler,
        dataset=dataset,
    )

    sampler = build_from_registry(sampler)
    assert sampler.num_replicas == 1
    assert sampler.rank == 0

    # test with setting rank
    sampler = dict(
        type=torch.utils.data.DistributedSampler,
        dataset=dataset,
        rank=1,
        num_replicas=2,
    )

    sampler = build_from_registry(sampler)
    assert sampler.num_replicas == 2
    assert sampler.rank == 1
