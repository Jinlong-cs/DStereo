from hat.data.datasets.rand_dataset import RandDataset
from hat.data.samplers.dist_set_epoch_dataset_sampler import (
    DistSetEpochDatasetSampler,
)


def test_DistSetEpochDatasetSampler():
    length = 100
    dataset = RandDataset(
        length=length,
        example=1,
    )
    sampler = DistSetEpochDatasetSampler(
        dataset=dataset,
        rank=0,
        num_replicas=1,
    )
    sampler.set_epoch(100)

    assert dataset.epoch == 100
