import math
from typing import Optional

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from hat.registry import OBJECT_REGISTRY
from hat.utils.global_var import get_value

__all__ = ["StatefulDistributedSampler"]


@OBJECT_REGISTRY.register
class StatefulDistributedSampler(DistributedSampler):
    """stateful distributed sampler.

    Support resume the data sampler from the training iteration.

    Args:
        dataset: Pytorch Dataset used for sampling.
        num_replicas: Number of processes participating in distributed training
        rank: Rank of the current process.
        shuffle: If ``True`` (default), sampler will shuffle the indices.
        seed: random seed used to shuffle the sampler if `shuffle=True`.
        drop_last: if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
        batch_size: How many samples per batch to load.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        batch_size: Optional[int] = None,
    ):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.start_iter = 0
        self.batch_size = batch_size

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        # resume the sampler
        if (
            get_value("dataloader_start_iter") is not None
            and get_value("dataloader_start_iter") > 0
        ):
            if self.batch_size is None:
                assert get_value("dataloader_batch_size") is not None
                assert (
                    get_value("dataloader_batch_size") > 0
                ), "batch_size not set for the sampler"
                batch_size = get_value("dataloader_batch_size")
            else:
                batch_size = self.batch_size
            start_index = get_value("dataloader_start_iter") * batch_size
            indices = indices[start_index:]
        return iter(indices)
