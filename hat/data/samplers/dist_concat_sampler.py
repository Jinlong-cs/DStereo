# Copyright (c) Horizon Robotics. All rights reserved.
import math
from typing import Optional

import torch
from torch.utils.data.distributed import DistributedSampler

from hat.data.datasets.dataset_wrappers import ConcatDataset
from hat.registry import OBJECT_REGISTRY
from hat.utils.distributed import get_dist_info, get_local_process_group

__all__ = ["DistConcatSampler"]


@OBJECT_REGISTRY.register
class DistConcatSampler(DistributedSampler):  # noqa: D205,D400
    """
    The hook API is specifically distributed sampler for concat dataset.
    The results of the sampler will ensure that the same time period is
    only performed on the same subdataset.

    Args:
        dataset: compose dataset
        num_replicas: same as DistributedSampler
        rank: Same as DistributedSampler
        shuffle: if shuffle data
        seed: random seed
        drop_last: if drop last
    """

    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if num_replicas is None or rank is None:
            rank, num_replicas = get_dist_info(get_local_process_group())

        assert isinstance(dataset, ConcatDataset)
        super(DistConcatSampler, self).__init__(
            dataset, num_replicas, rank, shuffle, seed, drop_last
        )  # noqa

        self.total_size = []
        for dataset in self.dataset.datasets:
            if self.drop_last and len(dataset) % self.num_replicas != 0:
                num_samples = math.ceil(
                    (len(dataset) - self.num_replicas) / self.num_replicas
                )
            else:
                num_samples = math.ceil(len(dataset) / self.num_replicas)
            self.total_size.append(num_samples * self.num_replicas)

    def __iter__(self):
        all_indices = []
        for i, each_len in enumerate(self.dataset.len_list):
            if self.shuffle:
                # deterministically shuffle based on epoch and seed
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                indices = torch.randperm(each_len, generator=g).tolist()
            else:
                indices = list(range(each_len))

            if not self.drop_last:
                # add extra samples to make it evenly divisible
                padding_size = self.total_size[i] - len(indices)
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    indices += (
                        indices * math.ceil(padding_size / len(indices))
                    )[:padding_size]
            else:
                # remove tail of data to make it evenly divisible.
                indices = indices[:each_len]

            # subsample
            indices = indices[self.rank : each_len : self.num_replicas]
            if i > 0:
                indices = [
                    ind + self.dataset.cumulative_sizes[i - 1]
                    for ind in indices
                ]

            all_indices += indices
        return iter(all_indices)
