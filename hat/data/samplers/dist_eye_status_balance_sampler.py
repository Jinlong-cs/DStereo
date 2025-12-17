import logging
import math
from typing import Iterator, List

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from hat.data.datasets.eye_status_dataset import EyeStatusDataset
from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)

__all__ = ["DistributedEyeStatusBalanceSampler"]


@OBJECT_REGISTRY.register
class DistributedEyeStatusBalanceSampler(DistributedSampler):
    """In one epoch period, do balance eye status sampling.

    Args:
        dataset : eye status dataset
        num_replicas : same as DistributedSampler
        rank : Same as DistributedSampler
        shuffle : if shuffle data
        drop_last: If ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``True``.
        seed : random seed
    """

    def __init__(
        self,
        dataset: EyeStatusDataset,
        num_replicas: int = None,
        rank: int = None,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available"
                )
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available"
                )
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )

        self._check_datasets(dataset)
        self.dataset = dataset

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.items_ttl = None
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self._reset_samples()

    def _check_datasets(self, dataset):
        assert isinstance(
            dataset, EyeStatusDataset
        ), "dataset must be EyeStatusDataset Instance"

        assert (
            len(dataset.datasets) > 0
        ), f"datasets num {len(self.dataset.datasets)} error"

    def _reset_samples(self):
        items_ttl = np.zeros(shape=0, dtype=np.int32)
        dataset_offset = 0
        for dataset in self.dataset.datasets:
            id_keys = list(dataset.id2range.keys())
            items_ttl_ = []
            for dst_id in id_keys:
                id_range = dataset.id2range[dst_id]
                aug_num = dataset.id_sample_num[dst_id]
                raw_num = id_range[1] - id_range[0]
                id_items_ttl = list(range(*id_range))
                if aug_num == 0:
                    continue

                raw_times = aug_num // raw_num
                residual = aug_num % raw_num
                res_list = []
                if residual > 0:
                    res_list = self.rng.choice(
                        id_items_ttl,
                        residual,
                        replace=False,
                    )
                id_items_ttl *= raw_times
                id_items_ttl += list(res_list)
                items_ttl_ += id_items_ttl
            items_ttl_ = np.array(items_ttl_).astype(np.int32)
            items_ttl_ += dataset_offset
            items_ttl = np.concatenate([items_ttl, items_ttl_])
            dataset_offset += len(dataset)

        self.items_ttl = items_ttl
        if self.drop_last and len(self.items_ttl) % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (len(self.items_ttl) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(
                len(self.items_ttl) / self.num_replicas
            )
        self.total_size = self.num_samples * self.num_replicas
        logger.info(
            f"Reset samples fin, "
            f"total size: {self.total_size} "
            f"num samples: {self.num_samples}"
        )

    def __iter__(self) -> Iterator[List[int]]:
        # subsample
        if self.epoch > 0:
            self._reset_samples()

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.items_ttl), generator=g).tolist()
        else:
            indices = list(range(len(self.items_ttl)))

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

        indices = np.array(self.items_ttl)[indices]

        return iter(indices)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples
