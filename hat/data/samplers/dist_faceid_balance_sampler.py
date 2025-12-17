import copy
import logging
import math
from typing import Iterator, List

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)

__all__ = ["DistributedFaceIDBalanceSampler"]


@OBJECT_REGISTRY.register
class DistributedFaceIDBalanceSampler(DistributedSampler):
    """In one epoch period, do balance faceid sampling.

    For more details, refer to wiki.
    http://wiki.hobot.cc/display/~han.tang/dist_faceid_balancesampler

    Args:
        dataset : faceid dataset
        bounds : balance sample bound.
        replace : replace replica sample.
        num_replicas : same as DistributedSampler
        rank : Same as DistributedSampler
        shuffle : if shuffle data
        seed : random seed
    """

    def __init__(
        self,
        dataset: Dataset,
        bounds: List[int],
        replace: bool = False,
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

        self.dataset = dataset
        self.replace = replace
        self.bounds = bounds

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.items_ttl = None
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self._reset_samples()

    def _select_id_range(self):
        selected_id_idx = copy.deepcopy(self.dataset.id_seq)
        return selected_id_idx

    def _reset_samples(self):
        selected_ids = self._select_id_range()
        if self.shuffle:
            self.rng.shuffle(selected_ids)
        nlower, nthresh, nhigher = self.bounds
        items_ttl = []
        for id_idx in selected_ids:
            nsamples = self.dataset.id_num[id_idx]
            n2sample = nlower if nsamples <= nthresh else nhigher
            cur_id_imgidx = list(range(*self.dataset.imgid2range[id_idx]))
            if nsamples < n2sample:
                items_ttl += cur_id_imgidx
                if self.replace:
                    items_ttl += list(
                        self.rng.choice(cur_id_imgidx, n2sample - nsamples)
                    )
            else:
                items_ttl += list(
                    self.rng.choice(cur_id_imgidx, n2sample, replace=False)
                )
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
            f"Reset samples fin,"
            f" total size: {self.total_size}"
            f" num samples: {self.num_samples}"
            f" num classes: {len(selected_ids)}"
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
