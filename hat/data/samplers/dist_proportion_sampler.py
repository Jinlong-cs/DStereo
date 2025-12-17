import logging
import math
from typing import Iterator, List, Mapping, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)

__all__ = [
    "DistributedProportionSampler",
]


@OBJECT_REGISTRY.register
class DistributedProportionSampler(DistributedSampler):
    """Sample instance according to a given data distribution in each epoch.

    Args:
        dataset: Dataset used for sampling.
        expect_distribution: Sample distribution,
            e.g. {label_1: proportion, ...},
            note: the key of expect_distribution must
            correspond to the value in flag of given dataset.
        task_name: Task name, usually used for multitask.
        num_reference: Reference number of total size.
        num_replicas: Number of processes participating in distributed
            training. By default, :attr:``world_size`` is retrieved from the
            current distributed group.
        rank: Rank of the current process within :attr:``num_replicas``.
            By default, :attr:``rank`` is retrieved from the
            current distributed group.
        drop_last: If ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``True``.
        seed: Random seed.
    """

    def __init__(
        self,
        dataset: Dataset,
        expect_distribution: Mapping[str, float],
        task_name: Optional[str] = None,
        num_reference: Optional[str] = None,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        drop_last: bool = True,
        seed: int = 0,
        **kwargs,
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

        if not hasattr(dataset, "flag"):
            raise AttributeError("the dataset must contain attribute `flag`")
        self.flag = dataset.flag
        self.task_name = task_name
        if not task_name:
            self.task_name = "current task"

        assert (
            sum(list(expect_distribution.values())) == 1
        ), "The probabilities of each category must sum to one!"
        self.expect_distribution = expect_distribution
        self.num_replicas = num_replicas
        self.num_reference = num_reference
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.seed = seed

        self._init_helper()
        # sample the index of each category based on expect_distribution
        self._sample_index()

    def _init_helper(self):
        # split the index of each category
        self._split_ind_of_each_category()
        # calc the total size
        self._calc_total_size()

    def _split_ind_of_each_category(self):
        # _ind_info: {'0': ind0_list, '1': ind1_list1, ...}
        self._ind_info = {}
        for cls, _ in self.expect_distribution.items():
            cls_ind = np.where(self.flag == int(float(cls)))[0]
            assert len(cls_ind) != 0, (
                f"For {self.task_name},"
                f"there is no data of category {cls} in the training set,"
                "please check it and adjust the value of expect_distribution!"
            )
            self._ind_info[cls] = cls_ind

    def _calc_total_size(self):
        if self.num_reference is None:
            total_num = 0
            for cls, ratio in self.expect_distribution.items():
                tmp_num = int(len(self._ind_info[cls]) / ratio)
                if tmp_num > total_num:
                    total_num = tmp_num
        else:
            total_num = self.num_reference

        # save num of each category in expect_distribution
        cum_num = 0
        cls_keys = list(self.expect_distribution.keys())
        for i, cls in enumerate(cls_keys):
            if i == len(cls_keys) - 1:
                self.expect_distribution[cls] = int(total_num - cum_num)
            else:
                cur_num = int(self.expect_distribution[cls] * total_num)
                self.expect_distribution[cls] = cur_num
                cum_num += cur_num

        # If the dataset length is evenly divisible by replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and total_num % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (total_num - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(total_num / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def _sample_index(self):
        self.items_ttl = []
        s = f"Total size of instances in {self.task_name}: {self.total_size}\n"
        rng = np.random.RandomState(self.seed + self.epoch)
        for cls, sample_num in self.expect_distribution.items():
            ind = self._ind_info[cls]
            rng.shuffle(ind)
            if sample_num > len(ind):
                self.items_ttl += list(ind)
                self.items_ttl += list(rng.choice(ind, sample_num - len(ind)))
            else:
                self.items_ttl += list(ind[:sample_num])
        logger.info(s)

    def __iter__(self) -> Iterator[List[int]]:
        if self.epoch > 0:
            self._sample_index()
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.items_ttl), generator=g).tolist()

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
