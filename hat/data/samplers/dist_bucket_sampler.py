# Copyright (c) Horizon Robotics, All rights reserved.


import math
import random
from typing import Iterator, Optional, TypeVar

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from hat.data.datasets.avspeech_dataset import CocktailDatasetV0

T_co = TypeVar("T_co", covariant=True)


class DistOrderBucketSampler(DistributedSampler[T_co]):
    """Distributed sampler that buckets data of dataset based on sample length.

    The size of each bucket is the same, and the samples in each bucket are
    shuffled. The length of samples are increase or decrease at each epoch.

    Args:
        dataset: dataset to sample from
        num_replicas: number of processes participating in distributed training
        rank: rank of the current process
        shuffle: if ``True``, sampler will shuffle the indices
            deterministically at each bucket
        seed: random seed used to shuffle the indices
            deterministically at each epoch
        drop_last: if ``True``, sampler will drop the last
            batch if its size would be less than ``batch_size``
        bucket_size: bucket size
        reverse: if ``True``, sampler will reverse the indices
            deterministically at each epoch, default is ``False``, which
            means the the length will increase at each epoch.
    """

    def __init__(
        self,
        dataset: CocktailDatasetV0,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        bucket_size: int = 10000,
        reverse: bool = False,
    ) -> None:
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        self.bucket_size = bucket_size
        self.reverse = reverse

    def __iter__(self) -> Iterator[T_co]:
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

        # 如果长度小于bucket_size的大小直接返回
        if len(indices) < self.bucket_size:
            return iter(indices)

        # 进行 bucket 操作
        info_iter = ((idx, self.dataset.info_list[idx]) for idx in indices)
        info_list = sorted(info_iter, key=lambda info: info[1]["length"])
        indices = []
        for beg_idx in range(0, self.num_samples, self.bucket_size):
            end_idx = min(beg_idx + self.bucket_size, self.num_samples)
            bucket_indices = [idx for idx, _ in info_list[beg_idx:end_idx]]
            random.shuffle(bucket_indices)
            indices.extend(bucket_indices)

        if self.reverse:
            return reversed(indices)
        return iter(indices)


class DistRandomBucketSampler(DistributedSampler[T_co]):
    """Distributed sampler that buckets data of dataset based on sample length.

    The size of each bucket is the same, and the samples in each bucket are
    shuffled. The length of samples are shuffled at different batch. The length
    of samples are almost the same in each batch.

    Args:
        dataset: dataset to sample from
        batch_size: batch size
        num_replicas: number of processes participating in distributed training
        rank: rank of the current process
        shuffle: if ``True``, sampler will shuffle the indices
            deterministically at each bucket
        seed: random seed used to shuffle the indices
            deterministically at each epoch
        drop_last: if ``True``, sampler will drop the last
            batch if its size would be less than ``batch_size``
        bucket_size: bucket size
    """

    def __init__(
        self,
        dataset: CocktailDatasetV0,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        bucket_size: int = 10000,
    ) -> None:
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        self.bucket_size = bucket_size
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[T_co]:
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

        # 如果长度小于bucket_size的大小直接返回
        if len(indices) < self.bucket_size:
            return iter(indices)

        # 进行 bucket 操作
        info_iter = [(idx, self.dataset.info_list[idx]) for idx in indices]
        info_list = sorted(info_iter, key=lambda info: info[1]["length"])
        batches = []
        for beg_idx in range(0, self.num_samples, self.bucket_size):
            end_idx = min(beg_idx + self.bucket_size, self.num_samples)
            bucket_indices = [idx for idx, _ in info_list[beg_idx:end_idx]]
            random.shuffle(bucket_indices)
            for batch_beg in range(0, len(bucket_indices), self.batch_size):
                batch_end = min(
                    batch_beg + self.batch_size, len(bucket_indices)
                )
                batches.append(bucket_indices[batch_beg:batch_end])

        random.Random(777).shuffle(batches)
        # random.shuffle(batches)
        indices = []
        last_batch = []
        for batch in batches:
            if len(batch) == self.batch_size:
                indices.extend(batch)
            else:
                last_batch.extend(batch)
                if len(last_batch) >= self.batch_size:
                    indices.extend(last_batch[: self.batch_size])
                    last_batch = last_batch[self.batch_size :]
        indices.extend(last_batch)
        assert len(indices) == self.num_samples
        return iter(indices)


class DistOrderBucketBatchSampler(object):
    """Batch sampler of DistOrderBucketSampler.

    Make total length of samples in a batch keep almost the same.

    Args:
        dataset: dataset to sample from
        sampler: sampler to sample indices
        max_times_in_batch: max times of samples in a batch
    """

    def __init__(
        self,
        dataset: CocktailDatasetV0,
        sampler: DistOrderBucketSampler,
        max_times_in_batch: int,
    ) -> None:
        self.dataset = dataset
        self.sampler = sampler
        self.max_times_in_batch = max_times_in_batch
        self._gen_iter()

    def _gen_iter(self):
        batch = []
        mini_batch = []
        longest_time = 0
        for sample in self.sampler:
            new_sample_time = self.dataset.info_list[sample]["length"]
            longest_time = max(longest_time, new_sample_time)
            times_after_padding = longest_time * (len(mini_batch) + 1)
            if times_after_padding > self.max_times_in_batch:
                batch.append(mini_batch)
                mini_batch = []
                mini_batch.append(sample)
                longest_time = new_sample_time
            else:
                mini_batch.append(sample)
        if len(mini_batch) > 0:
            batch.append(mini_batch)

        length_list = [None for _ in range(dist.get_world_size())]
        length = len(batch)
        dist.all_gather_object(length_list, length)
        keep_length = min(length_list)
        remove_num = length - keep_length
        if remove_num != 0:
            items = random.sample(batch, remove_num)
            for item in items:
                batch.remove(item)
        self.batch = batch

    def __next__(self):
        try:
            return next(self.batch_iter)
        except StopIteration:
            raise StopIteration

    def __iter__(self):
        self.batch_iter = iter(self.batch)
        return self

    def __len__(self):
        return len(self.batch)


class DistRandomBucketBatchSampler(DistributedSampler[T_co]):
    """Distributed sampler that buckets data of dataset based on sample length.

    The size of each bucket is the same, and the samples in each bucket are
    shuffled. The average lengths of samples at different batch are random.
    The total length of samples are almost the same in each batch.

    Args:
        dataset: dataset to sample from
        max_times_in_batch: max times in second of total length of samples
            in a batch
        num_replicas: number of processes participating in distributed training
        rank: rank of the current process
        shuffle: if ``True``, sampler will shuffle the indices
        seed: random seed used to shuffle the indices
        drop_last: if ``True``, sampler will drop the last
            batch if its size would be less than ``batch_size``
        bucket_size: bucket size
    """

    def __init__(
        self,
        dataset: CocktailDatasetV0,
        max_times_in_batch: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        bucket_size: int = 10000,
    ) -> None:
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        self.bucket_size = bucket_size
        self.max_times_in_batch = max_times_in_batch
        self._gen_iter()

    def _gen_iter(self):
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

        # 进行 bucket 操作
        info_iter = [(idx, self.dataset.info_list[idx]) for idx in indices]
        info_list = sorted(info_iter, key=lambda info: info[1]["length"])
        batches = []
        last_samples = []
        for beg_idx in range(0, self.num_samples, self.bucket_size):
            end_idx = min(beg_idx + self.bucket_size, self.num_samples)
            bucket_indices = [idx for idx, _ in info_list[beg_idx:end_idx]]
            random.shuffle(bucket_indices)
            longest_time = 0
            mini_batch = []
            for sample in bucket_indices:
                new_sample_time = self.dataset.info_list[sample]["length"]
                longest_time = max(longest_time, new_sample_time)
                times_after_padding = longest_time * (len(mini_batch) + 1)
                if times_after_padding > self.max_times_in_batch:
                    batches.append(mini_batch)
                    mini_batch = []
                    mini_batch.append(sample)
                    longest_time = new_sample_time
                else:
                    mini_batch.append(sample)

            if len(mini_batch) > 0:
                last_samples.extend(mini_batch)

        longest_time = 0
        mini_batch = []
        for sample in last_samples:
            new_sample_time = self.dataset.info_list[sample]["length"]
            longest_time = max(longest_time, new_sample_time)
            times_after_padding = longest_time * (len(mini_batch) + 1)
            if times_after_padding > self.max_times_in_batch:
                batches.append(mini_batch)
                mini_batch = []
                mini_batch.append(sample)
                longest_time = new_sample_time
            else:
                mini_batch.append(sample)

        if not self.drop_last and len(mini_batch) > 0:
            batches.append(mini_batch)

        length_list = [None for _ in range(dist.get_world_size())]
        length = len(batches)
        dist.all_gather_object(length_list, length)
        keep_length = min(length_list)
        batches = batches[:keep_length]
        random.Random(self.epoch).shuffle(batches)

        self.batches = batches

    def __next__(self):
        try:
            return next(self.batch_iter)
        except StopIteration:
            raise StopIteration

    def __iter__(self):
        self.batch_iter = iter(self.batches)
        return self

    def __len__(self):
        return len(self.batches)
