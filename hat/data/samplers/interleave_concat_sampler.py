# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Iterator

import torch
from torch.utils.data import Sampler
from torch.utils.data.dataset import ConcatDataset
from hat.registry import OBJECT_REGISTRY

__all__ = ["InterleaveConcatSampler"]


@OBJECT_REGISTRY.register
class InterleaveConcatSampler(Sampler[int]):
    """Interleave two sub-datasets: A0,B0,A1,B1..., repeat smaller one."""

    def __init__(self, dataset, shuffle: bool = False, seed: int = 0,  sampler_len = None) -> None:
        assert isinstance(dataset, ConcatDataset)
        assert len(dataset.datasets) == 2
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.len_a = len(dataset.datasets[0])
        self.len_b = len(dataset.datasets[1])
        self.offset_b = dataset.cumulative_sizes[0]
        self.sampler_len = sampler_len

    def __len__(self) -> int:
        return 2 * max(self.len_a, self.len_b)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        a_idx = list(range(self.len_a))
        b_idx = list(range(self.len_b))
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            a_idx = torch.randperm(self.len_a, generator=g).tolist()
            g.manual_seed(self.seed + self.epoch + 1)
            b_idx = torch.randperm(self.len_b, generator=g).tolist()
        total = max(self.len_a, self.len_b)
        if self.sampler_len is not None:
            total = min(total, self.sampler_len)
        for i in range(total):
            #print("sample a from: ",a_idx[i % self.len_a])
            yield a_idx[i % self.len_a]
            #print("sample b from: ",self.offset_b + b_idx[i % self.len_b])
            yield self.offset_b + b_idx[i % self.len_b]
