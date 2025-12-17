# Copyright (c) Horizon Robotics, All rights reserved.
import logging
import random
import sys
from math import ceil
from typing import Any, Sequence, Union

from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader


class ChainDataloader:
    """Chain dataloaders together.

    Args:
        dataloaders: A sequence of dataloaders.
        proportions: A sequence of proportions.
        seed: Random seed.
    """

    def __init__(
        self,
        dataloaders: Sequence[DataLoader],
        proportions: Sequence[Union[int, float]],
        seed: int = 0,
    ):
        assert len(dataloaders) == len(proportions) and len(dataloaders) > 1
        self.data_loaders = dataloaders
        self.proportions = proportions
        self.data_loader_iters = [
            iter(dataloader) for dataloader in dataloaders
        ]
        min_length = sys.maxsize
        for idx, dataloader in enumerate(dataloaders):
            if isinstance(dataloader.dataset, IterableDataset):
                continue
            _length = ceil(len(dataloader) / proportions[idx])
            min_length = _length if _length < min_length else min_length
        assert min_length != sys.maxsize
        self.length = min_length
        self.epoch = 0
        self.seed = seed

    def __iter__(self) -> Any:
        self.epoch += 1
        self.counter = 0
        indices = []
        for idx, proportion in enumerate(self.proportions):
            _length = ceil(self.length * proportion)
            indices.extend([idx for _ in range(_length)])
        random.Random(self.seed + self.epoch).shuffle(indices)
        self.iter_choose = iter(indices)
        return self

    def __next__(self) -> Any:
        if self.counter >= self.length:
            raise StopIteration

        try:
            mode = next(self.iter_choose)
            try:
                return next(self.data_loader_iters[mode])

            except StopIteration:
                logging.info(f"Dataloader {mode} train an epoch!")
                self.data_loader_iters[mode] = iter(self.data_loaders[mode])
                return next(self.data_loader_iters[mode])
        finally:
            self.counter += 1

    def __len__(self):
        return self.length


class MultiBatchChainDataloader:
    """Chain dataloaders together with multi-batch data.

    Args:
        dataloaders: A sequence of dataloaders.
        proportions: A sequence of proportions.
        batches: A sequence of batches.
        seed: Random seed.
    """

    def __init__(
        self,
        dataloaders: Sequence[DataLoader],
        proportions: Sequence[Union[int, float]],
        batches: Sequence[int],
        seed: int = 0,
    ):
        assert (
            len(dataloaders) == len(proportions) == len(batches)
            and len(dataloaders) >= 1
        )
        self.data_loaders = dataloaders
        self.proportions = proportions
        self.data_loader_iters = [
            iter(dataloader) for dataloader in dataloaders
        ]
        lengths = []
        for i, dataloader in enumerate(dataloaders):
            if isinstance(dataloader.dataset, IterableDataset):
                continue
            length = ceil(len(dataloader) * 1.0 / batches[i] / proportions[i])
            lengths.append(length)
        self.length = min(lengths)
        self.epoch = 0
        self.seed = seed
        self.batches = batches

    def __iter__(self) -> Any:
        self.epoch += 1
        self.counter = 0
        indices = []
        for idx, proportion in enumerate(self.proportions):
            _length = ceil(self.length * proportion)
            indices.extend([idx for _ in range(_length)])
        random.Random(self.seed + self.epoch).shuffle(indices)
        self.iter_choose = iter(indices)
        return self

    def __next__(self) -> Any:
        if self.counter >= self.length:
            raise StopIteration

        try:
            mode = next(self.iter_choose)
            batches = [
                self._get_one_batch(mode) for _ in range(self.batches[mode])
            ]
        finally:
            self.counter += 1

        return tuple(batches)

    def _get_one_batch(self, mode):
        try:
            return next(self.data_loader_iters[mode])

        except StopIteration:
            logging.info(f"dataloader {mode} train an epoch!")
            self.data_loader_iters[mode] = iter(self.data_loaders[mode])
            return next(self.data_loader_iters[mode])

    def __len__(self):
        return self.length
