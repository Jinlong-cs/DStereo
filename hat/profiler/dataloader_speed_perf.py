# Copyright (c) Horizon Robotics. All rights reserved.

import logging
import time
from typing import Iterator, Optional

import numpy as np
from torch.utils.data import DistributedSampler

from hat.data.dataloaders.multitask_loader import MultitaskLoader
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list
from hat.utils.generator import prefetch_iterator
from .profilers import BaseProfiler, PassThroughProfiler

__all__ = ["DataloaderSpeedPerf"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class DataloaderSpeedPerf:
    """
    Dataloader speed perf plugin, used to test dataloader loading speed.

    Args:
        dataloader: dataloader instance.
        iter_nums: perf iteration nums.
        frequent: log frequent.
        profiler: checking if there are any bottlenecks during dataloader.
    """

    def __init__(
        self,
        dataloader: Iterator,
        iter_nums: Optional[int] = 1000,
        frequent: Optional[int] = 1,
        profiler: Optional[BaseProfiler] = None,
    ):
        self.dataloader = dataloader
        self.iter_nums = iter_nums
        self.frequent = frequent
        if profiler is None:
            profiler = PassThroughProfiler()
        self.profiler = profiler

        if hasattr(dataloader, "batch_size"):
            self.batch_size = dataloader.batch_size
        elif isinstance(dataloader, MultitaskLoader):
            loaders = dataloader.loaders
            if isinstance(loaders, dict):
                self.batch_size = sum([loaders[k].batch_size for k in loaders])
            elif isinstance(loaders, list):
                self.batch_size = sum([item.batch_size for item in loaders])
            else:
                raise ValueError
        else:
            raise NotImplementedError(f"{type(loaders)} not supported")

    def run(self) -> None:
        btic = time.time()
        global_iter_id = 0
        epoch_id = 0
        speed_list, cost_list = [], []
        while True:
            self.data_loader_pr = self.profiler.profile_iterable(
                enumerate(prefetch_iterator(self.dataloader)),
                "get_batch_data",
            )
            self._set_epoch(epoch_id)
            for j, (_batch, _is_last_batch) in self.data_loader_pr:
                global_iter_id += 1
                if (j + 1) % self.frequent == 0:
                    cost = time.time() - btic
                    speed = self.frequent * self.batch_size / cost
                    s = f"Batch[{global_iter_id}] Speed: {speed:.2f} samples/sec | Cost: {cost:.3f} sec"  # noqa
                    logger.info(s)
                    speed_list.append(speed)
                    cost_list.append(cost)
                    btic = time.time()
                if global_iter_id == self.iter_nums:
                    s = f"Average Batch Speed: {np.mean(speed_list):.2f} samples/sec | Cost: {np.mean(cost_list):.2f} sec"  # noqa
                    logger.info(s)
                    self.profiler.describe()
                    return
            epoch_id += 1

    def _set_epoch(self, epoch_id):
        if hasattr(self.dataloader, "sampler"):
            sampler = self.dataloader.sampler
            if isinstance(sampler, dict):
                sampler = sampler.values()
            else:
                sampler = _as_list(sampler)

            for sa in sampler:
                if isinstance(sa, DistributedSampler):
                    sa.set_epoch(epoch_id)
