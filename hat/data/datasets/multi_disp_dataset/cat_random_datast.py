#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import random
import traceback
from bisect import bisect_right

from torch.utils.data.dataset import ConcatDataset

logger = logging.getLogger(__name__)
__all__ = ["CatRandomDataset"]


class CatRandomDataset(ConcatDataset):
    """Concatenate datasets and preserve optional resize-scale index tags."""

    def __getitem__(self, item):
        scale = item[1] if isinstance(item, tuple) else None
        while True:
            try:
                data = self._get_item(item)
                if not data["mask_flag"]:
                    item = random.randint(0, len(self) - 1)
                    if scale is not None:
                        item = (item, scale)
                    print("mask_flag False, retrying...")
                    continue
                return data
            except Exception as e:
                logger.info(traceback.format_exc())
                logger.info(e)
                item = random.randint(0, len(self) - 1)
                if scale is not None:
                    item = (item, scale)

    def _get_item(self, item):
        if not isinstance(item, tuple):
            return super().__getitem__(item)
        flat_index, scale = item
        dataset_index = bisect_right(self.cumulative_sizes, flat_index)
        if dataset_index == 0:
            sample_index = flat_index
        else:
            sample_index = (
                flat_index - self.cumulative_sizes[dataset_index - 1]
            )
        return self.datasets[dataset_index][(sample_index, scale)]
