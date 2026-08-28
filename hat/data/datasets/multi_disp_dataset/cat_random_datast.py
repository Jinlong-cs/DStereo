#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import math
import operator
import random
import traceback
from bisect import bisect_right

from torch.utils.data.dataset import ConcatDataset

logger = logging.getLogger(__name__)
__all__ = ["CatRandomDataset"]


class CatRandomDataset(ConcatDataset):
    """Concatenate datasets and preserve optional resize-scale index tags."""

    tagged_max_retries = 50

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            return self._legacy_getitem(item)

        flat_index, scale = self._normalize_tagged_index(item)
        last_error = None
        for _ in range(self.tagged_max_retries):
            try:
                data = self._get_tagged_item(flat_index, scale)
                if data["mask_flag"]:
                    return data
            except Exception as error:
                logger.info(traceback.format_exc())
                logger.info(error)
                last_error = error
            flat_index = random.randint(0, len(self) - 1)

        message = (
            "failed to load a valid scale-tagged sample after "
            f"{self.tagged_max_retries} attempts"
        )
        if last_error is not None:
            raise RuntimeError(message) from last_error
        raise RuntimeError(message)

    def _legacy_getitem(self, item):
        """Keep the original scalar-index retry behavior unchanged."""

        while True:
            try:
                data = super().__getitem__(item)
                if not data["mask_flag"]:
                    item = random.randint(0, len(self) - 1)
                    print("mask_flag False, retrying...")
                    continue
                return data
            except Exception as error:
                logger.info(traceback.format_exc())
                logger.info(error)
                item = random.randint(0, len(self) - 1)

    def _normalize_tagged_index(self, item):
        if len(item) != 2:
            raise ValueError(
                "scale-tagged indices must be "
                "(flat_dataset_index, scale) pairs"
            )
        flat_index = operator.index(item[0])
        if flat_index < 0 or flat_index >= len(self):
            raise IndexError(
                f"index {flat_index} is out of range for dataset "
                f"of size {len(self)}"
            )
        scale = float(item[1])
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"scale must be finite and positive, got {item[1]!r}"
            )
        return flat_index, scale

    def _get_tagged_item(self, flat_index, scale):
        dataset_index = bisect_right(self.cumulative_sizes, flat_index)
        if dataset_index == 0:
            sample_index = flat_index
        else:
            sample_index = (
                flat_index - self.cumulative_sizes[dataset_index - 1]
            )
        return self.datasets[dataset_index][(sample_index, scale)]
