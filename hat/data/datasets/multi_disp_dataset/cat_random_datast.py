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
    """Concatenate datasets and retry invalid samples with bounded attempts."""

    max_retries = 50

    def __getitem__(self, item):
        item = self._normalize_index(item)
        last_error = None
        for _ in range(self.max_retries):
            try:
                data = self._get_item(item)
                if not data["mask_flag"]:
                    item = self._replacement_index(item)
                    continue
                return data
            except Exception as e:
                logger.info(traceback.format_exc())
                logger.info(e)
                last_error = e
                item = self._replacement_index(item)

        message = (
            f"failed to load a valid sample after {self.max_retries} attempts"
        )
        if last_error is not None:
            raise RuntimeError(message) from last_error
        raise RuntimeError(message)

    def _normalize_index(self, item):
        if not isinstance(item, tuple):
            flat_index = operator.index(item)
            self._validate_flat_index(flat_index)
            return flat_index
        if len(item) != 2:
            raise ValueError(
                "scale-tagged indices must be (flat_dataset_index, scale) pairs"
            )
        flat_index = operator.index(item[0])
        self._validate_flat_index(flat_index)
        scale = float(item[1])
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"scale must be finite and positive, got {item[1]!r}"
            )
        return flat_index, scale

    def _validate_flat_index(self, flat_index):
        if flat_index >= len(self) or flat_index < -len(self):
            raise IndexError(
                f"index {flat_index} is out of range for dataset of size {len(self)}"
            )

    def _get_item(self, item):
        if not isinstance(item, tuple):
            return super().__getitem__(item)

        flat_index, scale = item
        if flat_index < 0:
            if -flat_index > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            flat_index = len(self) + flat_index
        dataset_index = bisect_right(self.cumulative_sizes, flat_index)
        if dataset_index == 0:
            sample_index = flat_index
        else:
            sample_index = (
                flat_index - self.cumulative_sizes[dataset_index - 1]
            )
        return self.datasets[dataset_index][(sample_index, scale)]

    def _replacement_index(self, item):
        replacement = random.randint(0, len(self) - 1)
        if isinstance(item, tuple):
            return replacement, item[1]
        return replacement
