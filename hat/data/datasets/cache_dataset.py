# Copyright (c) Horizon Robotics. All rights reserved.

import glob
import logging
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.utils.data as data

from hat.data.datasets.dataset_wrappers import ConcatDataset
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list, apply_to_collection
from hat.utils.cache import Cache, CacheIOType

__all__ = ["CacheDataset"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class CacheDataset(data.Dataset):
    """Dataset fetching training data from cache file.

    This dataset is used in feature cache training to speed up training.
    You should replace original dataset with `CacheDataset` when in
    `read_cache` mode.

    NOTE: write cache时采用ddp模式将会保存多个cache file，因此CacheDataset会
    找到data_path开头的所有cache file进行读取。

    Args:
        data_path: Cache data file path.
        input_name: Input name in cache file.
        wrapped_input_name: New input name. MUST be same as
            `wrapped_input_name` in FeatureCache callback.
        transforms: Transforms applied on data.
        extra_keys: Extra keys required by graph model.
    """

    def __init__(
        self,
        data_path: str,
        input_name: str,
        wrapped_input_name: Optional[str] = "cached_feature",
        transforms: Optional[Callable] = None,
        extra_keys: Optional[List[str]] = None,
    ):
        self.data_path = _as_list(data_path)

        cache_paths = []
        for p in self.data_path:
            cache_paths.extend(glob.glob(p.replace(".rec", "*.rec")))
        if len(cache_paths) == 0:
            raise FileNotFoundError("no cache file to read")

        multi_caches = []
        for one_path in cache_paths:
            multi_caches.append(
                Cache(
                    method=CacheIOType.MXRecord,
                    cache_file=one_path,
                    writable=False,
                )
            )

        self._cache = ConcatDataset(multi_caches)

        assert len(self._cache) > 0, f"empty cache file: {self.data_path}"
        logger.info(f"load {self.data_path}")
        self.input_name = input_name
        self.wrapped_input_name = wrapped_input_name
        self.transforms = transforms
        self.extra_keys = extra_keys

    def __len__(self):
        return len(self._cache)

    def __getitem__(self, idx):
        data = self._cache.__getitem__(idx)
        data = apply_to_collection(data, np.ndarray, torch.from_numpy)

        assert self.input_name in data, (
            f"input_name ({self.input_name}) is "
            "not in data, check your config to make sure it is correct."
        )
        # wrap to new graph model's wrapped_input_name
        # change input_name to wrapped_input_name, required by hatbc.
        data[self.wrapped_input_name] = {
            self.input_name: data.pop(self.input_name)
        }

        # add extra useless keys
        if self.extra_keys:
            for key in self.extra_keys:
                data[key] = ""

        if self.transforms:
            data = self.transforms(data)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"data_path={self.data_path}, "
        return repr_str
