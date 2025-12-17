# Copyright (c) Horizon Robotics. All rights reserved.
import contextlib
import logging
from typing import Any

from addict import Dict

try:
    import mxnet as mx
except ImportError:
    mx = None

import torch
import torch.distributed as dist

try:
    from auto_matrix.data.data_loader import build_gluon_data_loader
    from matrix_gluon.data.data_loader import build_multi_cached_data_loader
except ImportError:
    pass

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list
from hat.utils.deprecate import deprecated_warning
from hat.utils.logger import OutputLogger
from hat.utils.package_helper import require_packages

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class MultiCachedDataLoader(object):
    @require_packages("mxnet", "matrix_gluon")
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle,
        last_batch,
        num_workers,
        transform=None,
        batchify=None,
        chunk_size=32,
        use_prefetcher=False,
        min_chunk_num=4,
        max_chunk_num=8,
        min_prefetch=1,
        max_prefetch=2,
        batched_transform=True,
        skip_batchify=False,
        prefetcher_using_thread=True,
        is_nchw=False,
        input_padding=None,
    ):
        deprecated_warning(
            author="tian.li",
            old_name="MultiCachedDataLoader",
            deprecation_version="v1.2.0",
            removal_version="v1.4.0",
        )
        self._update_dataset_dist_info(dataset)
        transform = (
            transform
            if transform is None
            else [Dict(t) for t in _as_list(transform)]
        )
        with contextlib.redirect_stdout(OutputLogger(logger)):
            self._loader = build_multi_cached_data_loader(
                Dict(dataset),
                batch_size,
                shuffle,
                last_batch,
                num_workers,
                chunk_size=chunk_size,
                transform=transform,
                batchify=batchify,
                use_prefetcher=use_prefetcher,
                min_chunk_num=min_chunk_num,
                max_chunk_num=max_chunk_num,
                min_prefetch=min_prefetch,
                max_prefetch=max_prefetch,
                batched_transform=batched_transform,
                skip_batchify=skip_batchify,
                prefetcher_using_thread=prefetcher_using_thread,
                process_start_methods="spawn",
            )
        self.is_nchw = is_nchw

        assert input_padding is None or len(input_padding) == 4
        self._input_padding = input_padding

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)

    def __len__(self):
        return (
            sum([len(d.dataset) for d in self._loader._dataset.datasets])
            // self.batch_size
        )

    @property
    def batch_size(self):
        return self._batch_size

    @staticmethod
    def _update_dataset_dist_info(dataset):

        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")

        try:
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        except RuntimeError:
            logger.warning("Distributed environments not initialized")
            rank = 0
            world_size = 1

        def _rec_update_dict(item, **kwargs):
            if isinstance(item, list):
                for i in item:
                    _rec_update_dict(i, **kwargs)
            elif isinstance(item, dict):
                if item["type"] == "SplitDataset":
                    item.update(kwargs)
                elif "dataset" in item:
                    _rec_update_dict(item["dataset"], **kwargs)

        _rec_update_dict(dataset, part_id=rank, n_part=world_size)

    def nd_to_tensor(self, nd_dict):
        def _nd_to_np(nd):
            if isinstance(nd, list):
                return [_nd_to_np(_) for _ in nd]
            elif isinstance(nd, dict):
                return {k: _nd_to_np(v) for k, v in nd.items()}
            else:
                return (
                    torch.from_numpy(nd.asnumpy())
                    if isinstance(nd, mx.nd.NDArray)
                    else nd
                )

        res_dict = _nd_to_np(nd_dict)
        if not self.is_nchw:
            res_dict["img"] = res_dict["img"].permute(0, 3, 1, 2)
        if self._input_padding is not None:
            res_dict["im_hw"][:, 0] += (
                self._input_padding[2] + self._input_padding[3]
            )
            res_dict["im_hw"][:, 1] += (
                self._input_padding[0] + self._input_padding[1]
            )
        return res_dict

    def __iter__(self):
        return map(self.nd_to_tensor, iter(self._loader))


@OBJECT_REGISTRY.register
class GluonDataLoader(object):
    @require_packages("mxnet", "matrix_gluon")
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle,
        last_batch,
        num_workers,
        transform,
        batchify=None,
        sampler=None,
        input_padding=None,
    ):
        deprecated_warning(
            author="tian.li",
            old_name="GluonDataLoader",
            deprecation_version="v1.2.0",
            removal_version="v1.4.0",
        )
        self._batch_size = batch_size
        self._update_dataset_dist_info(dataset)
        transform = (
            transform
            if transform is None
            else [Dict(t) for t in _as_list(transform)]
        )
        with contextlib.redirect_stdout(OutputLogger(logger)):
            self._loader = build_gluon_data_loader(
                [Dict(d) for d in dataset],
                batch_size,
                last_batch,
                num_workers,
                shuffle,
                transform=transform,
                sampler=sampler,
                batchify=batchify,
            )

        assert input_padding is None or len(input_padding) == 4
        self._input_padding = input_padding

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)

    @property
    def batch_size(self):
        return self._batch_size

    @staticmethod
    def _update_dataset_dist_info(dataset):

        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")

        try:
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        except RuntimeError:
            logger.warning("Distributed environments not initialized")
            rank = 0
            world_size = 1

        def _rec_update_dict(item, **kwargs):
            if isinstance(item, list):
                for i in item:
                    _rec_update_dict(i, **kwargs)
            elif isinstance(item, dict):
                if item["type"] == "SplitDataset":
                    item.update(kwargs)
                elif "dataset" in item:
                    _rec_update_dict(item["dataset"], **kwargs)

        _rec_update_dict(dataset, part_id=rank, n_part=world_size)

    def nd_to_tensor(self, nd_dict):
        def _nd_to_np(nd):
            if isinstance(nd, list):
                return [_nd_to_np(_) for _ in nd]
            elif isinstance(nd, dict):
                return {k: _nd_to_np(v) for k, v in nd.items()}
            else:
                return (
                    torch.as_tensor(nd.asnumpy())
                    if isinstance(nd, mx.nd.NDArray)
                    else nd
                )

        res_dict = _nd_to_np(nd_dict)
        res_dict["img"] = res_dict["img"].permute(0, 3, 1, 2)

        if self._input_padding is not None:
            res_dict["im_hw"][:, 0] += (
                self._input_padding[2] + self._input_padding[3]
            )
            res_dict["im_hw"][:, 1] += (
                self._input_padding[0] + self._input_padding[1]
            )

        return res_dict

    def __iter__(self):
        return map(self.nd_to_tensor, iter(self._loader))
