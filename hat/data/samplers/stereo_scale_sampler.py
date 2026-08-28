"""Batch-aligned deterministic scale scheduling for stereo datasets."""

from __future__ import annotations
from typing import Iterator, Optional, Sequence

import torch
import torch.distributed as distributed
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from hat.registry import OBJECT_REGISTRY


def _distributed_size_and_rank(
    num_replicas: Optional[int], rank: Optional[int]
) -> tuple[int, int]:
    if num_replicas is None:
        num_replicas = (
            distributed.get_world_size() if distributed.is_initialized() else 1
        )
    if rank is None:
        rank = distributed.get_rank() if distributed.is_initialized() else 0
    return int(num_replicas), int(rank)


@OBJECT_REGISTRY.register
class StereoScaleSampler(DistributedSampler):
    """Interleave stereo datasets and assign one scale to each batch.

    Scale assignment happens before rank sharding, so every distributed rank
    uses the same shape at a given optimizer step. Every complete schedule
    cycle contains each configured scale exactly once in seeded random order.
    """

    def __init__(
        self,
        dataset: ConcatDataset,
        batch_size: int,
        scales: Sequence[float],
        shuffle: bool = True,
        seed: int = 0,
        sampler_len: Optional[int] = None,
        drop_empty: bool = False,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        drop_last: bool = True,
    ) -> None:
        if not isinstance(dataset, ConcatDataset):
            raise TypeError("StereoScaleSampler requires a ConcatDataset")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not scales:
            raise ValueError("at least one scale is required")
        normalized_scales = tuple(float(scale) for scale in scales)
        if len({round(scale, 8) for scale in normalized_scales}) != len(
            normalized_scales
        ):
            raise ValueError("scales must be unique")
        if any(not 0.0 < scale <= 1.0 for scale in normalized_scales):
            raise ValueError(
                f"scales must be in (0, 1], got {normalized_scales}"
            )
        if not drop_last:
            raise ValueError(
                "drop_last=False is unsupported because a partial global "
                "batch would break scale alignment across ranks"
            )

        resolved_replicas, resolved_rank = _distributed_size_and_rank(
            num_replicas, rank
        )
        super().__init__(
            dataset,
            num_replicas=resolved_replicas,
            rank=resolved_rank,
            shuffle=False,
            seed=int(seed),
            drop_last=True,
        )
        self.batch_size = int(batch_size)
        self.scales = normalized_scales
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.sampler_len = sampler_len
        self.drop_empty = bool(drop_empty)
        self._keep, self._lengths, self._offsets = self._dataset_layout(
            dataset, drop_empty=self.drop_empty
        )
        self._global_batch_count = self._compute_global_batch_count()
        self.num_samples = self._global_batch_count * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    @staticmethod
    def _dataset_layout(
        dataset: ConcatDataset,
        *,
        drop_empty: bool,
    ) -> tuple[list[int], list[int], list[int]]:
        lengths = [len(child) for child in dataset.datasets]
        if any(length == 0 for length in lengths):
            if not drop_empty:
                raise ValueError(
                    "StereoScaleSampler found an empty child dataset; set "
                    "drop_empty=True to ignore it"
                )
            keep = [
                index for index, length in enumerate(lengths) if length > 0
            ]
            if not keep:
                raise ValueError(
                    "StereoScaleSampler requires a non-empty dataset"
                )
        else:
            keep = list(range(len(lengths)))
        offsets = [0]
        offsets.extend(dataset.cumulative_sizes[:-1])
        return keep, lengths, offsets

    def _rows(self) -> int:
        rows = max(self._lengths[index] for index in self._keep)
        if self.sampler_len is not None:
            if int(self.sampler_len) <= 0:
                raise ValueError("sampler_len must be positive")
            rows = min(rows, int(self.sampler_len))
        return rows

    def _base_stream(self) -> list[int]:
        per_dataset = []
        for child_index in self._keep:
            length = self._lengths[child_index]
            indices = list(range(length))
            if self.shuffle:
                generator = torch.Generator()
                generator.manual_seed(
                    self.seed + self.epoch + 9973 * child_index
                )
                indices = torch.randperm(length, generator=generator).tolist()
            per_dataset.append(indices)

        stream = []
        for row in range(self._rows()):
            for child_position, child_index in enumerate(self._keep):
                local = per_dataset[child_position][
                    row % len(per_dataset[child_position])
                ]
                stream.append(self._offsets[child_index] + local)
        return stream

    def _compute_global_batch_count(self) -> int:
        global_batch = self.batch_size * self.num_replicas
        count = len(self._base_stream()) // global_batch
        count -= count % len(self.scales)
        if count <= 0:
            raise ValueError(
                "dataset/sampler_len is too short for one complete scale cycle"
            )
        return count

    @property
    def num_batches(self) -> int:
        return self._global_batch_count

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[tuple[int, float]]:
        stream = self._base_stream()
        global_batch_size = self.batch_size * self.num_replicas
        scale_order = list(range(len(self.scales)))

        for batch_index in range(self._global_batch_count):
            cycle_index, cycle_position = divmod(batch_index, len(self.scales))
            if self.shuffle and cycle_position == 0:
                generator = torch.Generator()
                generator.manual_seed(
                    self.seed + self.epoch * 1000003 + cycle_index
                )
                scale_order = torch.randperm(
                    len(self.scales), generator=generator
                ).tolist()
            start = batch_index * global_batch_size
            end = start + global_batch_size
            scale = self.scales[scale_order[cycle_position]]
            global_indices = stream[start:end]
            rank_start = self.rank * self.batch_size
            rank_end = rank_start + self.batch_size
            for flat_index in global_indices[rank_start:rank_end]:
                yield flat_index, scale


__all__ = ["StereoScaleSampler"]
