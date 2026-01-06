from __future__ import annotations

from typing import Iterator, List, Optional

import torch
from torch.utils.data import Sampler
from torch.utils.data.dataset import ConcatDataset
from hat.registry import OBJECT_REGISTRY

__all__ = ["InterleaveConcatSampler"]


@OBJECT_REGISTRY.register
class InterleaveConcatSampler(Sampler[int]):
    """
    Interleave multiple sub-datasets in a ConcatDataset in round-robin order:
      D0[0], D1[0], ..., Dk-1[0], D0[1], D1[1], ...

    Smaller datasets are repeated with modulo indexing.

    Args:
        dataset: must be torch.utils.data.dataset.ConcatDataset with >= 2 sub-datasets
        shuffle: shuffle each sub-dataset independently every epoch
        seed: base seed
        sampler_len: optional cap on the number of "rows" (i steps). Total yielded = rows * num_subdatasets.
                     Default rows = max(len(subdataset_i)).
        drop_empty: if True, ignore empty sub-datasets (len==0). If False, empty dataset raises.
    """

    def __init__(
        self,
        dataset: ConcatDataset,
        shuffle: bool = False,
        seed: int = 0,
        sampler_len: Optional[int] = None,
        drop_empty: bool = False,
    ) -> None:
        assert isinstance(dataset, ConcatDataset), "dataset must be a ConcatDataset"
        assert len(dataset.datasets) >= 2, "need at least 2 sub-datasets"

        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = int(seed)
        self.epoch = 0
        self.sampler_len = sampler_len
        self.drop_empty = drop_empty

        # Lengths of each sub-dataset
        lengths = [len(d) for d in dataset.datasets]
        if any(l == 0 for l in lengths):
            if drop_empty:
                # Keep only non-empty datasets
                self._keep = [i for i, l in enumerate(lengths) if l > 0]
                assert len(self._keep) >= 2, "after drop_empty, need at least 2 non-empty sub-datasets"
            else:
                raise ValueError(f"Found empty sub-dataset(s) with lengths={lengths}. "
                                 f"Set drop_empty=True to ignore them.")
        else:
            self._keep = list(range(len(lengths)))

        self.lengths = [lengths[i] for i in self._keep]

        # Offsets in the flattened ConcatDataset index space
        # dataset.cumulative_sizes[j] = sum_{t<=j} len(dataset.datasets[t])
        # offset for dataset j is sum_{t<j} len(dataset.datasets[t])
        offsets_all = [0]
        for sz in dataset.cumulative_sizes[:-1]:
            offsets_all.append(sz)

        self.offsets = [offsets_all[i] for i in self._keep]
        self.num_sub = len(self.lengths)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rows(self) -> int:
        # number of i-steps (each step yields one sample per sub-dataset)
        rows = max(self.lengths)
        if self.sampler_len is not None:
            rows = min(rows, int(self.sampler_len))
        return rows

    def __len__(self) -> int:
        return self._rows() * self.num_sub

    def __iter__(self) -> Iterator[int]:
        # Build per-subdataset index lists (optionally shuffled)
        per_idx: List[List[int]] = [list(range(L)) for L in self.lengths]

        if self.shuffle:
            # independent shuffle per subdataset, deterministic per epoch
            # use different seed stream per subdataset to avoid same permutations
            for j, L in enumerate(self.lengths):
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch + 9973 * j)
                per_idx[j] = torch.randperm(L, generator=g).tolist()

        rows = self._rows()
        for i in range(rows):
            for j in range(self.num_sub):
                local = per_idx[j][i % self.lengths[j]]
                yield self.offsets[j] + local

