from typing import Optional, Union

import torch.utils.data as data

from hat.registry import OBJECT_REGISTRY
from .memory_profiler import StageCPUMemoryProfiler
from .python_profiler import PythonProfiler

__all__ = [
    "PerfDataset",
]

_PROFILER = Union[
    StageCPUMemoryProfiler,
    PythonProfiler,
]


@OBJECT_REGISTRY.register
class PerfDataset(object):
    """
    Perf the inner dataset time cost or memory usage.

    If use StageCPUMemoryProfiler as profiler, This dataset will perf the
    memory usage of the internal dataset every interval count is executed.
    If use PythonProfiler as profiler, This dataset will perf the
    time cost of get data.

    Args:
        dataset: The perfed dataset used to get data.
        perf_interval: The interval count to perf dataset memory usage.
    """

    def __init__(
        self,
        dataset: data.Dataset,
        profiler: _PROFILER,
        perf_interval: int = 100,
        perf_data_len: Optional[int] = 5000,
    ):
        self.dataset = dataset
        self.perf_interval = perf_interval
        self.profiler = profiler
        self.perf_data_len = perf_data_len
        self.ncalls = 0
        self.perf_time = False

        assert isinstance(self.profiler, (PythonProfiler, StageCPUMemoryProfiler)), (
            f"Unsupported profiler: {self.profiler}, ",
            "expected PythonProfiler or StageCPUMemoryProfiler",
        )
        if isinstance(self.profiler, PythonProfiler):
            self.perf_time = True

    def _getitem_with_time_perf(self, idx):
        if self.ncalls < self.perf_data_len:
            with self.profiler.profile("dataset"):
                res = self.dataset[idx]
        else:
            res = self.dataset[idx]
        self.ncalls += 1
        if self.ncalls == self.perf_data_len:
            self.profiler.describe()
        return res

    def _getitem_with_mem_perf(self, idx):
        self.ncalls += 1
        if self.ncalls % self.perf_interval == 0:
            with self.profiler.profile(self.profiler.profile_action_name):
                res = self.dataset[idx]
            self.profiler.describe_midway(self.profiler.index - 1)
            return res
        else:
            return self.dataset[idx]

    def __getitem__(self, idx):
        if self.perf_time:
            return self._getitem_with_time_perf(idx)
        else:
            return self._getitem_with_mem_perf(idx)

    def __len__(self):
        return len(self.dataset)
