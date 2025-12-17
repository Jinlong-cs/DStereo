import os
from typing import Callable, List, Optional, Union

from hat.core.adapter import TorchVisionAdapter
from hat.core.compose_transform import Compose
from hat.registry import OBJECT_REGISTRY
from .profilers import SimpleProfiler
from .python_profiler import PythonProfiler

__all__ = ["PerfTransforms"]


@OBJECT_REGISTRY.register
class PerfTransforms(Compose):
    """
    This profiler uses PythonProfiler or SimpleProfiler to perf transforms.

    All transforms' name and it's pid and time cost information will be
    recorded in one profiler report. This profiler will perf 5000 data in
    default, If you want to perf the transforms for longer time, you need
    to set perf_data_len.

    Args:
        transforms: collection of functions transform that takes input
            sample and its target as entry and returns a transformed version.
        profiler: Object of SimpleProfiler or PythonProfiler to perf
            transforms
        perf_data_len: Transforms' input Data length, during those data
            usage the transform will be perfed, default is 5000.
    """

    def __init__(
        self,
        transforms: List[Callable],
        profiler: Union[SimpleProfiler, PythonProfiler],
        perf_data_len: Optional[int] = 5000,
    ):
        super(PerfTransforms, self).__init__(transforms)
        self.profiler = profiler
        self.perfed_data_len = 0
        self.perf_data_len = perf_data_len

    def __call__(self, data):
        self.perfed_data_len += 1
        for t in self.transforms:
            perf_obj = t
            if isinstance(t, TorchVisionAdapter):
                perf_obj = t.adapter
            with self.profiler.profile(
                f"on_{perf_obj.__class__.__name__}, pid: {os.getpid()}"
            ):
                data = t(data)
            if self.perfed_data_len == self.perf_data_len:
                self.profiler.describe()
        return data
