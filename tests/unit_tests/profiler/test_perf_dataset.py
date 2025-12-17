import os

import pytest

from hat.registry import build_from_registry

try:
    import memray
except ImportError:
    memray = None


@pytest.mark.skipif(memray is None, reason="need memray")
def test_perf_mem_dataset():
    dataset_length = 100
    profiler = dict(
        type="StageCPUMemoryProfiler",
        profile_action_name="perf_dataset",
        leaks=False,
        dirpath="./",
        filename="stage_cpu_profiler",
    )
    dataset = dict(
        type="PerfDataset",
        dataset=dict(
            type="SimpleDataset",
            start=0,
            length=dataset_length,
        ),
        profiler=profiler,
        perf_interval=10,
    )
    perf_mem_dataset = build_from_registry(dataset)
    for i in range(dataset_length):
        perf_mem_dataset[i]
    pid = os.getpid()
    for i in range(10):
        assert os.path.exists(
            f"memray-flamegraph-stage_cpu_profiler_rank0_{i}_{pid}.html"
        )


def test_perf_time_dataset():
    dataset_length = 100
    python_profiler = dict(
        type="PythonProfiler",
        dirpath="work_dirs/hat_logss",
        filename="python_profiler",
    )
    dataset = dict(
        type="PerfDataset",
        dataset=dict(
            type="SimpleDataset",
            start=0,
            length=dataset_length,
        ),
        profiler=python_profiler,
        perf_data_len=100,
    )
    perf_time_dataset = build_from_registry(dataset)
    for i in range(dataset_length):
        perf_time_dataset[i]

    os.path.exists("python_profiler.txt")
