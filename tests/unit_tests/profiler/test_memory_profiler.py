import os

try:
    import memray
except ImportError:
    memray = None

from distutils.version import LooseVersion

import pytest
import torch

from hat.profiler.memory_profiler import (
    CPUMemoryProfiler,
    GPUMemoryProfiler,
    StageCPUMemoryProfiler,
)
from hat.utils.distributed import get_local_host
from tests.unit_tests.base import BoringModel, FakeAuto2dDataset


@pytest.fixture
def gpu_memory_profiler():
    return GPUMemoryProfiler()


def test_gpu_memory_profiler_describe(gpu_memory_profiler):
    dataset = FakeAuto2dDataset()
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
    model = BoringModel()
    for _, data in enumerate(data_loader):
        with gpu_memory_profiler.profile("model_forward"):
            output = model(data)  # noqa: F841
    gpu_memory_profiler.describe()


def test_gpu_memory_profiler_snapshot(tmpdir):
    profiler = GPUMemoryProfiler(dirpath=tmpdir, record_snapshot=True)
    dataset = FakeAuto2dDataset()
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
    model = BoringModel()
    for _, data in enumerate(data_loader):
        with profiler.profile("model_forward"):
            output = model(data)  # noqa: F841

    profiler.describe()
    if LooseVersion(torch.__version__) >= LooseVersion(
        "1.13.0"
    ) and not LooseVersion(torch.__version__) >= LooseVersion("2.0.1"):
        assert os.path.exists(os.path.join(tmpdir, "snapshots"))
        assert os.path.exists(
            os.path.join(
                tmpdir,
                f"snapshot_{get_local_host()}-rank0.pkl",
            )
        )
    else:
        assert profiler.memory_snapshot.record_snapshot is False


@pytest.fixture
def cpu_memory_profiler():
    return CPUMemoryProfiler()


def test_cpu_memory_profiler_describe(cpu_memory_profiler):
    dataset = FakeAuto2dDataset()
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
    model = BoringModel()
    for _, data in enumerate(data_loader):
        with cpu_memory_profiler.profile("model_forward"):
            output = model(data)  # noqa: F841
    cpu_memory_profiler.describe()


@pytest.fixture
def stage_cpu_memory_leaks_profiler(tmpdir):
    os.environ["PYTHONMALLOC"] = "malloc"
    return StageCPUMemoryProfiler(
        profile_action_name="model_forward",
        dirpath=tmpdir,
        filename="model_forward_leaks_ut",
    )


@pytest.mark.skipif(memray is None, reason="memray is required")
def test_stage_cpu_memory_leaks_profiler_describe(
    stage_cpu_memory_leaks_profiler, tmpdir
):
    dataset = FakeAuto2dDataset()
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
    model = BoringModel()
    for _, data in enumerate(data_loader):
        with stage_cpu_memory_leaks_profiler.profile("model_forward"):
            output = model(data)  # noqa: F841
    stage_cpu_memory_leaks_profiler.describe()

    assert os.path.exists(
        os.path.join(
            tmpdir,
            f"memray-flamegraph-model_forward_leaks_"
            f"ut_rank0_0_{os.getpid()}.html",
        )
    )
    os.environ.pop("PYTHONMALLOC")


@pytest.fixture
def stage_cpu_memory_profiler(tmpdir):
    return StageCPUMemoryProfiler(
        profile_action_name="model_forward",
        dirpath=tmpdir,
        leaks=False,
        filename="model_forward_ut",
    )


@pytest.mark.skipif(memray is None, reason="memray is required")
def test_stage_cpu_memory_profiler_describe(stage_cpu_memory_profiler, tmpdir):
    dataset = FakeAuto2dDataset()
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
    model = BoringModel()
    for _, data in enumerate(data_loader):
        with stage_cpu_memory_profiler.profile("model_forward"):
            output = model(data)  # noqa: F841
    stage_cpu_memory_profiler.describe()
    assert os.path.exists(
        os.path.join(
            tmpdir,
            f"memray-flamegraph-model_forward_ut_rank0_0_{os.getpid()}.html",
        )
    )
