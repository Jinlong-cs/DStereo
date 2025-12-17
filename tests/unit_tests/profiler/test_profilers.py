import os
import random
import time

import numpy as np
import pytest
import torch

from hat.profiler.profilers import SimpleProfiler
from hat.registry import build_from_registry
from hat.utils.generator import prefetch_iterator
from tests.unit_tests.base import BoringModel, FakeAuto2dDataset

PROFILER_OVERHEAD_MAX_TOLERANCE = 0.0005


def _sleep_generator(durations):
    """
    the profile_iterable method needs an iterable in which we can ensure
    that we're properly timing how long it takes to call __next__
    """
    for duration in durations:
        time.sleep(duration)
        yield duration


@pytest.fixture
def simple_profiler(request):
    if request is not None and hasattr(request, "param"):
        os.environ["HAT_MONITOR_INTERVAL"] = str(request.param)
    return SimpleProfiler()


@pytest.mark.parametrize(
    ["action", "expected"],
    [
        pytest.param("a", [3, 1]),
        pytest.param("b", [2]),
        pytest.param("c", [1]),
    ],
)
def test_simple_profiler_durations(
    simple_profiler, action: str, expected: list
):
    """Ensure the reported durations are reasonably accurate."""

    for duration in expected:
        with simple_profiler.profile(action):
            time.sleep(duration)

    # different environments have different precision
    # when it comes to time.sleep()
    np.testing.assert_allclose(
        simple_profiler.recorded_durations[action], expected, rtol=0.2
    )


@pytest.mark.parametrize(
    ["action", "expected"],
    [
        pytest.param("a", [3, 1]),
        pytest.param("b", [2]),
        pytest.param("c", [1]),
    ],
)
def test_simple_profiler_iterable_durations(
    simple_profiler, action: str, expected: list
):
    """Ensure the reported durations are reasonably accurate."""
    iterable = _sleep_generator(expected)

    for _ in simple_profiler.profile_iterable(iterable, action):
        pass
    np.testing.assert_allclose(
        simple_profiler.recorded_durations[action][:-1], expected, rtol=0.2
    )


def test_simple_profiler_overhead(simple_profiler, n_iter=5):
    """Ensure that the profiler doesn't introduce too much
    overhead during training."""
    for _ in range(n_iter):
        with simple_profiler.profile("no-op"):
            pass

    durations = np.array(simple_profiler.recorded_durations["no-op"])
    assert all(durations < PROFILER_OVERHEAD_MAX_TOLERANCE)


def test_simple_profiler_value_errors(simple_profiler):
    """Ensure errors are raised where expected."""

    action = "test"
    with pytest.raises(ValueError):
        simple_profiler.stop(action)

    simple_profiler.start(action)

    with pytest.raises(ValueError):
        simple_profiler.start(action)

    simple_profiler.stop(action)


def test_simple_profiler_describe(simple_profiler):
    dataset = FakeAuto2dDataset()
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
    model = BoringModel()
    for _, data in enumerate(data_loader):
        with simple_profiler.profile("model_forward"):
            output = model(data)  # noqa: F841
    simple_profiler.describe()


@pytest.mark.parametrize(
    "simple_profiler",
    [
        0.00001,
    ],
    indirect=True,
)
def test_simple_profiler_monitor(simple_profiler):
    data_loader = dict(
        type=torch.utils.data.DataLoader,
        dataset=dict(
            type="RandDataset",
            length=50,
            example=(torch.randn((3, 14, 14)), random.randint(0, 3 - 1)),
            clone=True,
        ),
        sampler=dict(type=torch.utils.data.DistributedSampler),
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    data_loader = build_from_registry(data_loader)
    model = BoringModel()
    data_loader_pr = simple_profiler.profile_iterable(
        enumerate(prefetch_iterator(data_loader)),
        "get_batch_data",
    )
    while True:
        index, (batch, _) = next(data_loader_pr)
        if index < 49:
            with simple_profiler.profile("model_forward_step_end"):
                output = model(batch)  # noqa: F841
        else:
            with simple_profiler.profile("model_forward_loop_end"):
                output = model(batch)  # noqa: F841
            break
