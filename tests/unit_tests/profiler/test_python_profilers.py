import time

import numpy as np
import pytest
import torch

from hat.profiler.python_profiler import PythonProfiler
from tests.unit_tests.base import BoringModel, FakeAuto2dDataset

PROFILER_OVERHEAD_MAX_TOLERANCE = 0.0005


def _get_python_cprofile_total_duration(profile):
    return sum([x.inlinetime for x in profile.getstats()])


def _sleep_generator(durations):
    """
    the profile_iterable method needs an iterable in which we can ensure
    that we're properly timing how long it takes to call __next__
    """
    for duration in durations:
        time.sleep(duration)
        yield duration


@pytest.fixture
def python_profiler():
    return PythonProfiler()


@pytest.mark.parametrize(
    ["action", "expected"],
    [
        pytest.param("a", [3, 1]),
        pytest.param("b", [2]),
        pytest.param("c", [1]),
    ],
)
def test_python_profiler_durations(
    python_profiler, action: str, expected: list
):

    for duration in expected:
        with python_profiler.profile(action):
            time.sleep(duration)

    # different environments have different precision
    # when it comes to time.sleep()
    recored_total_duration = _get_python_cprofile_total_duration(
        python_profiler.profiled_actions[action]
    )
    expected_total_duration = np.sum(expected)
    np.testing.assert_allclose(
        recored_total_duration, expected_total_duration, rtol=0.2
    )


@pytest.mark.parametrize(
    ["action", "expected"],
    [
        pytest.param("a", [3, 1]),
        pytest.param("b", [2]),
        pytest.param("c", [1]),
    ],
)
def test_python_profiler_iterable_durations(
    python_profiler, action: str, expected: list
):
    """Ensure the reported durations are reasonably accurate."""
    iterable = _sleep_generator(expected)

    for _ in python_profiler.profile_iterable(iterable, action):
        pass

    recored_total_duration = _get_python_cprofile_total_duration(
        python_profiler.profiled_actions[action]
    )
    expected_total_duration = np.sum(expected)
    np.testing.assert_allclose(
        recored_total_duration, expected_total_duration, rtol=0.2
    )


def test_python_profiler_overhead(python_profiler, n_iter=5):
    """
    ensure that the profiler doesn't introduce too much
    overhead during training
    """
    for _ in range(n_iter):
        with python_profiler.profile("no-op"):
            pass

    action_profile = python_profiler.profiled_actions["no-op"]
    total_duration = _get_python_cprofile_total_duration(action_profile)
    average_duration = total_duration / n_iter
    assert average_duration < PROFILER_OVERHEAD_MAX_TOLERANCE


def test_python_profiler_describe(python_profiler):
    dataset = FakeAuto2dDataset()
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
    model = BoringModel()
    for _, data in enumerate(data_loader):
        with python_profiler.profile("model_forward"):
            output = model(data)  # noqa: F841
    python_profiler.describe()
