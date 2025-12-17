# Copyright (c) Horizon Robotics. All rights reserved.
"""
This file is modified from pytorch-lightning.

checking if there are any bottlenecks in your code.
"""
import cProfile
import io
import logging
import pstats
from pathlib import Path
from typing import Dict, Optional, Union

from hat.profiler.profilers import BaseProfiler
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "PythonProfiler",
]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class PythonProfiler(BaseProfiler):  # noqa: D205,D400
    """
    This profiler uses Python's cProfiler to record more detailed
    information about time spent in each function call recorded
    during a given action. The output is quite verbose and you should
    only use this if you want very detailed reports.
    """

    def __init__(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        line_count_restriction: float = 1.0,
        output_filename: Optional[str] = None,
    ) -> None:  # noqa: D205,D400
        """
        Args:
            dirpath: Directory path for the ``filename``.
            filename: If present, filename where the profiler results will be
            saved instead of printing to stdout. The ``.txt`` extension will
            be used automatically.
            line_count_restriction: this can be used to limit the number of
            functions reported for each action. either an integer
            (to select a count of lines), or a decimal fraction between 0.0
            and 1.0 inclusive (to select a percentage of lines)

        Raises:
            ValueError:
                If you attempt to stop recording an action which was
                never started.
        """
        super(PythonProfiler, self).__init__(
            dirpath=dirpath, filename=filename
        )
        self.profiled_actions: Dict[str, cProfile.Profile] = {}
        self.line_count_restriction = line_count_restriction
        logger.warning(
            "Sometimes you want to get a specific line's information,"
            "we recommand you not to use multiple processes in your code."
        )
        logger.warning(
            "PythonProfiler is suitable for perf of cpu operation,"
            "not recommended for GPU training operation."
        )

    def start(self, action_name: str) -> None:
        if action_name not in self.profiled_actions:
            self.profiled_actions[action_name] = cProfile.Profile()
        self.profiled_actions[action_name].enable()

    def stop(self, action_name: str) -> None:
        pr = self.profiled_actions.get(action_name)
        if pr is None:
            raise ValueError(
                f"Attempting to stop recording an action "
                f"({action_name}) which was never started."
            )
        pr.disable()

    def summary(self) -> str:
        recorded_stats = {}
        for action_name, pr in self.profiled_actions.items():
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
            ps.print_stats(self.line_count_restriction)
            recorded_stats[action_name] = s.getvalue()
        return self._stats_to_str(recorded_stats)

    def teardown(self) -> None:
        super(PythonProfiler, self).teardown()
        self.profiled_actions = {}

    def __reduce__(self):
        # avoids `TypeError: cannot pickle 'cProfile.Profile' object`
        return (
            self.__class__,
            (),
            {
                "dirpath": self.dirpath,
                "filename": self.filename,
                "line_count_restriction": self.line_count_restriction,
            },
        )
