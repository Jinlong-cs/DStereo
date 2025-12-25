import logging
import os
from collections import defaultdict
from typing import Optional

from hat.registry import OBJECT_REGISTRY
from hat.utils.dynamo import CompileBackendWrapper
from hat.utils.global_var import get_value
from hat.utils.package_helper import require_packages
from .profilers import BaseProfiler

logger = logging.getLogger(__name__)

__all__ = ["DynamoProfiler"]


def format_str(msg):
    ret = "=" * 50 + f"{msg}" + "=" * 50 + "\n"
    return ret


@OBJECT_REGISTRY.register
class DynamoProfiler(BaseProfiler):
    """Torch dynamo profiler.

    Args:
        dirpath: Directory path for the ``filename``.
        filename: If present, filename where the profiler results will be
            saved instead of printing to stdout. The ``.log`` extension will
            be used automatically.
        skip_step: Skip step will not profile. Defaults to 1.
        reset_on_per_batch: Whether to reset graph-break and guard-failures
            info on per batch. Default True.
            If True, the output will always be infos of current batch.
            If False, the output will be accumulated for all batch.
    """

    RECORD_FUNCTIONS = {"model_forward"}

    @require_packages("torch>=2.0")
    def __init__(
        self,
        dirpath: Optional[str] = None,
        filename: Optional[str] = None,
        skip_step: int = 1,
        reset_on_per_batch: bool = True,
    ) -> None:
        super().__init__(dirpath, filename)

        self.steps = defaultdict()
        self.recorded_metrics = defaultdict(defaultdict)
        self.current_actions = set()
        self.skip_step = skip_step
        self.reset_on_per_batch = reset_on_per_batch

        # set flag for compile backend
        self._set_mark()

    def start(self, action_name: str) -> None:
        if action_name in self.current_actions:
            raise ValueError(
                f"Attempted to start {action_name} " f"which has already started."
            )
        if self._match_any_record_func(
            action_name=action_name,
            record_funcs=self.RECORD_FUNCTIONS,
            strict=False,
        ):
            self.current_actions.add(action_name)

    def stop(self, action_name: str) -> None:
        if self._match_any_record_func(
            action_name=action_name,
            record_funcs=self.RECORD_FUNCTIONS,
            strict=False,
        ):
            if action_name not in self.current_actions:
                raise ValueError(
                    f"Attempting to stop recording an action "
                    f"{action_name} which was never started."
                )
            self.current_actions.remove(action_name)
            step = self.steps.get(action_name, 0)

            if step >= self.skip_step:
                metric = {
                    "explain": self.dynamo_profiler.get_explain_output(),
                    "guard_failure": self.dynamo_profiler.get_guard_failures_output(),  # noqa E501
                }
                self.recorded_metrics[str(step)][action_name] = metric

            self.steps[action_name] = step + 1
            if self.reset_on_per_batch:
                self.dynamo_profiler.reset()

    def summary(self) -> str:
        summary = ""
        if self._stage is not None:
            summary += f"{self._stage.upper()} "
        summary = "Torch Compile Profile Result: \n"
        for step, action_metric in self.recorded_metrics.items():
            for action_name, metric in action_metric.items():
                summary += format_str(
                    msg=f"STEP {step}, {action_name} GRAPH BREAK INFO"
                )
                explain_output = str(metric["explain"])
                summary += f"{explain_output} \n\n"
                guard_failure = str(metric["guard_failure"])
                summary += format_str(
                    msg=f"STEP {step}, {action_name} GUARD FAILURE INFO"
                )
                summary += guard_failure

        return summary

    @property
    def dynamo_profiler(self) -> CompileBackendWrapper:
        prof = get_value(CompileBackendWrapper.compile_wrapper_name)
        return prof

    def _prepare_filename(
        self,
        with_hostname: bool = False,
        extension: str = ".log",
    ) -> str:
        return super()._prepare_filename(with_hostname, extension=extension)

    def _set_mark(self):
        os.environ["HAT_WITH_DYNAMO_PROFILER"] = "1"

    def teardown(self) -> None:
        self.describe()
        return super().teardown()
