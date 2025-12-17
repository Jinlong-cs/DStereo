# Copyright (c) Horizon Robotics. All rights reserved.
"""
This file is modified from pytorch-lightning.

checking if there are any bottlenecks in your code.
"""
import inspect
import logging
import os
import warnings
from distutils.version import LooseVersion
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd.profiler import EventList, record_function
from torch.profiler import (
    ProfilerAction,
    ProfilerActivity,
    tensorboard_trace_handler,
)
from torch.utils.hooks import RemovableHandle

from hat.registry import OBJECT_REGISTRY
from hat.utils.logger import rank_zero_warn
from .profilers import BaseProfiler

_PROFILER = Union[
    torch.autograd.profiler.profile,
    torch.cuda.profiler.profile,
    torch.autograd.profiler.emit_nvtx,
]


_DEFAULT_RECORD_FUNCS = {
    "optimizer_zero_grad",
    "batch_transforms",
    "model_forward",
    "model_backward",
    "optimizer_step",
}

_SPECIAL_RECORDS_IN_MULTIBATCH = {
    "batch_transforms",
    "model_forward",
    "model_backward",
}

logger = logging.getLogger(__name__)


class RegisterRecordFunction:
    """Add labels for module names around the forward function.

    Args:
        model: Model to record.
    """

    def __init__(self, model: nn.Module) -> None:
        self._model = model
        self._records: Dict[str, record_function] = {}
        self._handles: Dict[str, List["RemovableHandle"]] = {}

    def _start_recording_forward(
        self, _: nn.Module, input: Tensor, record_name: str
    ) -> Tensor:  # noqa D401
        """Module recording hook.

        Note:
            To visualize module in tensorboard, it depends on the
            special support for Pytorch-Lighting in torch-tb-profiler.
            So add the prefix of `[pl][module]` here.
        """
        record = record_function("[pl][module]" + record_name)
        record.__enter__()
        self._records[record_name] = record
        return input

    def _stop_recording_forward(
        self, _: nn.Module, __: Tensor, output: Tensor, record_name: str
    ) -> Tensor:
        self._records[record_name].__exit__(None, None, None)
        return output

    def __enter__(self) -> None:
        for module_name, module in self._model.named_modules():
            if module_name:
                full_name = (
                    f"{type(module).__module__}.{type(module).__name__}"
                )
                record_name = f"{full_name}: {module_name}"
                pre_forward_handle = module.register_forward_pre_hook(
                    partial(
                        self._start_recording_forward, record_name=record_name
                    )
                )
                post_forward_handle = module.register_forward_hook(
                    partial(
                        self._stop_recording_forward, record_name=record_name
                    )
                )

                self._handles[module_name] = [
                    pre_forward_handle,
                    post_forward_handle,
                ]

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        for handles in self._handles.values():
            for h in handles:
                h.remove()
        self._handles = {}


class ScheduleWrapper:
    """ScheduleWrapper.

    This class is used to override the schedule logic from the profiler \
    and perform recording for `optimizer_zero_grad`, `model_forward`、 \
    `model_backward`、`optimizer_step`.
    """

    RECORD_NUM_COUNTER_FORMAT = "_num_%s"
    RECORD_REACHED_END_FORMAT = "_%s_reached_end"

    def __init__(self, schedule: Callable, record_funcs: Set = None) -> None:
        self._schedule = schedule
        self.record_funcs = (
            _DEFAULT_RECORD_FUNCS if record_funcs is None else record_funcs
        )
        self.reset()

    def setup(self, start_action_name: str) -> None:
        if start_action_name not in self.record_funcs and any(
            start_action_name.startswith(k)
            for k in list(_SPECIAL_RECORDS_IN_MULTIBATCH)  # noqa E501
        ):
            self.record_funcs.add(start_action_name)
            setattr(
                self, self.RECORD_NUM_COUNTER_FORMAT % start_action_name, 0
            )  # noqa E501
            setattr(
                self,
                self.RECORD_REACHED_END_FORMAT % start_action_name,
                False,
            )  # noqa E501
        self._start_action_name = start_action_name

    def pre_step(self, current_action: str) -> None:
        self._current_action = current_action

    def reset(self):
        for record in list(self.record_funcs):
            setattr(self, self.RECORD_NUM_COUNTER_FORMAT % record, 0)
            setattr(self, self.RECORD_REACHED_END_FORMAT % record, False)

        # used to stop profiler when
        # `ProfilerAction.RECORD_AND_SAVE` is reached.
        self._current_action: Optional[str] = None
        self._start_action_name: Optional[str] = None

    @property
    def num_step(self) -> int:
        attr_name = self.RECORD_NUM_COUNTER_FORMAT % self._current_action
        return getattr(self, attr_name, 0)

    def _step(self) -> None:
        attr_name = self.RECORD_NUM_COUNTER_FORMAT % self._current_action
        if hasattr(self, attr_name):
            step = getattr(self, attr_name)
            setattr(self, attr_name, step + 1)

    @property
    def has_finished(self) -> bool:
        attr_name = self.RECORD_REACHED_END_FORMAT % self._current_action
        return getattr(self, attr_name, False)

    def __call__(self, num_step: int) -> "ProfilerAction":
        # ignore the provided input. Keep internal state instead.
        if self.has_finished:
            return ProfilerAction.NONE
        self._step()
        action = self._schedule(max(self.num_step, 0))
        if action == ProfilerAction.RECORD_AND_SAVE:
            attr_name = self.RECORD_REACHED_END_FORMAT % self._current_action
            if hasattr(self, attr_name):
                setattr(self, attr_name, True)

        return action


@OBJECT_REGISTRY.register
class PyTorchProfiler(BaseProfiler):
    """This profiler uses PyTorch's Autograd Profiler and lets you inspect the cost of.

    different operators inside your model - both on the CPU and GPU

    Args:
        dirpath: Directory path for the ``filename``.
        filename: If present, filename where the profiler results will be
            saved instead of printing to stdout. The ``.txt`` extension will
            be used automatically.

        group_by_input_shapes: Include operator input shapes and group calls by shape.

        emit_nvtx: Context manager that makes every autograd operation emit an NVTX range
            Run::

                nvprof --profile-from-start off -o trace_name.prof -- <regular command here>

            To visualize, you can either use::

                nvvp trace_name.prof
                torch.autograd.profiler.load_nvprof(path)

        export_to_chrome: Whether to export the sequence of profiled operators for Chrome.
            It will generate a ``.json`` file which can be read by Chrome.

        row_limit: Limit the number of rows in a table, ``-1`` is a special value that
            removes the limit completely.

        sort_by_key: Attribute used to sort entries. By default
            they are printed in the same order as they were registered.
            Valid keys include: ``cpu_time``, ``cuda_time``, ``cpu_time_total``,
            ``cuda_time_total``, ``cpu_memory_usage``, ``cuda_memory_usage``,
            ``self_cpu_memory_usage``, ``self_cuda_memory_usage``, ``count``.

        record_functions: Set of profiled functions which will create a context manager on.
            Any other will be pass through.
        step_function: Profiled function, which means that after the function ends,
            the current iteration is end and next iteration will start.

        record_module_names: Whether to add module names while recording autograd operation.

        profiler_kwargs: Keyword arguments for the PyTorch profiler. This depends on your PyTorch version

        Raises:
            Exception:
                If arg ``sort_by_key`` is not present in ``AVAILABLE_SORT_KEYS``.
                If arg ``schedule`` is not a ``Callable``.
                If arg ``schedule`` does not return a ``torch.profiler.ProfilerAction``.
    """  # noqa

    RECORD_FUNCTIONS = {
        "optimizer_zero_grad",
        "batch_transforms",
        "model_forward",
        "model_backward",
        "optimizer_step",
    }
    STEP_FUNCTION = "optimizer_step"
    AVAILABLE_SORT_KEYS = {
        "cpu_time",
        "cuda_time",
        "cpu_time_total",
        "cuda_time_total",
        "cpu_memory_usage",
        "cuda_memory_usage",
        "self_cpu_memory_usage",
        "self_cuda_memory_usage",
        "count",
    }

    def __init__(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        group_by_input_shapes: bool = False,
        emit_nvtx: bool = False,
        export_to_chrome: bool = True,
        row_limit: int = 20,
        sort_by_key: Optional[str] = None,
        record_functions: Set[str] = None,
        step_function: Optional[str] = None,
        record_module_names: bool = False,
        **profiler_kwargs: Any,
    ) -> None:
        super().__init__(dirpath=dirpath, filename=filename)

        self._group_by_input_shapes = (
            group_by_input_shapes
            and profiler_kwargs.get("record_shapes", False)
        )
        self._emit_nvtx = emit_nvtx
        self._export_to_chrome = export_to_chrome
        self._row_limit = row_limit
        self._sort_by_key = (
            sort_by_key
            or f"{'cuda' if profiler_kwargs.get('use_cuda', False) else 'cpu'}_time_total"  # noqa
        )

        self._record_functions = (
            record_functions if record_functions else self.RECORD_FUNCTIONS
        )
        self._step_function = (
            step_function if step_function else self.STEP_FUNCTION
        )

        self._record_module_names = record_module_names
        self._profiler_kwargs = profiler_kwargs

        self.profiler: Optional[_PROFILER] = None
        self.function_events: Optional["EventList"] = None
        self._register = None
        self._parent_profiler: Optional[_PROFILER] = None
        self._recording_map: Dict[str, record_function] = {}
        self._start_action_name: Optional[str] = None
        self._schedule: Optional[ScheduleWrapper] = None
        self._hat_module: Optional[nn.Module] = None

        self._init_kineto(profiler_kwargs)

        if self._sort_by_key not in self.AVAILABLE_SORT_KEYS:
            raise KeyError(
                f"Found sort_by_key: {self._sort_by_key}. "
                f"Should be within {self.AVAILABLE_SORT_KEYS}. "
            )

    def _init_kineto(self, profiler_kwargs: Any) -> None:
        has_schedule = "schedule" in profiler_kwargs
        self._has_on_trace_ready = "on_trace_ready" in profiler_kwargs

        schedule = profiler_kwargs.get("schedule", None)
        if schedule is not None:
            if not isinstance(schedule, Callable):
                raise TypeError(
                    f"Schedule should be a callable. Found: {schedule}"
                )
            action = schedule(0)
            if not isinstance(action, ProfilerAction):
                raise TypeError(
                    f"Schedule should return "
                    f"a `torch.profiler.ProfilerAction`. Found: {action}"
                )
        self._default_schedule()
        schedule = schedule if has_schedule else self._default_schedule()
        self._schedule = (
            ScheduleWrapper(schedule) if schedule is not None else schedule
        )
        self._profiler_kwargs["schedule"] = self._schedule

        activities = profiler_kwargs.get("activities", None)
        self._profiler_kwargs["activities"] = (
            activities or self._default_activities()
        )
        self._export_to_flame_graph = profiler_kwargs.get(
            "export_to_flame_graph", False
        )
        self._metric = profiler_kwargs.get("metric", "self_cpu_time_total")
        with_stack = (
            profiler_kwargs.get("with_stack", False)
            or self._export_to_flame_graph
        )
        self._profiler_kwargs["with_stack"] = with_stack

        self._warm_up_for_cupti()

    def _should_override_schedule(self) -> bool:
        return False

    @staticmethod
    @lru_cache(1)
    def _default_schedule() -> Optional[callable]:
        # Those schedule defaults allow the profiling
        # overhead to be negligible over training time.
        return torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)

    def _default_activities(self) -> List["ProfilerActivity"]:
        activities = []
        if self._profiler_kwargs.get("use_cpu", True):
            activities.append(ProfilerActivity.CPU)
        if self._profiler_kwargs.get("use_cuda", torch.cuda.is_available()):
            activities.append(ProfilerActivity.CUDA)
        return activities

    def start(self, action_name: str) -> None:
        if self.profiler is None:
            # close profiler if it is already opened.
            # might happen if 2 profilers
            # are created and the first one did not call `describe`
            try:
                torch.autograd._disable_profiler()
            except (AttributeError, RuntimeError):
                pass

            if self._schedule is not None:
                self._schedule.setup(action_name)

            self._create_profilers()

            profiler = self.profiler.__enter__()
            if profiler is not None:
                self.profiler = profiler

            if self._parent_profiler is not None:
                self._parent_profiler.__enter__()

            if self._register is not None:
                self._register.__enter__()

        if (
            self._hat_module is not None
            and self._register is None
            and self._record_module_names
        ):
            self._register = RegisterRecordFunction(self._hat_module)
            self._register.__enter__()

        if (
            self.profiler is not None
            and self._match_any_record_func(
                action_name, self._record_functions
            )
            and action_name not in self._recording_map
        ):
            recording = record_function(action_name)
            recording.__enter__()
            self._recording_map[action_name] = recording

    def stop(self, action_name: str) -> None:
        if action_name in self._recording_map:
            self._recording_map[action_name].__exit__(None, None, None)
            del self._recording_map[action_name]

        if self._emit_nvtx:
            return

        if self.profiler is not None and action_name.startswith(
            self._step_function
        ):
            if self._schedule is not None:
                self._schedule.pre_step(action_name)

            # the default schedule requires a minimum of 5
            # steps to properly work: `wait=1, warmup=1, active=3`.
            # otherwise, this will raise a `segmentation fault`.
            if self._should_override_schedule():
                warnings.warn(
                    "The PyTorch Profiler default schedule will be "
                    "overridden as there is not enough "
                    "steps to properly record traces."
                )
                self._schedule = None
                self.profiler.schedule = (
                    torch.profiler.profiler._default_schedule_fn
                )

            def on_trace_ready(profiler):
                if self.dirpath is not None:
                    if self._export_to_chrome:
                        handler = tensorboard_trace_handler(
                            dir_name=self.dirpath,
                            worker_name=self._prepare_filename(
                                with_hostname=True, extension=None
                            ),
                            # use_gzip=True,
                        )
                        handler(profiler)

                    if self._export_to_flame_graph:
                        path = os.path.join(
                            self.dirpath,
                            self._prepare_filename(
                                with_hostname=True, extension=".stack"
                            ),
                        )
                        profiler.export_stacks(path, metric=self._metric)
                else:
                    rank_zero_warn(
                        "The PyTorchProfiler failed to export "
                        "trace as `dirpath` is None"
                    )

            if not self._has_on_trace_ready:
                self.profiler.on_trace_ready = on_trace_ready

            if self._schedule is not None:
                self.profiler.step_num = self._schedule.num_step
            self.profiler.step()

            if self._hat_module is not None and LooseVersion(
                torch.__version__.split("+")[0]
            ) == LooseVersion("1.10.2"):
                """
                Note:
                    Here is way to solve the problem of not displaying
                    `Module view` in tensorboard in torch1.10.2.
                    Need to add the `Framework=pytorch-lightning` here.
                """

                self.profiler.add_metadata("Framework", "pytorch-lightning")

    def summary(self) -> str:
        if not self._profiler_kwargs.get("enabled", True) or self._emit_nvtx:
            return ""

        self._delete_profilers()

        if not self.function_events:
            return ""
        data = self.function_events.key_averages(
            group_by_input_shapes=self._group_by_input_shapes
        )
        table = data.table(
            sort_by=self._sort_by_key, row_limit=self._row_limit
        )

        recorded_stats = {"records": table}
        return self._stats_to_str(recorded_stats)

    def _create_profilers(self) -> None:
        if self.profiler is not None:
            return

        if self._emit_nvtx:
            if self._parent_profiler is None:
                self._parent_profiler = torch.cuda.profiler.profile()
            self.profiler = self._create_profiler(
                torch.autograd.profiler.emit_nvtx
            )
        else:
            self._parent_profiler = None
            self.profiler = self._create_profiler(torch.profiler.profile)

    def _create_profiler(self, profiler: Type[_PROFILER]) -> _PROFILER:
        init_parameters = inspect.signature(profiler.__init__).parameters
        kwargs = {
            k: v
            for k, v in self._profiler_kwargs.items()
            if k in init_parameters
        }
        return profiler(**kwargs)

    def _cache_functions_events(self) -> None:
        if self._emit_nvtx:
            return
        if isinstance(self.profiler, torch.profiler.profile):
            self.function_events = self.profiler.events()
        elif isinstance(self.profiler, torch.autograd.profiler.profile):
            self.function_events = self.profiler.function_events

    def _delete_profilers(self) -> None:
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            self._cache_functions_events()
            self.profiler = None

        if self._schedule is not None:
            self._schedule.reset()

        if self._parent_profiler is not None:
            self._parent_profiler.__exit__(None, None, None)
            self._parent_profiler = None

        if self._register is not None:
            self._register.__exit__(None, None, None)
            self._register = None

    def teardown(self) -> None:
        self._delete_profilers()

        for k in self._recording_map:
            self.stop(k)
        self._recording_map = {}

        super().teardown()

    def _warm_up_for_cupti(self):
        """Warmup profile to fix torch profile bug in torch 1.13.0.

        Note:
            According to https://github.com/pytorch/pytorch/issues/99734
            and https://github.com/pytorch/kineto/issues/756, the torch profile
            has bugs torch 1.13.0, here is a way to fix.
        """
        if LooseVersion(torch.__version__) >= LooseVersion("1.13.0"):
            with torch.autograd.profiler.profile(
                enabled=True,
                use_cuda=True,
                use_kineto=True,
            ) as _:
                logger.info("Running dummy profiler warmup for CUPTI.")

    def set_model(self, model):
        self._hat_module = model
