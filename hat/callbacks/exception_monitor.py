# Copyright (c) Horizon Robotics. All rights reserved.

import logging
import math
import time
from typing import Callable, List, Optional

from hat.core.event import EventStorage
from hat.registry import OBJECT_REGISTRY
from hat.utils.aidi import get_aidi_client
from hat.utils.distributed import rank_zero_only
from hat.utils.package_helper import require_packages
from .callbacks import CallbackMixin

logger = logging.getLogger(__name__)

__all__ = [
    "ExceptionMonitor",
    "default_monitor_func",
]


def default_monitor_func(values: List):
    exception_values = []
    for k, v in values:
        try:
            if math.isnan(v) or math.isinf(v):
                exception_values.append((k, v))
        except Exception:
            pass

    return exception_values


@OBJECT_REGISTRY.register
class ExceptionMonitor(CallbackMixin):
    """Callback used to monitor exception value.

    Args:
        step_monitor_freq: Monitor every `step_monitor_freq` steps.
            If < 1, disable step monitor output.
        epoch_monitor_freq: Monitor every `epoch_monitor_freq` epochs.
            If < 1, disable epoch monitor output.
        monitor_prefix: Monitor info prefix.
        reset_monitor_by: When monitor value is reset. Can be
            'step' or 'monitor' or 'epoch'.
        monitor_func: Function with storage values as inputs,
            monitor exception values.
        monitor_key: key name for monitor in storage. default: `monitor_obj`.
    """

    @require_packages("aidisdk")
    def __init__(
        self,
        step_monitor_freq: Optional[int] = -1,
        epoch_monitor_freq: Optional[int] = -1,
        monitor_prefix: Optional[str] = "",
        reset_monitor_by: Optional[str] = "step",
        monitor_func: Optional[Callable] = default_monitor_func,
        monitor_key: Optional[str] = "monitor_obj",
    ):

        self.step_monitor_freq = step_monitor_freq
        self.epoch_monitor_freq = epoch_monitor_freq
        self.monitor_prefix = monitor_prefix
        assert reset_monitor_by in ("step", "log", "epoch")
        self.reset_monitor_by = reset_monitor_by
        self.monitor_func = monitor_func
        self.monitor_key = monitor_key

    def _reset(self, storage: EventStorage, key: str):
        storage.clear_key(key)

    @rank_zero_only
    def on_loop_begin(self, **kwargs):
        logger.info("Monitor Exception value in Process.")

    @rank_zero_only
    def on_epoch_begin(self, storage: EventStorage, **kwargs):
        self._reset(storage, self.monitor_key)

    @rank_zero_only
    def on_step_end(
        self,
        step_id,
        storage: EventStorage,
        **kwargs,
    ):
        if (
            self.step_monitor_freq > 0
            and (step_id + 1) % self.step_monitor_freq == 0
        ):
            monitor_values = storage.get(self.monitor_key)
            exception_value = self.monitor_func(monitor_values)
            if len(exception_value) > 0:
                self._log(exception_value)
                self._monitor(exception_value)
            if self.reset_monitor_by == "monitor":
                self._reset(storage, self.monitor_key)
        if self.reset_monitor_by == "step":
            self._reset(storage, self.monitor_key)

    @rank_zero_only
    def on_epoch_end(
        self,
        epoch_id,
        storage: EventStorage,
        **kwargs,
    ):
        if (
            self.epoch_monitor_freq > 0
            and (epoch_id + 1) % self.epoch_monitor_freq == 0
        ):
            monitor_values = storage.get(self.monitor_key)
            exception_value = self.monitor_func(monitor_values)
            if len(exception_value) > 0:
                self._log(exception_value)
                self._monitor(exception_value)

    def _log(self, exception_value):
        log_info = self.monitor_prefix
        log_info += " Exception value: "
        for k, v in exception_value:
            if isinstance(v, (int, float)):
                log_info += "%s[%.4f] " % (k, v)
            else:
                log_info += "%s[%s] " % (str(k), str(v))
        logger.warning(log_info)

    def _monitor(self, exception_value):
        value_info = "Exception value: "
        for k, v in exception_value:
            if isinstance(v, (int, float)):
                value_info += "%s[%.4f] " % (k, v)
            else:
                value_info += "%s[%s] " % (str(k), str(v))

        try:
            client = get_aidi_client()
            time_stamp = time.strftime(
                "%Y-%m-%d-%H-%M-%S", time.localtime(int(time.time()))
            )
            monitor_info = f"Process: {self.monitor_prefix}"
            monitor_info += f"\nTime: {time_stamp}\n"
            monitor_info += value_info
            client.notify.message(
                method=["feishu"],
                subject="exception value warning",
                content=monitor_info,
            ).send(to=[client.session.current_user], sendfrom="HAT")
        except Exception as e:
            logger.warning("failed to send exception value: %s" % str(e))
            client = None
