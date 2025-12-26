# Copyright (c) Horizon Robotics. All rights reserved.

import logging
import os
import time
from typing import Iterable, Optional

import torch

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list
from hat.utils.distributed import get_dist_info
from .callbacks import CallbackMixin
from .tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


def _read_cpu_times():
    try:
        with open("/proc/stat", "r") as f:
            line = f.readline()
        if not line.startswith("cpu "):
            return None
        parts = line.strip().split()[1:]
        values = [int(p) for p in parts]
        if len(values) < 5:
            return None
        idle = values[3] + values[4]  # idle + iowait
        total = sum(values)
        return total, idle
    except Exception:
        return None


def _read_meminfo():
    try:
        info = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                key, rest = line.split(":", 1)
                val = rest.strip().split()[0]
                info[key] = int(val)
        return info
    except Exception:
        return None


def _mem_used_ratio():
    info = _read_meminfo()
    if not info:
        return None
    total = info.get("MemTotal")
    if total is None or total == 0:
        return None
    available = info.get("MemAvailable")
    if available is None:
        available = info.get("MemFree", 0) + info.get("Buffers", 0) + info.get("Cached", 0)
    used = total - available
    return float(used) / float(total)


def _avg(vals: Iterable[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(sum(vals)) / float(len(vals))


def _extract_total_loss(model_outs):
    if not isinstance(model_outs, dict):
        return None
    losses = model_outs.get("losses")
    if losses is None:
        indexed = []
        for k, v in model_outs.items():
            if not isinstance(k, str) or not k.startswith("losses_"):
                continue
            idx = k.split("losses_", 1)[-1]
            if idx.isdigit():
                indexed.append((int(idx), v))
        if indexed:
            losses = [v for _, v in sorted(indexed, key=lambda x: x[0])]
        else:
            return None
    if isinstance(losses, torch.Tensor):
        return losses
    if isinstance(losses, (list, tuple)):
        loss_list = [loss for loss in losses if loss is not None]
        if not loss_list:
            return None
        return sum(loss_list)
    return None


@OBJECT_REGISTRY.register
class SystemMonitor(CallbackMixin):
    def __init__(
        self,
        save_dir: str,
        log_interval: int = 20,
        mode: str = "train",
        log_system: bool = True,
        log_time: bool = True,
        log_train_loss: bool = True,
        log_val_loss: bool = True,
        gpu_device_ids: Optional[Iterable[int]] = None,
        overwrite: bool = False,
    ):
        self.save_dir = save_dir
        self.log_interval = int(max(1, log_interval))
        self.mode = mode
        self.log_system = bool(log_system)
        self.log_time = bool(log_time)
        self.log_train_loss = bool(log_train_loss)
        self.log_val_loss = bool(log_val_loss)
        self.gpu_device_ids = list(gpu_device_ids) if gpu_device_ids else None
        self.overwrite = overwrite

        self._writer = None
        self._step_start_time = None
        self._data_ready_time = None
        self._last_train_loss = None
        self._cpu_prev = None

        self._nvml_inited = False
        self._nvml_available = False
        self._nvml_warned = False
        self._nvml = None

    def _is_rank0(self):
        rank, _ = get_dist_info()
        return rank == 0

    def _should_log(self, global_step_id):
        if global_step_id is None:
            return False
        return (global_step_id + 1) % self.log_interval == 0

    def _get_gpu_ids(self):
        if self.gpu_device_ids:
            return self.gpu_device_ids
        if not torch.cuda.is_available():
            return []
        return [torch.cuda.current_device()]

    def _gpu_mem_used_ratio(self):
        if not torch.cuda.is_available():
            return None
        ratios = []
        for dev in self._get_gpu_ids():
            try:
                free, total = torch.cuda.mem_get_info(dev)
                if total > 0:
                    ratios.append(float(total - free) / float(total))
            except Exception:
                continue
        return _avg(ratios)

    def _init_nvml(self):
        if self._nvml_inited:
            return
        self._nvml_inited = True
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._nvml_available = True
        except Exception:
            self._nvml_available = False

    def _gpu_util_percent_avg(self):
        if not torch.cuda.is_available():
            return None
        self._init_nvml()
        if not self._nvml_available:
            if not self._nvml_warned:
                logger.warning("pynvml not available; skip gpu/util_percent_avg.")
                self._nvml_warned = True
            return None
        utils = []
        for dev in self._get_gpu_ids():
            try:
                handle = self._nvml.nvmlDeviceGetHandleByIndex(dev)
                util = self._nvml.nvmlDeviceGetUtilizationRates(handle).gpu
                utils.append(float(util))
            except Exception:
                continue
        return _avg(utils)

    def _cpu_util_percent(self):
        cur = _read_cpu_times()
        if cur is None:
            return None
        if self._cpu_prev is None:
            self._cpu_prev = cur
            return None
        total, idle = cur
        prev_total, prev_idle = self._cpu_prev
        self._cpu_prev = cur
        delta_total = total - prev_total
        delta_idle = idle - prev_idle
        if delta_total <= 0:
            return None
        return 100.0 * (delta_total - delta_idle) / float(delta_total)

    def _log_scalars(self, tag_vals, step):
        if not self._is_rank0():
            return
        if self._writer is None:
            return
        for tag, val in tag_vals.items():
            if val is None:
                continue
            self._writer.add_scalar(tag, float(val), global_step=step)

    def _get_loss_from_metrics(self, metrics):
        if metrics is None:
            return None
        for metric in metrics:
            names, values = metric.get()
            for name, value in zip(_as_list(names), _as_list(values)):
                if str(name).lower() == "loss":
                    return float(value)
        return None

    def log_val_metrics(self, metrics, step):
        if not self.log_val_loss:
            return
        if step is None:
            return
        loss_val = self._get_loss_from_metrics(metrics)
        if loss_val is None:
            return
        self._log_scalars({"val/total_loss": loss_val}, step)

    def on_loop_begin(self, **kwargs):
        if not self._is_rank0():
            return
        if self.overwrite:
            os.makedirs(self.save_dir, exist_ok=True)
        self._writer = SummaryWriter(self.save_dir)

    def on_loop_end(self, **kwargs):
        if self._writer is not None:
            self._writer.close()
        self._writer = None

    def on_step_begin(self, **kwargs):
        if self.mode != "train":
            return
        if self.log_time:
            self._step_start_time = time.time()
            self._data_ready_time = None

    def on_batch_begin(self, **kwargs):
        if self.mode != "train":
            return
        if self.log_time and self._step_start_time is not None:
            self._data_ready_time = time.time()

    def on_batch_end(self, model_outs=None, **kwargs):
        if self.mode != "train":
            return
        if self.log_train_loss:
            self._last_train_loss = _extract_total_loss(model_outs)

    def on_step_end(self, global_step_id=None, **kwargs):
        if self.mode != "train":
            return
        if not self._should_log(global_step_id):
            return
        tag_vals = {}
        if self.log_system:
            tag_vals["gpu/mem_used_ratio"] = self._gpu_mem_used_ratio()
            tag_vals["gpu/util_percent_avg"] = self._gpu_util_percent_avg()
            tag_vals["cpu/util_percent"] = self._cpu_util_percent()
            tag_vals["ram/used_ratio"] = _mem_used_ratio()
        if self.log_time and self._step_start_time is not None:
            now = time.time()
            tag_vals["time/step_time_ms"] = (now - self._step_start_time) * 1000.0
            if self._data_ready_time is not None:
                tag_vals["time/data_time_ms"] = (
                    self._data_ready_time - self._step_start_time
                ) * 1000.0
        if self.log_train_loss and self._last_train_loss is not None:
            tag_vals["train/total_loss"] = float(self._last_train_loss)
        self._log_scalars(tag_vals, global_step_id)

    def on_epoch_end(self, epoch_id=None, global_step_id=None, train_metrics=None, **kwargs):
        if self.mode != "val":
            return
        step = global_step_id if global_step_id is not None else epoch_id
        self.log_val_metrics(train_metrics, step)
