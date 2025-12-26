# Copyright (c) Horizon Robotics. All rights reserved.

import copy
import logging
import os
import random
import time
from typing import Iterable, Optional

import numpy as np
import torch

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list, to_cuda
from hat.utils.checkpoint import load_state_dict
from hat.utils.distributed import get_dist_info
from .callbacks import CallbackMixin
from DStereo.common import disp2rgb

logger = logging.getLogger(__name__)

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    wandb = None


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
        idle = values[3] + values[4]
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
        available = info.get("MemFree", 0) + info.get("Buffers", 0) + info.get(
            "Cached", 0
        )
    used = total - available
    return float(used) / float(total)


def _avg(vals: Iterable[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(sum(vals)) / float(len(vals))


def _extract_losses(model_outs):
    if not isinstance(model_outs, dict):
        return []
    losses = model_outs.get("losses")
    if losses is not None:
        if isinstance(losses, torch.Tensor):
            return [losses]
        if isinstance(losses, (list, tuple)):
            return [loss for loss in losses if loss is not None]
        return []
    indexed = []
    for k, v in model_outs.items():
        if not isinstance(k, str) or not k.startswith("losses_"):
            continue
        idx = k.split("losses_", 1)[-1]
        if idx.isdigit():
            indexed.append((int(idx), v))
    if not indexed:
        return []
    return [v for _, v in sorted(indexed, key=lambda x: x[0])]


def _named_loss_items(losses):
    if not losses:
        return []
    items = [("init_smooth_l1", losses[0])]
    for idx, loss in enumerate(losses[1:]):
        items.append((f"iter_{idx}_weighted_l1", loss))
    return items


@OBJECT_REGISTRY.register
class WandbLogger:
    def __init__(
        self,
        project: str = "dstereo_vis",
        name: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        resume: Optional[str] = None,
        run_id: Optional[str] = None,
        config: Optional[dict] = None,
        log_every_steps: int = 20,
        log_system_metrics: bool = True,
        log_time_metrics: bool = True,
        log_train_loss: bool = True,
        log_train_subloss: bool = True,
        log_lr: bool = True,
    ):
        self.project = project
        self.name = name
        self.tags = list(tags) if tags else None
        self.resume = resume
        self.run_id = run_id
        self.config = config
        self.log_every_steps = int(max(1, log_every_steps))
        self.log_system_metrics = bool(log_system_metrics)
        self.log_time_metrics = bool(log_time_metrics)
        self.log_train_loss = bool(log_train_loss)
        self.log_train_subloss = bool(log_train_subloss)
        self.log_lr = bool(log_lr)

        self._step_start_time = None
        self._data_ready_time = None
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
        return (global_step_id + 1) % self.log_every_steps == 0

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

    def _gpu_mem_used_ratio(self):
        if not torch.cuda.is_available():
            return None
        ratios = []
        for dev in [torch.cuda.current_device()]:
            try:
                free, total = torch.cuda.mem_get_info(dev)
                if total > 0:
                    ratios.append(float(total - free) / float(total))
            except Exception:
                continue
        return _avg(ratios)

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
        for dev in [torch.cuda.current_device()]:
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

    def _log(self, data, step):
        if not self._is_rank0() or wandb is None or wandb.run is None:
            return
        wandb.log(data, step=step)

    def log_metrics(self, metrics, step):
        if not metrics:
            return
        self._log(metrics, step)

    def log_images(self, images, step):
        if not images:
            return
        self._log(images, step)

    def on_loop_begin(self, **kwargs):
        if not self._is_rank0():
            return
        if wandb is None:
            logger.warning("wandb is not available; skip W&B logging.")
            return
        if wandb.run is None:
            init_kwargs = dict(project=self.project, name=self.name)
            if self.tags:
                init_kwargs["tags"] = self.tags
            if self.run_id:
                init_kwargs["id"] = self.run_id
            if self.resume:
                init_kwargs["resume"] = self.resume
            if self.config is not None:
                init_kwargs["config"] = self.config
            wandb.init(**init_kwargs)

    def on_loop_end(self, **kwargs):
        if not self._is_rank0() or wandb is None or wandb.run is None:
            return
        wandb.finish()

    def on_step_begin(self, **kwargs):
        if self.log_time_metrics:
            self._step_start_time = time.time()
            self._data_ready_time = None

    def on_batch_begin(self, **kwargs):
        if self.log_time_metrics and self._step_start_time is not None:
            self._data_ready_time = time.time()

    def on_batch_end(self, model_outs=None, global_step_id=None, **kwargs):
        if not self._should_log(global_step_id):
            return
        if self.log_train_loss:
            losses = _extract_losses(model_outs)
            if losses:
                total = sum([loss for loss in losses if loss is not None])
                if isinstance(total, torch.Tensor):
                    total_val = float(total.detach().cpu().item())
                    self.log_metrics({"train/total_loss": total_val}, global_step_id)
                if self.log_train_subloss:
                    for name, loss in _named_loss_items(losses):
                        if loss is None:
                            continue
                        self.log_metrics(
                            {f"train/loss/{name}": float(loss.detach().cpu().item())},
                            global_step_id,
                        )

    def on_step_end(self, global_step_id=None, optimizer=None, **kwargs):
        if not self._should_log(global_step_id):
            return
        data = {}
        if self.log_lr and optimizer is not None:
            try:
                data["lr"] = float(optimizer.param_groups[-1]["lr"])
            except Exception:
                pass
        if self.log_system_metrics:
            data["gpu/mem_used_ratio"] = self._gpu_mem_used_ratio()
            data["gpu/util_percent_avg"] = self._gpu_util_percent_avg()
            data["cpu/util_percent"] = self._cpu_util_percent()
            data["ram/used_ratio"] = _mem_used_ratio()
        if self.log_time_metrics and self._step_start_time is not None:
            now = time.time()
            data["time/step_time_ms"] = (now - self._step_start_time) * 1000.0
            if self._data_ready_time is not None:
                data["time/data_time_ms"] = (
                    self._data_ready_time - self._step_start_time
                ) * 1000.0
        if data:
            self.log_metrics(data, global_step_id)

    def log_val_metrics(self, metrics, step):
        if metrics is None:
            return
        data = {}
        for metric in metrics:
            names, values = metric.get()
            for name, value in zip(_as_list(names), _as_list(values)):
                key = str(name).lower()
                if key == "loss":
                    tag = "val/total_loss"
                elif key == "epe":
                    tag = "val/epe"
                else:
                    tag = f"val/{key}"
                data[tag] = float(value)
        if data:
            self.log_metrics(data, step)


@OBJECT_REGISTRY.register
class WandbCallback(CallbackMixin):
    def __init__(self, **kwargs):
        self._wandb_logger = WandbLogger(**kwargs)

    def _call_logger(self, method_name, **kwargs):
        method = getattr(self._wandb_logger, method_name, None)
        if method is not None:
            method(**kwargs)

    def on_loop_begin(self, **kwargs):
        self._call_logger("on_loop_begin", **kwargs)

    def on_loop_end(self, **kwargs):
        self._call_logger("on_loop_end", **kwargs)

    def on_epoch_begin(self, **kwargs):
        self._call_logger("on_epoch_begin", **kwargs)

    def on_epoch_end(self, **kwargs):
        self._call_logger("on_epoch_end", **kwargs)

    def on_step_begin(self, **kwargs):
        self._call_logger("on_step_begin", **kwargs)

    def on_step_end(self, **kwargs):
        self._call_logger("on_step_end", **kwargs)

    def on_batch_begin(self, **kwargs):
        self._call_logger("on_batch_begin", **kwargs)

    def on_batch_end(self, **kwargs):
        self._call_logger("on_batch_end", **kwargs)

    def on_backward_begin(self, **kwargs):
        self._call_logger("on_backward_begin", **kwargs)

    def on_backward_end(self, **kwargs):
        self._call_logger("on_backward_end", **kwargs)

    def on_optimizer_step_begin(self, **kwargs):
        self._call_logger("on_optimizer_step_begin", **kwargs)

    def on_forward_begin(self, **kwargs):
        self._call_logger("on_forward_begin", **kwargs)

    def on_forward_end(self, **kwargs):
        self._call_logger("on_forward_end", **kwargs)

    def log_val_metrics(self, metrics, step):
        self._wandb_logger.log_val_metrics(metrics, step)

    def log_metrics(self, metrics, step):
        self._wandb_logger.log_metrics(metrics, step)

    def log_images(self, images, step):
        self._wandb_logger.log_images(images, step)


@OBJECT_REGISTRY.register
class FixedSampleTracker(CallbackMixin):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        collate_fn,
        fixed_train_index: int = 0,
        fixed_val_index: int = 0,
        vis_num_samples: int = 5,
        vis_strategy: str = "fixed",
        vis_seed: int = 0,
        vis_train_indices=None,
        vis_val_indices=None,
        pretrained_ckpt: str = None,
        log_pretrained_baseline: bool = True,
        vis_every_steps: int = 0,
        vis_every_epochs: int = 1,
        vis_at_start: bool = False,
        maxdisp: float = 96.0,
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.collate_fn = collate_fn
        self.fixed_train_index = int(fixed_train_index)
        self.fixed_val_index = int(fixed_val_index)
        self.vis_num_samples = int(max(1, vis_num_samples))
        self.vis_strategy = str(vis_strategy).lower()
        self.vis_seed = int(vis_seed)
        self.vis_train_indices = self._normalize_indices(vis_train_indices)
        self.vis_val_indices = self._normalize_indices(vis_val_indices)
        self.pretrained_ckpt = pretrained_ckpt
        self.log_pretrained_baseline = bool(log_pretrained_baseline)
        self.vis_every_steps = int(vis_every_steps)
        self.vis_every_epochs = int(vis_every_epochs)
        self.vis_at_start = bool(vis_at_start)
        self.maxdisp = float(maxdisp)

        if self.vis_strategy not in ("fixed", "random"):
            raise ValueError(
                f"Unsupported vis_strategy={self.vis_strategy}, use fixed|random."
            )

        if self.vis_seed >= 0:
            self._rng_train = random.Random(self.vis_seed)
            self._rng_val = random.Random(self.vis_seed + 1)
        else:
            self._rng_train = random.Random()
            self._rng_val = random.Random()

        self._train_len = None
        self._val_len = None
        self._baseline_model = None
        self._baseline_ready = False
        self._logged_start = False

    def _has_train(self):
        return self.train_dataset is not None

    def _is_rank0(self):
        rank, _ = get_dist_info()
        return rank == 0

    def _infer_device(self, model):
        model_to_use = model.module if hasattr(model, "module") else model
        for param in model_to_use.parameters():
            return param.device
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    def _normalize_indices(self, indices):
        if indices is None:
            return None
        if isinstance(indices, str):
            parts = indices.replace(",", " ").split()
            indices = [int(p) for p in parts if p]
        elif isinstance(indices, (list, tuple)):
            indices = [int(i) for i in indices]
        else:
            indices = [int(indices)]
        return indices if indices else None

    def _get_dataset_len(self, dataset, name):
        try:
            dataset_len = len(dataset)
        except Exception as exc:
            raise ValueError(f"{name} dataset has no valid length: {exc}") from exc
        if dataset_len <= 0:
            raise ValueError(f"{name} dataset is empty.")
        return dataset_len

    def _validate_indices(self, indices, dataset_len, name):
        if not indices:
            return
        for idx in indices:
            if idx < 0 or idx >= dataset_len:
                raise ValueError(
                    f"{name} index {idx} out of range [0, {dataset_len - 1}]."
                )

    def _expand_indices(self, indices):
        if not indices:
            return []
        if len(indices) >= self.vis_num_samples:
            return list(indices[: self.vis_num_samples])
        repeats = (self.vis_num_samples + len(indices) - 1) // len(indices)
        expanded = (list(indices) * repeats)[: self.vis_num_samples]
        return expanded

    def _sample_random_indices(self, dataset_len, rng):
        if dataset_len >= self.vis_num_samples:
            return rng.sample(range(dataset_len), self.vis_num_samples)
        return [rng.randrange(dataset_len) for _ in range(self.vis_num_samples)]

    def _resolve_indices(self, dataset_len, fixed_index, override_indices, rng):
        if override_indices:
            return self._expand_indices(override_indices)
        if self.vis_strategy == "random":
            return self._sample_random_indices(dataset_len, rng)
        fixed_index = fixed_index % dataset_len
        return [(fixed_index + offset) % dataset_len for offset in range(self.vis_num_samples)]

    def _get_batch(self, dataset, index):
        sample = dataset[index]
        if self.collate_fn is None:
            return sample
        return self.collate_fn([sample])

    def _build_baseline(self, model, device):
        if self._baseline_ready or not self.log_pretrained_baseline:
            return
        if not self.pretrained_ckpt:
            logger.warning("Baseline ckpt not provided; skip pretrained baseline.")
            self._baseline_ready = True
            return
        model_to_copy = model.module if hasattr(model, "module") else model
        try:
            baseline = copy.deepcopy(model_to_copy)
        except Exception as exc:
            logger.warning("Baseline model deepcopy failed: %s", exc)
            self._baseline_ready = True
            return
        for param in baseline.parameters():
            param.requires_grad_(False)
        baseline.eval()
        try:
            load_state_dict(
                baseline,
                path_or_dict=self.pretrained_ckpt,
                map_location="cpu",
                allow_miss=True,
                ignore_extra=True,
                ignore_tensor_shape=True,
                verbose=False,
            )
        except Exception as exc:
            logger.warning("Baseline ckpt load failed: %s", exc)
            self._baseline_ready = True
            return
        if device is not None:
            baseline.to(device)
        self._baseline_model = baseline
        self._baseline_ready = True

    def on_loop_begin(self, **kwargs):
        if not self._is_rank0():
            return
        if wandb is None or wandb.run is None:
            return
        if self._has_train():
            self._train_len = self._get_dataset_len(self.train_dataset, "train")
        self._val_len = self._get_dataset_len(self.val_dataset, "val")
        if self._has_train():
            self._validate_indices(self.vis_train_indices, self._train_len, "train")
        self._validate_indices(self.vis_val_indices, self._val_len, "val")
        if self.vis_strategy == "fixed":
            if self._has_train():
                self._validate_indices(
                    [self.fixed_train_index], self._train_len, "train"
                )
            self._validate_indices([self.fixed_val_index], self._val_len, "val")
        if (
            self.vis_at_start
            and not self._logged_start
            and kwargs.get("model") is not None
        ):
            model = kwargs.get("model")
            device = self._infer_device(model)
            if getattr(device, "type", "cuda") != "cuda":
                return
            self._build_baseline(model, device)
            val_indices = self._resolve_indices(
                self._val_len,
                self.fixed_val_index,
                self.vis_val_indices,
                self._rng_val,
            )
            self._log_samples(
                self.val_dataset,
                val_indices,
                model,
                device,
                "val/current_raw",
                0,
                0,
                tag_prefix="val_vis/current_raw",
            )
            if self._baseline_model is not None and self.log_pretrained_baseline:
                self._log_samples(
                    self.val_dataset,
                    val_indices,
                    self._baseline_model,
                    device,
                    "val/pretrained_raw",
                    0,
                    0,
                    tag_prefix="val_vis/pretrained_raw",
                )
            self._logged_start = True

    def _should_vis(self, global_step_id, epoch_id):
        if self.vis_every_steps > 0 and global_step_id is not None:
            return (global_step_id + 1) % self.vis_every_steps == 0
        if self.vis_every_epochs > 0 and epoch_id is not None:
            return (epoch_id + 1) % self.vis_every_epochs == 0
        return False

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def _get_first(self, x):
        if isinstance(x, (list, tuple)):
            return x[0]
        return x

    def _stack_panel(self, images):
        imgs = [img for img in images if img is not None and getattr(img, "ndim", 0) == 3]
        if not imgs:
            return None
        min_h = min(img.shape[0] for img in imgs)
        min_w = min(img.shape[1] for img in imgs)
        if min_h <= 0 or min_w <= 0:
            return None
        imgs = [img[:min_h, :min_w] for img in imgs]
        return np.concatenate(imgs, axis=1)

    def _run_forward(self, model, batch, device):
        model_to_use = model.module if hasattr(model, "module") else model
        was_training = model_to_use.training
        model_to_use.eval()
        with torch.no_grad():
            batch_cuda = to_cuda(batch, device=device)
            outputs = model_to_use(batch_cuda)
        if was_training:
            model_to_use.train()
        return outputs, batch_cuda

    def _extract_pred(self, outputs):
        if isinstance(outputs, dict):
            pred = outputs.get("pred_disps")
            if pred is not None:
                return pred
        if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            return outputs[0]
        return None

    def _format_sample_id(self, batch, index):
        if not isinstance(batch, dict):
            return str(index)
        sample_idx = self._get_first(batch.get("sample_idx"))
        left_name = self._get_first(batch.get("left_img_name"))
        dataset_name = self._get_first(batch.get("dataset_name"))
        parts = []
        if dataset_name:
            parts.append(str(dataset_name))
        if left_name:
            parts.append(os.path.basename(str(left_name)))
        elif sample_idx is not None:
            parts.append(f"local={sample_idx}")
        if index is not None:
            parts.append(f"idx={index}")
        return " | ".join(parts) if parts else str(index)

    def _build_caption(self, split, index, step, epoch_id, epe=None):
        parts = [split]
        if index is not None:
            parts.append(f"idx={index}")
        if epoch_id is not None:
            parts.append(f"epoch={epoch_id}")
        if step is not None:
            parts.append(f"step={step}")
        if epe is not None:
            parts.append(f"epe={epe:.4f}")
        return " ".join(parts)

    def _build_row(self, batch, pred, split, step, epoch_id, index):
        left = self._get_first(batch.get("left_img")) if isinstance(batch, dict) else None
        right = self._get_first(batch.get("right_img")) if isinstance(batch, dict) else None
        gt = batch.get("gt_disp") if isinstance(batch, dict) else None

        pred_np = self._to_numpy(self._get_first(pred))
        if pred_np is None:
            return None, None, None

        if pred_np.ndim == 3:
            pred_np = pred_np[0]

        left_np = None
        right_np = None
        left_img = None
        right_img = None
        if left is not None:
            left_np = self._to_numpy(left)
            if left_np.ndim == 3:
                left_img = wandb.Image(left_np[:, :, ::-1])
        if right is not None:
            right_np = self._to_numpy(right)
            if right_np.ndim == 3:
                right_img = wandb.Image(right_np[:, :, ::-1])

        pred_vis = disp2rgb(pred_np, self.maxdisp, 0.1)
        pred_img = wandb.Image(pred_vis[:, :, ::-1])

        gt_img = None
        err_img = None
        gt_vis = None
        err_vis = None
        epe = None
        if gt is not None:
            gt_np = self._to_numpy(self._get_first(gt))
            if gt_np.ndim == 3:
                gt_np = gt_np[0]
            gt_vis = disp2rgb(gt_np, self.maxdisp, 0.1)
            gt_img = wandb.Image(gt_vis[:, :, ::-1])
            err = abs(pred_np - gt_np)
            err_vis = disp2rgb(err, self.maxdisp, 0.1)
            err_img = wandb.Image(err_vis[:, :, ::-1])
            valid = (gt_np > 0) & (gt_np < self.maxdisp)
            if valid.any():
                epe = float(err[valid].mean())

        caption = self._build_caption(split, index, step, epoch_id, epe)
        sample_id = self._format_sample_id(batch, index)
        panel = self._stack_panel([left_np, right_np, pred_vis, gt_vis, err_vis])
        return (
            [left_img, right_img, pred_img, gt_img, err_img, caption, sample_id, epe],
            panel,
            caption,
        )

    def _log_samples(
        self,
        dataset,
        indices,
        model,
        device,
        split,
        step,
        epoch_id,
        tag_prefix=None,
    ):
        if not indices:
            return
        tag_prefix = tag_prefix or split
        table = wandb.Table(
            columns=["left", "right", "pred", "gt", "err", "caption", "sample_id", "epe"]
        )
        media = {}
        for slot, idx in enumerate(indices):
            try:
                batch = self._get_batch(dataset, idx)
            except Exception as exc:
                logger.warning("%s sample fetch failed (idx=%s): %s", split, idx, exc)
                continue
            outputs, _ = self._run_forward(model, batch, device)
            pred = self._extract_pred(outputs)
            if pred is None:
                continue
            row, panel, caption = self._build_row(
                batch, pred, split, step, epoch_id, idx
            )
            if row is not None:
                table.add_data(*row)
            if panel is not None:
                media[f"{tag_prefix}/vis_samples/{slot}"] = wandb.Image(
                    panel[:, :, ::-1], caption=caption
                )
        if table.data:
            wandb.log({f"{tag_prefix}/vis_samples": table}, step=step)
        if media:
            wandb.log(media, step=step)

    def on_step_end(self, model=None, device=None, global_step_id=None, epoch_id=None, **kwargs):
        if not self._is_rank0():
            return
        if wandb is None or wandb.run is None:
            return
        if not self._should_vis(global_step_id, None):
            return
        if model is None or device is None:
            return
        self._build_baseline(model, device)
        if self._has_train() and self._train_len is None:
            self._train_len = self._get_dataset_len(self.train_dataset, "train")
        if self._val_len is None:
            self._val_len = self._get_dataset_len(self.val_dataset, "val")
        if self._has_train():
            train_indices = self._resolve_indices(
                self._train_len,
                self.fixed_train_index,
                self.vis_train_indices,
                self._rng_train,
            )
        val_indices = self._resolve_indices(
            self._val_len,
            self.fixed_val_index,
            self.vis_val_indices,
            self._rng_val,
        )
        if self._has_train():
            self._log_samples(
                self.train_dataset,
                train_indices,
                model,
                device,
                "train/current_raw",
                global_step_id,
                epoch_id,
                tag_prefix="train_vis/current_raw",
            )
        self._log_samples(
            self.val_dataset,
            val_indices,
            model,
            device,
            "val/current_raw",
            global_step_id,
            epoch_id,
            tag_prefix="val_vis/current_raw",
        )
        if self._baseline_model is not None and self.log_pretrained_baseline:
            self._log_samples(
                self.val_dataset,
                val_indices,
                self._baseline_model,
                device,
                "val/pretrained_raw",
                global_step_id,
                epoch_id,
                tag_prefix="val_vis/pretrained_raw",
            )

    def on_epoch_end(self, model=None, device=None, global_step_id=None, epoch_id=None, **kwargs):
        if not self._is_rank0():
            return
        if wandb is None or wandb.run is None:
            return
        if self.vis_every_steps > 0:
            return
        if not self._should_vis(None, epoch_id):
            return
        if model is None or device is None:
            return
        step = global_step_id if global_step_id is not None else epoch_id
        self._build_baseline(model, device)
        if self._has_train() and self._train_len is None:
            self._train_len = self._get_dataset_len(self.train_dataset, "train")
        if self._val_len is None:
            self._val_len = self._get_dataset_len(self.val_dataset, "val")
        if self._has_train():
            train_indices = self._resolve_indices(
                self._train_len,
                self.fixed_train_index,
                self.vis_train_indices,
                self._rng_train,
            )
        val_indices = self._resolve_indices(
            self._val_len,
            self.fixed_val_index,
            self.vis_val_indices,
            self._rng_val,
        )
        if self._has_train():
            self._log_samples(
                self.train_dataset,
                train_indices,
                model,
                device,
                "train/current_raw",
                step,
                epoch_id,
                tag_prefix="train_vis/current_raw",
            )
        self._log_samples(
            self.val_dataset,
            val_indices,
            model,
            device,
            "val/current_raw",
            step,
            epoch_id,
            tag_prefix="val_vis/current_raw",
        )
        if self._baseline_model is not None and self.log_pretrained_baseline:
            self._log_samples(
                self.val_dataset,
                val_indices,
                self._baseline_model,
                device,
                "val/pretrained_raw",
                step,
                epoch_id,
                tag_prefix="val_vis/pretrained_raw",
            )
