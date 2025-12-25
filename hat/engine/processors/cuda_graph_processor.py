# Copyright (c) Horizon Robotics. All rights reserved.
import logging
import os
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch
from torch.cuda.amp import GradScaler, autocast

from hat.core.event import EventStorage
from hat.profiler.profilers import BaseProfiler, PassThroughProfiler
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list, to_cuda
from hat.utils.channels_last import convert_memory_format
from .processor import BasicBatchProcessor

__all__ = ["CudaGraphBatchProcessor"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class CudaGraphBatchProcessor(BasicBatchProcessor):
    def __init__(
        self,
        need_grad_update: bool,
        batch_transforms: Optional[List] = None,
        inverse_transforms: Optional[List] = None,
        loss_collector: Callable = None,
        enable_amp: bool = False,
        enable_amp_dtype: torch.dtype = torch.float16,
        enable_channels_last: bool = False,
        channels_last_keys: Optional[Sequence[str]] = None,
        grad_scaler: GradScaler = None,
    ):
        super().__init__(
            need_grad_update=need_grad_update,
            batch_transforms=batch_transforms,
            inverse_transforms=inverse_transforms,
            loss_collector=loss_collector,
            enable_amp=enable_amp,
            enable_amp_dtype=enable_amp_dtype,
            enable_apex=False,
            enable_channels_last=enable_channels_last,
            channels_last_keys=channels_last_keys,
            grad_scaler=grad_scaler,
        )

        self.image = None
        self.target = None
        self.graph_model = None

        assert (
            os.environ.get("HAT_USE_CUDAGRAPH", "0") == "1"
        ), "Please set HAT_USE_CUDAGRAPH=1 while use cuda-graph."

    def __call__(
        self,
        step_id: int,
        batch: Union[Tuple[Any], List[Any], object],
        model: torch.nn.Module,
        device: Union[int, None],
        optimizer=None,
        storage: EventStorage = None,
        batch_begin_callback: Callable = None,
        batch_end_callback: Callable = None,
        backward_begin_callback: Callable = None,
        backward_end_callback: Callable = None,
        optimizer_step_begin_callback: Callable = None,
        forward_begin_callback: Callable = None,
        forward_end_callback: Callable = None,
        profiler: Optional[Union[BaseProfiler, str]] = None,
    ):
        assert self.need_grad_update == model.training, (
            "%s vs. %s, set model to training/eval mode by "
            "model.train()/model.eval() when need_grad_update or not"
            % (self.need_grad_update, model.training)
        )

        if batch_begin_callback is not None:
            batch_begin_callback(batch=batch)

        if profiler is not None:
            profiler = PassThroughProfiler()

        if self.need_grad_update:
            with profiler.profile("optimizer_zero_grad"):
                optimizer.zero_grad(set_to_none=True)

        if device is not None:
            batch = to_cuda(batch, device, non_blocking=True)
        else:
            pass

        if self.transforms is not None:
            with profiler.profile("batch_transforms"):
                batch = self.transforms(batch)

        if forward_begin_callback is not None:
            forward_begin_callback(batch=batch, model=model)

        if self.enable_channels_last:
            batch = convert_memory_format(
                batch, self.channels_last_keys, torch.channels_last
            )

        image = batch["img"]
        target = batch.get("labels", None)

        def capture_forward(image, target):
            grad_decorator = (
                torch.enable_grad if self.need_grad_update else torch.no_grad
            )
            auto_cast = autocast(enabled=self.enable_amp, dtype=self.enable_amp_dtype)

            with profiler.profile("model_forward"):
                with auto_cast:
                    with grad_decorator():
                        model_outs = model(image, target)

            if forward_end_callback is not None:
                forward_end_callback(model_outs=model_outs)

            if self.loss_collector is not None:
                losses = self.loss_collector(model_outs)
            else:
                losses = None

            if self.need_grad_update:
                loss = sum([loss for loss in _as_list(losses) if loss is not None])
                assert isinstance(loss, torch.Tensor), type(loss)
                loss_scalar = loss.sum()

                with profiler.profile("model_backward"):
                    if backward_begin_callback:
                        backward_begin_callback()
                    if self.ga_step > 1 and step_id % self.ga_step != 0:
                        with model.no_sync():
                            self.grad_scaler.scale(loss_scalar).backward()
                    else:
                        self.grad_scaler.scale(loss_scalar).backward()
                    if backward_end_callback:
                        backward_end_callback()

            if not self.enable_amp:
                with profiler.profile("optimizer_step"):
                    if optimizer_step_begin_callback is not None:
                        optimizer_step_begin_callback(grad_scaler=self.grad_scaler)
                    self.grad_scaler.step(optimizer)
                    self.grad_scaler.update()

        # 11 is the num of warmup for ddp + cuda_graph
        if step_id <= 11:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                capture_forward(image, target)
            torch.cuda.current_stream().wait_stream(s)
        elif self.graph_model is None:
            torch.cuda.synchronize()
            self.image = image.clone()
            self.target = target.clone()
            self.graph_model = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph_model):
                capture_forward(self.image, self.target)
        else:
            self.image.copy_(image)
            self.target.copy_(target)
            self.graph_model.replay()

        if self.enable_amp:
            # optimizer step cannot be captured in graph while using amp.
            with profiler.profile("optimizer_step"):
                if optimizer_step_begin_callback is not None:
                    optimizer_step_begin_callback(grad_scaler=self.grad_scaler)
                self.grad_scaler.step(optimizer)
                self.grad_scaler.update()

        if self.enable_amp:
            storage.put("grad_scaler", self.grad_scaler.state_dict(), always_dict=True)
