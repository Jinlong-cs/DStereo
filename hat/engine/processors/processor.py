# Copyright (c) Horizon Robotics. All rights reserved.

import logging
from abc import ABC, abstractmethod
from decimal import localcontext
from distutils.version import LooseVersion
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch
from horizon_plugin_pytorch.qtensor import QTensor
from torch.cuda.amp import GradScaler, autocast

from hat.core.compose_transform import Compose
from hat.core.event import EventStorage
from hat.profiler.profilers import BaseProfiler, PassThroughProfiler
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import (
    _as_list,
    _call_as_tensor,
    flatten,
    regroup,
    to_cuda,
)
from hat.utils.channels_last import convert_memory_format
from hat.utils.deterministic import maybe_cast_batch_for_deterministic
from hat.utils.logger import MSGColor, format_msg
from hat.utils.package_helper import check_packages_available

try:
    import apex
except ImportError:
    apex = None

try:
    import deepspeed
except ImportError:
    deepspeed = None


__all__ = ["BatchProcessorMixin", "BasicBatchProcessor", "MultiBatchProcessor"]

logger = logging.getLogger(__name__)


class BatchProcessorMixin(ABC):
    """Batch Processor Interface."""

    @abstractmethod
    def __call__(
        self,
        batch: Union[Tuple[Any], List[Any], object],
        model: torch.nn.Module,
        device: Union[int, None],
        optimizer=None,
        batch_begin_callback: Callable = None,
        batch_end_callback: Callable = None,
        backward_begin_callback: Callable = None,
        backward_end_callback: Callable = None,
        optimizer_step_begin_callback: Callable = None,
        forward_begin_callback: Callable = None,
        forward_end_callback: Callable = None,
    ):
        pass

    def deepspeed_config_check(self):
        if self.ga_step > 1:
            raise ValueError(
                "When using deepspeed, grad_accumulation_step should be 1 "
                f"in {__class__.__name__}, and if you want to use "
                "gradient accumulation, please configure it in "
                "deepspeed config file."
            )
        if self.enable_amp:
            msg = (
                "When both enable amp and use deepspeed, "
                "auto cast will be triggered, this may cause training "
                "efficiency reduction, please make sure you know what "
                "you are doing. "
                "Suggest to set enable_amp to False when use deepspeed."
            )

            logger.error(format_msg(msg, MSGColor.RED))
        if self.enable_apex:
            raise ValueError("Apex is not compatibility with deepspeed")
        if hasattr(self, "delay_sync") and self.delay_sync:
            raise ValueError("delay_sync is not supported with deepspeed now")


@OBJECT_REGISTRY.register
class BasicBatchProcessor(BatchProcessorMixin):  # noqa: D205,D400
    """
    Processor dealing with `(inputs, target)` batch, and the model output is a
    `(losses, preds)` pair.

    It is suitable for training (need_grad_update) or validation
    (not need_grad_update).

    Args:
        need_grad_update: Whether need gradient update, True for training,
            False for Validation.
        batch_transforms: Config of batch transforms.
        inverse_transforms: Config of transforms,
            used for infer results transform inverse.
        loss_collector: A callable object used to collect loss Tensors in model
            outputs.
        enable_amp: Whether training with `Automatic Mixed Precision`.
        enable_amp_dtype: The dtype of amp, float16 or bfloat16.
        enabel_apex: Whether training with `Apex`.
        enable_channels_last: Whether to use `channel_last` memory_format.
        channels_last_keys: Keys in batch need to convert to channels_last.
            if None, all 4d-tensor in batch data will convert to channels_last.
        grad_scaler: An instance ``scaler`` of :class:`GradScaler`
            helps perform the steps of gradient scaling conveniently.
        grad_accumulation_step: The step of grad accumulation.
            Gradient accumulation refers to multiple backwards passes are
            performed before updating the parameters. The goal is to update
            the model's parameters based on different steps,
            instead of performing an update after every single batch.
    """

    def __init__(
        self,
        need_grad_update: bool,
        batch_transforms: Optional[List] = None,
        inverse_transforms: Optional[List] = None,
        loss_collector: Callable = None,
        enable_amp: bool = False,
        enable_amp_dtype: torch.dtype = torch.float16,
        enable_apex: bool = False,
        enable_channels_last: bool = False,
        channels_last_keys: Optional[Sequence[str]] = None,
        grad_scaler: torch.cuda.amp.GradScaler = None,
        grad_accumulation_step: int = 1,
    ):
        if need_grad_update:
            assert (
                loss_collector is not None
            ), "Provide `loss_collector` when need_grad_update"
            assert callable(loss_collector)
        if enable_amp and enable_apex:
            raise RuntimeError("enable_amp and enable_apex cannot be true together.")
        if enable_apex and apex is None:
            check_packages_available("apex")

        if enable_amp_dtype == torch.bfloat16:
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("current gpu devices do not support bfloat16.")

        self.need_grad_update = need_grad_update
        self.enable_amp_dtype = enable_amp_dtype
        self.enable_apex = enable_apex
        self.enable_channels_last = enable_channels_last
        self.channels_last_keys = channels_last_keys
        self.loss_collector = loss_collector
        if grad_scaler is not None:
            self.grad_scaler = grad_scaler
        else:
            self.grad_scaler = GradScaler(enabled=enable_amp)
        self.ga_step = grad_accumulation_step
        self.enable_amp = self.grad_scaler.is_enabled()
        if enable_amp:
            assert self.enable_amp, (
                "When grad_scaler is not None, enable_amp does not work."
                "You set enable_amp is {}, but the enable_amp of "
                "grad_scaler is {}. Please check your config!!"
            ).format(enable_amp, self.enable_amp)
        if batch_transforms:
            if isinstance(batch_transforms, (list, tuple)):
                batch_transforms = Compose(batch_transforms)  # noqa
            self.transforms = batch_transforms
        else:
            self.transforms = None
        self.inverse_transforms = inverse_transforms
        self.use_deepspeed = None

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
        if self.use_deepspeed is None and deepspeed is not None:
            self.use_deepspeed = isinstance(
                model, (deepspeed.DeepSpeedEngine, deepspeed.InferenceEngine)
            )
            if self.use_deepspeed:
                self.deepspeed_config_check()

        if batch_begin_callback is not None:
            batch_begin_callback(batch=batch)

        if profiler is None:
            profiler = PassThroughProfiler()

        # 0. reset grad
        if (
            not self.use_deepspeed
            and self.need_grad_update
            and step_id % self.ga_step == 0
        ):
            with profiler.profile("optimizer_zero_grad"):
                optimizer.zero_grad(set_to_none=True)

        if device is not None:
            batch = to_cuda(batch, device, non_blocking=True)
        else:
            # run on cpu
            pass

        if self.transforms is not None:
            with profiler.profile("batch_transforms"):
                batch = self.transforms(batch)

        # 1. forward
        if forward_begin_callback is not None:
            forward_begin_callback(batch=batch, model=model)

        if self.enable_channels_last:
            batch = convert_memory_format(
                batch, self.channels_last_keys, torch.channels_last
            )

        batch = maybe_cast_batch_for_deterministic(batch)

        grad_decorator = torch.enable_grad if self.need_grad_update else torch.no_grad
        if not self.enable_apex:
            auto_cast = autocast(enabled=self.enable_amp, dtype=self.enable_amp_dtype)
        else:
            auto_cast = localcontext()
        with profiler.profile("model_forward"):
            with auto_cast:
                with grad_decorator():
                    if (step_id + 1) % self.ga_step != 0:
                        # Only work while using apex.
                        if hasattr(model, "disable_allreduce"):
                            model.disable_all_reduce()
                    else:
                        if hasattr(model, "enable_allreduce"):
                            model.enable_allreduce()
                    # model outputs can be in any format
                    model_outs = model(*_as_list(batch))

        if self.inverse_transforms is not None:
            model_outs = self.inverse_transforms(model_outs, batch)

        if forward_end_callback is not None:
            forward_end_callback(model_outs=model_outs)

        # 2. filter out loss Tensors in model outputs
        if self.loss_collector is not None:
            losses = self.loss_collector(model_outs)
        else:
            losses = None

        # 2. backward & step
        if self.need_grad_update:
            # Not allow to backward each loss independently, so sum them
            loss = sum([loss for loss in _as_list(losses) if loss is not None])
            assert isinstance(loss, torch.Tensor), type(loss)
            # mean of grad accumulation step
            loss_scalar = loss.sum() / self.ga_step

            # when grad_scaler is not enable, equivalent to loss.backward()
            with profiler.profile("model_backward"):
                if backward_begin_callback:
                    backward_begin_callback()
                if self.enable_apex:
                    with apex.amp.scale_loss(loss_scalar, optimizer) as loss_s:
                        loss_s.backward()
                elif self.use_deepspeed:
                    model.backward(loss_scalar)
                else:
                    if (step_id + 1) % self.ga_step != 0:
                        with model.no_sync():
                            self.grad_scaler.scale(loss_scalar).backward()
                    else:
                        self.grad_scaler.scale(loss_scalar).backward()
                if backward_end_callback:
                    backward_end_callback(batch=batch, grad_scaler=self.grad_scaler)

            if (step_id + 1) % self.ga_step == 0:
                # when grad_scaler is not enable, equivalent to optimizer.step() # noqa E501
                with profiler.profile("optimizer_step"):
                    if optimizer_step_begin_callback is not None:
                        optimizer_step_begin_callback(grad_scaler=self.grad_scaler)
                    if self.enable_apex:
                        optimizer.step()
                    elif self.use_deepspeed:
                        model.step()
                    else:
                        self.grad_scaler.step(optimizer)
                        self.grad_scaler.update()

        if batch_end_callback is not None:
            batch_end_callback(
                batch=batch,
                losses=losses,
                model_outs=model_outs,
            )
        if self.enable_amp:
            storage.put("grad_scaler", self.grad_scaler.state_dict(), always_dict=True)


@OBJECT_REGISTRY.register
class MultiBatchProcessor(BatchProcessorMixin):
    """
    Processor can forward backward multiple batches within a training step (before `optimizer.step()`).

    It is useful for:

    (1) Training a multitask model on single task annotation samples, of which
    each task forward backward its batch sequentially within a multitask training step

    (2) Training on a memory shortage GPU and want to increase batch size,
    you are able to forward backward multiple batches within a training step

    .. note::

        Example multitask: vehicle, person and traffic light detection.
        Single task annotation means only annotate vehicle bounding boxes on an image with vehicle,
        person, and traffic light objects.

    .. note::

        Multiple batches should be organized in tuple format, e.g.

        * `batch = (batch1, batch2, ...)`

        If not, it will be treated as a single batch, e.g.

        * `batch = dict(inputs=xx, target=xx)`

        * `batch = [inputs, target]`

        See code below for extra explanation.

    It is much general in usage than `BasicBatchProcessor` , batch and model
    outputs can be in any format, but note that if batch is a tuple means it contains multiple batches.

    It is Hardware independent, run on cpu (device None) or gpu
    (device is gpu id).

    It is suitable for training (need_grad_update) and validation
    (not need_grad_update).

    Args:
        need_grad_update: Whether need gradient update, True for training,
            False for Validation.
        batch_transforms: Config of batch transforms.
        inverse_transforms: Config of transforms,
            used for infer results transform inverse.
        loss_collector: A callable object used to collect loss Tensors in model
            outputs.
        enable_amp: Whether training with `Automatic Mixed Precision`.
        enable_amp_dtype: The dtype of amp, float16 or bfloat16.
        enabel_apex: Whether training with `Apex`.
        enable_channels_last: Whether training with `channels_last`.
        channels_last_keys: Keys in batch need to convert to channels_last.
            if None, all 4d-tensor in batch data will convert to channels_last.
        delay_sync: Whether delay sync grad when train on DDP.
            Refer to: DDP.no_sync() API
        empty_cache: Whether to execute torch.cuda.empty_cache() after each
            forward and backward run
        grad_scaler: An instance ``scaler`` of :class:`GradScaler`
            helps perform the steps of gradient scaling conveniently.
        grad_accumulation_step: The step of grad accumulation.
            Gradient accumulation refers to multiple backwards passes are
            performed before updating the parameters. The goal is to update
            the model's parameters based on different steps,
            instead of performing an update after every single batch.
    """  # noqa

    def __init__(
        self,
        need_grad_update: bool,
        batch_transforms: Optional[List] = None,
        inverse_transforms: Optional[List] = None,
        loss_collector: Callable = None,
        enable_amp: bool = False,
        enable_amp_dtype: torch.dtype = torch.float16,
        enable_apex: bool = False,
        enable_channels_last: bool = False,
        channels_last_keys: Optional[Sequence[str]] = None,
        delay_sync: bool = False,
        empty_cache: bool = False,
        grad_scaler: torch.cuda.amp.GradScaler = None,
        grad_accumulation_step: Union[int, str] = 1,
    ):
        if need_grad_update:
            assert (
                loss_collector is not None
            ), "Provide `loss_collector` when need_grad_update"
            assert callable(loss_collector)
        if enable_amp and enable_apex:
            raise RuntimeError("enable_amp and enable_apex cannot be true together.")
        if enable_apex and apex is None:
            check_packages_available("apex")

        self.ga_step = grad_accumulation_step
        if delay_sync or self.ga_step > 1:
            torch_version = torch.__version__
            assert (enable_apex and apex is not None) or LooseVersion(
                torch_version
            ) >= LooseVersion(
                "1.10.2"
            ), "Delay sync or grad accumulation \
                need apex enabled or higher version of torch."

        if enable_amp_dtype == torch.bfloat16:
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("current gpu devices do not support bfloat16.")

        self.need_grad_update = need_grad_update
        self.loss_collector = loss_collector
        self.enable_amp_dtype = enable_amp_dtype
        self.enable_apex = enable_apex
        self.enable_channels_last = enable_channels_last
        self.channels_last_keys = channels_last_keys
        self.delay_sync = delay_sync
        self.empty_cache = empty_cache
        if grad_scaler is not None:
            self.grad_scaler = grad_scaler
        else:
            self.grad_scaler = GradScaler(enabled=enable_amp)
        self.enable_amp = self.grad_scaler.is_enabled()
        if enable_amp:
            assert self.enable_amp, (
                "When grad_scaler is not None, enable_amp does not work."
                "You set enable_amp is {}, but the enable_amp of "
                "grad_scaler is {}. Please check your config!!"
            ).format(enable_amp, self.enable_amp)
        if batch_transforms:
            if isinstance(batch_transforms, (list, tuple)):
                batch_transforms = Compose(batch_transforms)  # noqa
            self.transforms = batch_transforms
        else:
            self.transforms = None
        self.inverse_transforms = inverse_transforms
        self.use_deepspeed = None

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
        if self.use_deepspeed is None and deepspeed is not None:
            self.use_deepspeed = isinstance(
                model, (deepspeed.DeepSpeedEngine, deepspeed.InferenceEngine)
            )
            if self.use_deepspeed:
                self.deepspeed_config_check()

        assert self.need_grad_update == model.training, (
            "%s vs. %s, set model to training/eval mode by "
            "model.train()/model.eval() when need_grad_update or not"
            % (self.need_grad_update, model.training)
        )
        if profiler is None:
            profiler = PassThroughProfiler()

        # 0. reset grad
        if (
            not self.use_deepspeed
            and self.need_grad_update
            and step_id % self.ga_step == 0
        ):
            with profiler.profile("optimizer_zero_grad"):
                optimizer.zero_grad(set_to_none=True)

        if isinstance(batch, tuple):
            # Means that `batch_data` contains multiple batches, e.g.
            # (1) contains task specific batches of a `multitask model`
            # batch_data = (
            #    [task1_data1, task1_data2, ...],   # task1 batch
            #    [task2_data1, task2_data2, ...],   # task2 batch
            #    [task3_data1, task3_data2, ...],   # can be list/tuple of objs
            #    task4_data                         # or just a single obj
            #    ...
            # )
            #
            # (2) contains multiple batches for a single task model
            # batch_data = (
            #    [batch1_data1, batch1_data2, ...],
            #    [batch2_data1, batch2_data2, ...], # can be list/tuple of objs
            #    data1                              # or just a single obj
            #    ...
            # )
            batches = batch
        else:
            # Means that `data` just contains a single batch, e.g.
            # (1) is a single obj
            # batch_data = task_data  # e.g. a dict(inputs=xx, target=xx)
            #
            # (2) is a list (NOT A TUPLE) of objs
            # batch_data = [task_data1, task_data2, ...]
            #
            # convert to tuple
            batches = (batch,)

        # for each batch in multiple batches
        last_batch_idx = len(batches) - 1
        for idx, batch_i in enumerate(batches):
            if batch_begin_callback is not None:
                batch_begin_callback(batch=batch_i, batch_idx=idx)

            if device is not None:
                batch_i = to_cuda(batch_i, device, non_blocking=True)
            else:
                # run on cpu
                pass

            if isinstance(batch_i, Tuple) and len(batch_i) == 2:
                profile_suffix = batch_i[1]
            else:
                profile_suffix = idx

            if self.transforms is not None:
                with profiler.profile(f"batch_transforms_{profile_suffix}"):
                    batch_i = (self.transforms(batch_i[0]), batch_i[1])

            if self.enable_channels_last:
                batch_i = convert_memory_format(
                    batch_i, self.channels_last_keys, torch.channels_last
                )

            batch_i = maybe_cast_batch_for_deterministic(batch_i)

            # 1. forward
            grad_decorator = (
                torch.enable_grad if self.need_grad_update else torch.no_grad
            )
            if not self.enable_apex:
                auto_cast = autocast(
                    enabled=self.enable_amp, dtype=self.enable_amp_dtype
                )
            else:
                auto_cast = localcontext()

            if forward_begin_callback is not None:
                forward_begin_callback(batch=batch_i, model=model)

            with profiler.profile(f"model_forward_{profile_suffix}"):
                with auto_cast:
                    with grad_decorator():
                        if self.delay_sync and idx != last_batch_idx:
                            # delay sync grad util last batch in mt tuple.
                            if hasattr(model, "disable_allreduce"):
                                model.disable_allreduce()
                        elif (
                            step_id + 1
                        ) % self.ga_step != 0 and idx != last_batch_idx:
                            # delay sync grad by grad accumulation step.
                            if hasattr(model, "disable_allreduce"):
                                model.disable_allreduce()
                        else:
                            # only support enable_allreduce in apex
                            if hasattr(model, "enable_allreduce"):
                                model.enable_allreduce()
                        # model outputs can be in any format
                        model_outs = model(*_as_list(batch_i))

            if self.inverse_transforms is not None:
                model_outs = self.inverse_transforms(model_outs, batch_i)

            if forward_end_callback is not None:
                forward_end_callback(model_outs=model_outs, batch_idx=idx)

            # 2. filter out loss Tensors in model outputs
            if self.loss_collector is not None:
                losses = self.loss_collector(model_outs)
            else:
                losses = None

            if self.empty_cache:
                torch.cuda.empty_cache()

            # 3. backward
            if self.need_grad_update:
                # Not allow to backward each loss independently, so sum them
                loss = sum([loss for loss in _as_list(losses) if loss is not None])
                assert isinstance(loss, torch.Tensor), type(loss)
                # mean of grad accumulation step
                loss_scalar = loss.sum() / self.ga_step
                if backward_begin_callback:
                    backward_begin_callback(batch=batch_i)
                # when grad_scaler is not enable, equivalent to loss.backward()
                with profiler.profile(f"model_backward_{profile_suffix}"):
                    if self.enable_apex:
                        with apex.amp.scale_loss(loss_scalar, optimizer) as loss_s:
                            loss_s.backward()
                    else:
                        if self.delay_sync and idx != last_batch_idx:
                            # delay sync grad util last batch in mt tuple.
                            with model.no_sync():
                                self.grad_scaler.scale(loss_scalar).backward()
                        elif self.use_deepspeed:
                            model.backward(loss_scalar)
                        elif (
                            step_id + 1
                        ) % self.ga_step != 0 and idx != last_batch_idx:
                            # delay sync grad by grad accumulation step.
                            with model.no_sync():
                                self.grad_scaler.scale(loss_scalar).backward()
                        else:
                            self.grad_scaler.scale(loss_scalar).backward()
                if backward_end_callback:
                    backward_end_callback(batch=batch_i, batch_idx=idx)

            if batch_end_callback is not None:
                batch_end_callback(batch=batch_i, losses=losses, model_outs=model_outs)

        # 4. update grad
        if self.need_grad_update and (step_id + 1) % self.ga_step == 0:
            # when grad_scaler is not enable, equivalent to optimizer.step()
            with profiler.profile("optimizer_step"):
                if optimizer_step_begin_callback is not None:
                    optimizer_step_begin_callback(grad_scaler=self.grad_scaler)
                if self.enable_apex:
                    optimizer.step()
                elif self.use_deepspeed:
                    model.step()
                else:
                    self.grad_scaler.step(optimizer)
                    self.grad_scaler.update()

        if self.empty_cache:
            torch.cuda.empty_cache()

        if self.enable_amp:
            storage.put("grad_scaler", self.grad_scaler.state_dict(), always_dict=True)


@OBJECT_REGISTRY.register
class MultiStageBatchProcessor(MultiBatchProcessor):
    """
    Supports multiple stage backward.

    It is a memory saving processor by forward-backward each split task
    individually, if more than one task is trained in a single step.

    Args:
        need_grad_update: Whether need gradient update, True for training,
            False for Validation.
        batch_transforms: Config of batch transforms.
        inverse_transforms: Config of transforms,
            used for infer results transform inverse.
        loss_collector: A callable object used to collect loss Tensors in model
            outputs.
        enable_amp: Whether training with `Automatic Mixed Precision`.
        enabel_apex: Whether training with `Apex`.
        enable_channels_last: Whether training with `channels_last`.
        channels_last_keys: Keys in batch need to convert to channels_last.
            if None, all 4d-tensor in batch data will convert to channels_last.
        delay_sync: Whther delay sync grad when train on DDP.
            Refer to: DDP.no_sync() API
        grad_scaler: An instance ``scaler`` of :class:`GradScaler`
            helps perform the steps of gradient scaling conveniently.
        grad_accumulation_step: The frequence of grad accumulation.
            Gradient accumulation refers to multiple backwards passes are
            performed before updating the parameters. The goal is to update
            the model's parameters based on different steps,
            instead of performing an update after every single batch.
    """

    def __init__(
        self,
        need_grad_update: bool,
        batch_transforms: Optional[List] = None,
        inverse_transforms: Optional[List] = None,
        loss_collector: Callable = None,
        enable_amp: bool = False,
        enable_amp_dtype: torch.dtype = torch.float16,
        enable_apex: bool = False,
        enable_channels_last: bool = False,
        channels_last_keys: Optional[Sequence[str]] = None,
        delay_sync: bool = False,
        empty_cache: bool = False,
        grad_scaler: torch.cuda.amp.GradScaler = None,
        split_node_name: Optional[str] = None,
        grad_accumulation_step: Union[int, str] = 1,
    ):
        super().__init__(
            need_grad_update,
            batch_transforms,
            inverse_transforms,
            loss_collector,
            enable_amp,
            enable_amp_dtype,
            enable_apex,
            enable_channels_last,
            channels_last_keys,
            delay_sync,
            empty_cache,
            grad_scaler,
            grad_accumulation_step,
        )
        self.split_node_name = split_node_name

    @property
    def _model_cache(self):
        if not hasattr(self, "_model_cache_"):
            self._model_cache_ = {}
        return self._model_cache_

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
        if self.use_deepspeed is None and deepspeed is not None:
            self.use_deepspeed = isinstance(
                model, (deepspeed.DeepSpeedEngine, deepspeed.InferenceEngine)
            )
            if self.use_deepspeed:
                self.deepspeed_config_check()

        assert self.need_grad_update == model.training, (
            "%s vs. %s, set model to training/eval mode by "
            "model.train()/model.eval() when need_grad_update or not"
            % (self.need_grad_update, model.training)
        )

        if profiler is None:
            profiler = PassThroughProfiler()

        # 0. reset grad
        if self.need_grad_update and step_id % self.ga_step == 0:
            with profiler.profile("optimizer_zero_grad"):
                optimizer.zero_grad(set_to_none=True)

        if isinstance(batch, tuple):
            # Means that `batch_data` contains multiple batches, e.g.
            # (1) contains task specific batches of a `multitask model`
            # batch_data = (
            #    [task1_data1, task1_data2, ...],   # task1 batch
            #    [task2_data1, task2_data2, ...],   # task2 batch
            #    [task3_data1, task3_data2, ...],   # can be list/tuple of objs
            #    task4_data                         # or just a single obj
            #    ...
            # )
            #
            # (2) contains multiple batches for a single task model
            # batch_data = (
            #    [batch1_data1, batch1_data2, ...],
            #    [batch2_data1, batch2_data2, ...], # can be list/tuple of objs
            #    data1                              # or just a single obj
            #    ...
            # )
            batches = batch
        else:
            # Means that `data` just contains a single batch, e.g.
            # (1) is a single obj
            # batch_data = task_data  # e.g. a dict(inputs=xx, target=xx)
            #
            # (2) is a list (NOT A TUPLE) of objs
            # batch_data = [task_data1, task_data2, ...]
            #
            # convert to tuple
            batches = (batch,)

        last_batch_idx = len(batches) - 1

        # for each batch in multiple batches
        for idx, batch_i in enumerate(batches):
            if batch_begin_callback is not None:
                batch_begin_callback(batch=batch_i, batch_idx=idx)

            if device is not None:
                batch_i = to_cuda(batch_i, device, non_blocking=True)
            else:
                # run on cpu
                pass

            if isinstance(batch_i, Tuple) and len(batch_i) == 2:
                profile_suffix = batch_i[1]
            else:
                profile_suffix = idx

            if self.transforms is not None:
                with profiler.profile(f"batch_transforms_{profile_suffix}"):
                    batch_i = (self.transforms(batch_i[0]), batch_i[1])

            batch_i = maybe_cast_batch_for_deterministic(batch_i)

            if self.enable_channels_last:
                torch_version = torch.__version__
                if LooseVersion(torch_version) < LooseVersion("1.13.0"):
                    msg = (
                        "Channels last is not supported in "
                        + "MultiStageBatchProcessor"
                        + "while the version of torch <= 1.10.2."
                        + "tensor.detach() and channels_last"
                        + "have some conflicts. Please refer to"
                        + "https://github.com/pytorch/pytorch/pull/65594."
                    )
                    raise RuntimeError(msg)
                else:
                    batch_i[0] = convert_memory_format(
                        batch_i[0],
                        self.channels_last_keys,
                        torch.channels_last,
                    )

            # check splitable
            sort_names = sorted(_as_list(batch_i[1]))
            tag = (
                ">".join(sort_names) if isinstance(sort_names, Sequence) else sort_names
            )
            if tag in self._model_cache:
                # use cache
                (
                    common_model,
                    split_model,
                    split_model_input_names,
                    split_model_opt_input_names,
                ) = self._model_cache[tag]
            else:
                dist_type = torch.nn.parallel.DistributedDataParallel
                if self.use_deepspeed:
                    dist_type = (
                        deepspeed.DeepSpeedEngine,
                        deepspeed.InferenceEngine,
                    )
                if isinstance(model, dist_type):
                    (
                        common_model,
                        split_model,
                        split_model_input_names,
                        split_model_opt_input_names,
                    ) = model.module.split_module(
                        batch_i[1], split_node_name=self.split_node_name
                    )
                    if common_model and split_model:
                        group_common = torch.distributed.new_group()
                        group_split = torch.distributed.new_group()
                        common_kwargs = {
                            "find_unused_parameters": model.find_unused_parameters,  # noqa
                            "broadcast_buffers": model.broadcast_buffers,
                            "device_ids": model.device_ids,
                        }
                        if hasattr(model, "assign_module_buffers"):
                            common_kwargs["assign_module_buffers"] = (
                                model.assign_module_buffers
                            )
                        common_model = type(model)(
                            module=common_model,
                            process_group=group_common,
                            **common_kwargs,
                        )
                        split_model = type(model)(
                            module=split_model,
                            process_group=group_split,
                            **common_kwargs,
                        )
                else:
                    (
                        common_model,
                        split_model,
                        split_model_input_names,
                        split_model_opt_input_names,
                    ) = model.split_module(
                        batch_i[1], split_node_name=self.split_node_name
                    )
                self._model_cache[tag] = (
                    common_model,
                    split_model,
                    split_model_input_names,
                    split_model_opt_input_names,
                )

            grad_decorator = (
                torch.enable_grad if self.need_grad_update else torch.no_grad
            )
            if not self.enable_apex:
                auto_cast = autocast(enabled=self.enable_amp)
            else:
                auto_cast = localcontext()

            if forward_begin_callback is not None:
                forward_begin_callback(batch=batch_i, model=model)

            if common_model and split_model:
                # common forward
                with auto_cast:
                    with grad_decorator():
                        if self.delay_sync and idx != last_batch_idx:
                            if hasattr(common_model, "disable_allreduce"):
                                # delay sync grad until last batch
                                common_model.disable_allreduce()
                        elif (
                            step_id + 1
                        ) % self.ga_step != 0 and idx != last_batch_idx:
                            # delay sync grad by grad accumulation freq.
                            if hasattr(common_model, "disable_allreduce"):
                                common_model.disable_allreduce()
                        else:
                            # only support enable_allreduce in apex
                            if hasattr(common_model, "enable_allreduce"):
                                common_model.enable_allreduce()
                        # model outputs can be in any format
                        common_model_outs = common_model(*_as_list(batch_i[0]))
                flatten_common_outputs, outputs_layout = flatten(common_model_outs)
                detached_common_outputs = [
                    tensor.detach() for tensor in flatten_common_outputs
                ]
                detached_common_outputs = [
                    (
                        _call_as_tensor(torch.Tensor.requires_grad_, tensor)
                        if isinstance(tensor, QTensor)
                        else tensor.requires_grad_()
                    )
                    for tensor in detached_common_outputs
                ]

                split_inputs = regroup(detached_common_outputs, outputs_layout)[0]

                model_outs = None
                total_loss = []
                # split forward and backward
                common_model_output_keys = list(split_inputs.keys())
                for k, v in batch_i[0].items():
                    if (
                        k in (split_model_input_names + split_model_opt_input_names)
                        and k not in batch_i[1]
                    ):
                        split_inputs[k] = v

                for output_name in batch_i[1]:
                    split_inputs[output_name] = batch_i[0][output_name]
                    with auto_cast:
                        with grad_decorator():
                            if self.delay_sync and idx != last_batch_idx:
                                if hasattr(split_model, "disable_allreduce"):
                                    # delay sync grad util last batch
                                    split_model.disable_allreduce()
                            elif (
                                step_id + 1
                            ) % self.ga_step != 0 and idx != last_batch_idx:
                                if hasattr(split_model, "disable_allreduce"):
                                    # delay sync grad by grad accumulation freq
                                    split_model.disable_allreduce()
                            else:
                                # only support enable_allreduce in apex
                                if hasattr(split_model, "enable_allreduce"):
                                    split_model.enable_allreduce()
                            partial = split_model(split_inputs, output_name)
                            if model_outs:
                                model_outs = model_outs + partial
                            else:
                                model_outs = partial

                    split_inputs.pop(output_name)
                    if self.empty_cache:
                        torch.cuda.empty_cache()

                    if self.loss_collector is not None:
                        losses = self.loss_collector(partial)
                    else:
                        losses = None

                    loss = sum([loss for loss in _as_list(losses) if loss is not None])

                    total_loss.extend(losses)
                    # mean of grad accumulation step
                    loss_scalar = loss.sum() / self.ga_step
                    if self.enable_apex:
                        with apex.amp.scale_loss(loss_scalar, optimizer) as loss_s:
                            loss_s.backward()
                    else:
                        if self.delay_sync and idx != last_batch_idx:
                            with split_model.no_sync():
                                self.grad_scaler.scale(loss_scalar).backward()
                        elif self.use_deepspeed:
                            split_model.backward(loss_scalar)
                        elif (
                            step_id + 1
                        ) % self.ga_step != 0 and idx != last_batch_idx:
                            with split_model.no_sync():
                                self.grad_scaler.scale(loss_scalar).backward()
                        else:
                            self.grad_scaler.scale(loss_scalar).backward()
                for k in list(split_inputs.keys()):
                    if k not in common_model_output_keys:
                        split_inputs.pop(k)
                if forward_end_callback is not None:
                    forward_end_callback(model_outs=model_outs, batch_idx=idx)
                # common backward
                flatten_split_inputs, _ = flatten(split_inputs)
                valid_split_inputs = [
                    tensor
                    for tensor in flatten_split_inputs
                    if _call_as_tensor(torch.Tensor.grad.__get__, tensor) is not None
                ]
                last_valid_idx = len(valid_split_inputs) - 1
                for i, tensor in enumerate(valid_split_inputs):
                    retain_graph = i != last_valid_idx
                    if self.delay_sync and idx != last_batch_idx:
                        with common_model.no_sync():
                            flatten_common_outputs[i].backward(
                                _call_as_tensor(torch.Tensor.grad.__get__, tensor),
                                retain_graph=retain_graph,
                            )
                    else:
                        flatten_common_outputs[i].backward(
                            _call_as_tensor(torch.Tensor.grad.__get__, tensor),
                            retain_graph=retain_graph,
                        )
                if backward_end_callback:
                    backward_end_callback(batch=batch_i, batch_idx=idx)
                if batch_end_callback is not None:
                    batch_end_callback(
                        batch=batch_i, losses=total_loss, model_outs=model_outs
                    )
                if self.empty_cache:
                    torch.cuda.empty_cache()
            else:
                # 1. forward
                with profiler.profile(f"model_forward_{profile_suffix}"):
                    with auto_cast:
                        with grad_decorator():
                            if self.delay_sync and idx != last_batch_idx:
                                if self.enable_apex and isinstance(
                                    model,
                                    apex.parallel.DistributedDataParallel,
                                ):
                                    # delay sync grad until last batch
                                    model.disable_allreduce()
                                    model_outs = model(*_as_list(batch_i))
                                else:
                                    model_outs = model(*_as_list(batch_i))
                            else:
                                # only support enable_allreduce in apex
                                if hasattr(model, "enable_allreduce"):
                                    model.enable_allreduce()
                                # model outputs can be in any format
                                model_outs = model(*_as_list(batch_i))

                if self.inverse_transforms is not None:
                    model_outs = self.inverse_transforms(model_outs, batch_i)

                if forward_end_callback is not None:
                    forward_end_callback(model_outs=model_outs, batch_idx=idx)

                # 2. filter out loss Tensors in model outputs
                if self.loss_collector is not None:
                    losses = self.loss_collector(model_outs)
                else:
                    losses = None

                if self.empty_cache:
                    torch.cuda.empty_cache()

                # 3. backward
                if self.need_grad_update:
                    # Not allow to backward each loss independently,
                    # so sum them
                    loss = sum([loss for loss in _as_list(losses) if loss is not None])
                    assert isinstance(loss, torch.Tensor), type(loss)
                    # mean of grad accumulation step
                    loss_scalar = loss.sum() / self.ga_step
                    if backward_begin_callback:
                        backward_begin_callback(batch=batch_i)
                    # when grad_scaler is not enable,
                    # equivalent to loss.backward()
                    with profiler.profile(f"model_backward_{profile_suffix}"):
                        if self.enable_apex:
                            with apex.amp.scale_loss(loss_scalar, optimizer) as loss_s:
                                loss_s.backward()
                        else:
                            if self.delay_sync and idx != last_batch_idx:
                                with model.no_sync():
                                    self.grad_scaler.scale(loss_scalar).backward()

                            elif self.use_deepspeed:
                                model.backward(loss_scalar)
                            else:
                                self.grad_scaler.scale(loss_scalar).backward()
                    if backward_end_callback:
                        backward_end_callback(batch=batch_i, batch_idx=idx)

                if batch_end_callback is not None:
                    batch_end_callback(
                        batch=batch_i, losses=losses, model_outs=model_outs
                    )

        # 4. update grad
        if self.need_grad_update and (step_id + 1) % self.ga_step == 0:
            # when grad_scaler is not enable, equivalent to optimizer.step()
            with profiler.profile("optimizer_step"):
                if optimizer_step_begin_callback is not None:
                    optimizer_step_begin_callback(grad_scaler=self.grad_scaler)
                if self.enable_apex:
                    optimizer.step()
                elif self.use_deepspeed:
                    model.step()
                else:
                    self.grad_scaler.step(optimizer)
                    self.grad_scaler.update()

        if self.empty_cache:
            torch.cuda.empty_cache()

        if self.enable_amp:
            storage.put("grad_scaler", self.grad_scaler.state_dict(), always_dict=True)
