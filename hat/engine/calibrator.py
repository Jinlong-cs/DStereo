# Copyright (c) Horizon Robotics. All rights reserved.
# type: ignore

import logging
import os
import time
from typing import Dict, Iterable, List, Optional, Sequence, Union

import horizon_plugin_pytorch as horizon
import torch
import torch.nn as nn

from hat.callbacks import CallbackMixin
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list
from hat.utils.logger import MSGColor, format_msg
from hat.utils.package_helper import check_packages_available
from .launcher import register_launcher
from .loop_base import LoopBase
from .processors import BatchProcessorMixin

__all__ = ["Calibrator"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register_module
class Calibrator(LoopBase):
    """Calibrator is a tool for calibration.

    The abundant callbacks in trainer is also supported.

    Args:
        model: `nn.Module` instance.
        data_loader: Validation data loader.
        batch_processor: Batch processor config.
        device: Int gpu id or None.
        model_convert_pipeline: Define the process of model convert.
            e.g. convert float model to qat model, convert qat model
            to quantize model.
        num_steps: Num of calibration steps, should be non-negative integer.
        callbacks: Callbacks.
        val_metrics: Metrics on validation data.
        profiler: To profile individual steps during training and
            assist in identifying bottlenecks.
        log_interval: Logging output frequency.
        auto_calibration: Whether to enable auto calibration to search optimal
            observer automatically.
        auto_calibration_config: Custom config for auto calibration.
            Default keys are:

            .. code-block:: python

                auto_calibration_config = {
                    # candidate observers to search
                    "observer_list": ["percentile", "mse", "kl", "min_max"],

                    # candidate parameters for percentile observer
                    percentile_list: [99.995],

                    # whether to load data to device in advance.
                    # greatly boost performance at cost of
                    # more memory occupation
                    "preload_data": False,

                }

        weight_reconstruction: Whether to enable weight reconstruction after
            calibration.
        weight_reconstruction_config: Custom config for weight reconstruction.
            Default keys are:

            .. code-block:: python

                weight_reconstruction_config = {
                    # whether to load data to device in advance.
                    # greatly boost performance at cost of
                    # more memory occupation
                    "preload_data": False,

                }
    """

    def __init__(
        self,
        model: nn.Module,
        data_loader: Iterable,
        batch_processor: BatchProcessorMixin,
        device: Optional[int] = None,
        model_convert_pipeline: Optional[Union[Dict, List]] = None,
        num_steps: Optional[int] = None,
        callbacks: Optional[Sequence[Union[dict, CallbackMixin]]] = None,
        val_metrics: Optional[dict] = None,
        profiler: Optional[dict] = None,
        log_interval: int = 0,
        auto_calibration: bool = False,
        auto_calibration_config: Optional[dict] = None,
        weight_reconstruction: bool = False,
        weight_reconstruction_config: Optional[dict] = None,
        compiler: Optional[Dict] = None,
    ):
        # set `update_interval` for KL observer according to `num_steps`
        # for best performance
        if (
            not auto_calibration
            and model_convert_pipeline.qconfig_params.get(
                "activation_calibration_observer"
            )
            == "kl"
        ):
            qkwargs = model_convert_pipeline.qconfig_params.get(
                "activation_calibration_qkwargs", {}
            )
            qkwargs.update(update_interval=num_steps)
            model_convert_pipeline.qconfig_params[
                "activation_calibration_qkwargs"
            ] = qkwargs

            model_convert_pipeline.refresh_global_qconfig()

        super(Calibrator, self).__init__(
            model=model,
            data_loader=data_loader,
            optimizer=None,
            batch_processor=batch_processor,
            device=device,
            model_convert_pipeline=model_convert_pipeline,
            num_steps=num_steps,
            stop_by="step",
            callbacks=callbacks,
            val_metrics=val_metrics,
            profiler=profiler,
            log_interval=log_interval,
            compiler=compiler,
        )

        if len(self.callbacks) == 0:
            logger.info(
                format_msg(
                    "`callbacks` is empty, make sure you want this",
                    MSGColor.RED,
                )
            )

        if batch_processor.need_grad_update:
            batch_processor.need_grad_update = False
        assert batch_processor.need_grad_update is False

        self.auto_calibration = auto_calibration
        self.weight_reconstruction = weight_reconstruction
        self.batch_transform = batch_processor.transforms

        if self.auto_calibration:
            check_packages_available("horizon_plugin_pytorch>=2.0.1")

            logger.info(
                format_msg(
                    "Auto calibration enabled. "
                    "Skip standard calibration steps."
                    "`num_steps` will be used by auto calibration instead.",
                    MSGColor.GREEN,
                )
            )

            self.num_auto_batches = self.num_steps
            self.num_steps = 0

            self.auto_calibration_config = (
                auto_calibration_config
                if auto_calibration_config is not None
                else {}
            )
            self.auto_preload_data = self.auto_calibration_config.pop(
                "preload_data", False
            )

        if self.weight_reconstruction:
            check_packages_available("horizon_plugin_pytorch>=2.0.1")

            self.weight_reconstruction_config = (
                weight_reconstruction_config
                if weight_reconstruction_config is not None
                else {}
            )
            self.wr_preload_data = self.weight_reconstruction_config.pop(
                "preload_data", False
            )

            if "num_steps" not in self.weight_reconstruction_config:
                self.weight_reconstruction_config["num_steps"] = 1000

    def _get_preloaded_data(self, num_batches):
        logger.info(format_msg("Preloading data...", MSGColor.GREEN))
        calib_batches = []
        count = 0
        for batch in self.data_loader:
            if self.batch_transform is not None:
                batch = self.batch_transform(batch)
            calib_batches.append(batch)
            count += 1
            if count == num_batches:
                break

        return calib_batches, count * self.data_loader.batch_size

    def on_epoch_begin(self, **kwargs):
        self.model.eval()
        horizon.quantization.set_fake_quantize(
            self.model, horizon.quantization.FakeQuantState.CALIBRATION
        )
        return super().on_epoch_begin(**kwargs)

    def on_epoch_end(self, **kwargs):
        if self.auto_calibration:
            if self.auto_preload_data:
                calib_batches, num_samples = self._get_preloaded_data(
                    self.num_auto_batches
                )
                logger.info(
                    format_msg(
                        "Preload data for auto calibration, total numbers "
                        f"of samples: {num_samples}.",
                        MSGColor.GREEN,
                    )
                )
            else:
                logger.warning(
                    format_msg(
                        "`preload_data` is `False` for auto calibration. "
                        "This may result in significant I/O bottlenecks, "
                        "as auto calibration requires frequent data access. "
                        "Setting `preload_data = True` "
                        "in `auto_calibration_config` "
                        "can greatly boost performance at cost of "
                        "more memory occupation.",
                        MSGColor.RED,
                    )
                )

            horizon.quantization.auto_calibrate(
                self.model,
                calib_batches if self.auto_preload_data else self.data_loader,
                self.num_auto_batches,
                None if self.auto_preload_data else self.batch_transform,
                **self.auto_calibration_config,
            )

        self.model.eval()
        horizon.quantization.set_fake_quantize(
            self.model, horizon.quantization.FakeQuantState.VALIDATION
        )
        return super().on_epoch_end(**kwargs)

    def on_loop_end(self, model, **kwargs):
        if self.weight_reconstruction:
            if self.wr_preload_data:
                calib_batches, num_samples = self._get_preloaded_data(10)
                logger.info(
                    format_msg(
                        "Preload data for weight reconstruction, total numbers"
                        f" of samples: {num_samples}.",
                        MSGColor.GREEN,
                    )
                )
            else:
                logger.warning(
                    format_msg(
                        "`preload_data` is `False` for weight reconstruction. "
                        "This may result in significant I/O bottlenecks, "
                        "as weight reconstruction requires "
                        "frequent data access. "
                        "Setting `preload_data = True` "
                        "in `auto_calibration_config` "
                        "can greatly boost performance at cost of "
                        "more memory occupation.",
                        MSGColor.RED,
                    )
                )

            model.eval()

            reconstruction_time_begin = time.time()

            horizon.quantization.weight_reconstruction(
                model,
                calib_batches if self.wr_preload_data else self.data_loader,
                None if self.wr_preload_data else self.batch_transform,
                self.weight_reconstruction_config,
            )

            reconstruction_time = time.time() - reconstruction_time_begin
            logger.info(
                "Weight Reconstruction Cost Time: %.3fs" % reconstruction_time
            )

            model.eval()
            horizon.quantization.set_fake_quantize(
                model, horizon.quantization.FakeQuantState.VALIDATION
            )

        return super().on_loop_end(model=model, **kwargs)


def launch(
    main_func, device_ids=None, dist_url=None, dist_launcher=None, args=()
):
    if device_ids is None:
        current_device = None
    else:
        device_ids = _as_list(device_ids)
        if len(device_ids) > 1:
            msg = (
                "`Calibrator` only support one device, but get %s, now "
                "only use device %d." % (device_ids, device_ids[0])
            )
            logger.info(format_msg(msg, MSGColor.GREEN))
            device_ids = device_ids[:1]

        # Note: if device_ids=[3], then after setting
        # `CUDA_VISIBLE_DEVICES`, index of gpu 3 becomes 0.
        current_device = int(device_ids[0])
        torch.cuda.set_device(current_device)

    # only support run on cpu0-gpu0 while using mpi
    if dist_launcher == "mpi":
        import mpi4py.MPI as MPI

        comm = MPI.COMM_WORLD
        local_rank = comm.Get_rank()
        if local_rank > 0:
            return
    elif dist_launcher == "torch":
        local_rank = int(os.environ["RANK"])
        if local_rank > 0:
            return

    main_func(current_device, *args)


register_launcher("Calibrator", launch)
