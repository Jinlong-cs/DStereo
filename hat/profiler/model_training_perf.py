# Copyright (c) Horizon Robotics. All rights reserved.

import logging
import time
from functools import partial
from typing import Optional

import numpy as np

from hat.callbacks.metric_updater import MetricUpdater
from hat.callbacks.monitor import StatsMonitor
from hat.data.dataloaders.multitask_loader import MultitaskLoader
from hat.registry import OBJECT_REGISTRY
from .profilers import BaseProfiler, PassThroughProfiler

__all__ = ["ModelTrainingPerf"]

logger = logging.getLogger(__name__)


def fit(trainer, iter_nums, frequent, batch_size, profiler):
    """Perf model training, similar to Trainer.fit."""
    self = trainer
    btic = time.time()
    global_step_id = 0
    batch = None
    speed_list, cost_list = [], []

    self.model.train()
    with profiler.profile("on_loop_begin"):
        self.on_loop_begin(
            model=self.model,
            optimizer=self.optimizer,
            data_loader=self.data_loader,
            num_epochs=self.num_epochs,
            num_steps=self.num_steps,
            loop=self,
            train_metrics=self.train_metrics,
            val_metrics=self.val_metrics,
            storage=self.storage,
        )

    for epoch_id in range(self.start_epoch, self.num_epochs):
        self.model.train()
        with profiler.profile("on_epoch_begin"):
            self.on_epoch_begin(
                model=self.model,
                epoch_id=epoch_id,
                optimizer=self.optimizer,
                global_step_id=global_step_id,
                train_metrics=self.train_metrics,
                val_metrics=self.val_metrics,
                storage=self.storage,
                data_loader=self.data_loader,
            )

        while True:
            step_id = global_step_id

            if batch is None:
                batch = next(iter(self.data_loader))

            with profiler.profile("on_step_begin"):
                self.on_step_begin(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch_id=epoch_id,
                    step_id=step_id,
                    data_loader=self.data_loader,
                    start_epoch=self.start_epoch,
                    start_step=self.start_step,
                    global_step_id=global_step_id,
                    train_metrics=self.train_metrics,
                    val_metrics=self.val_metrics,
                    storage=self.storage,
                )

            self.batch_processor(
                step_id,
                batch,
                self.model,
                self.device,
                optimizer=self.optimizer,
                storage=self.storage,
                batch_begin_callback=partial(
                    self.on_batch_begin,
                    global_step_id=global_step_id,
                    step_id=step_id,
                    epoch_id=epoch_id,
                    train_metrics=self.train_metrics,
                    val_metrics=self.val_metrics,
                    storage=self.storage,
                ),
                batch_end_callback=partial(
                    self.on_batch_end,
                    global_step_id=global_step_id,
                    step_id=step_id,
                    epoch_id=epoch_id,
                    train_metrics=self.train_metrics,
                    val_metrics=self.val_metrics,
                    storage=self.storage,
                ),
                backward_begin_callback=partial(
                    self.on_backward_begin,
                    model=self.model,
                    optimizer=self.optimizer,
                    global_step_id=global_step_id,
                    step_id=step_id,
                    epoch_id=epoch_id,
                ),
                backward_end_callback=partial(
                    self.on_backward_end,
                    model=self.model,
                    optimizer=self.optimizer,
                    global_step_id=global_step_id,
                    step_id=step_id,
                    epoch_id=epoch_id,
                ),
                optimizer_step_begin_callback=partial(
                    self.on_optimizer_step_begin,
                    model=self.model,
                    optimizer=self.optimizer,
                ),
                profiler=profiler,
                forward_begin_callback=partial(
                    self.on_forward_begin,
                    global_step_id=global_step_id,
                    step_id=step_id,
                    epoch_id=epoch_id,
                ),
                forward_end_callback=partial(
                    self.on_forward_end,
                    global_step_id=global_step_id,
                    step_id=step_id,
                    epoch_id=epoch_id,
                ),
            )

            with profiler.profile("on_step_end"):
                self.on_step_end(
                    epoch_id=epoch_id,
                    step_id=step_id,
                    global_step_id=global_step_id,
                    data_loader=self.data_loader,
                    model=self.model,
                    ema_model=self.ema_model,
                    optimizer=self.optimizer,
                    num_steps=self.num_steps,
                    device=self.device,
                    train_metrics=self.train_metrics,
                    val_metrics=self.val_metrics,
                    storage=self.storage,
                    callbacks=self.callbacks,
                )

            if (global_step_id + 1) % frequent == 0:
                cost = time.time() - btic
                speed = frequent * batch_size / cost
                s = f"Batch[{global_step_id}] Speed: {speed:.2f} samples/sec | Cost: {cost:.3f} sec"  # noqa
                logger.info(s)
                speed_list.append(speed)
                cost_list.append(cost)
                btic = time.time()

            global_step_id += 1
            if global_step_id >= iter_nums:
                s = f"Average Batch Speed: {np.mean(speed_list):.2f} samples/sec | Cost: {np.mean(cost_list):.2f} sec"  # noqa
                logger.info(s)
                return

        with profiler.profile("on_epoch_end"):
            self.on_epoch_end(
                epoch_id=epoch_id,
                global_step_id=global_step_id,
                model=self.model,
                ema_model=self.ema_model,
                optimizer=self.optimizer,
                num_epochs=self.num_epochs,
                device=self.device,
                train_metrics=self.train_metrics,
                val_metrics=self.val_metrics,
                storage=self.storage,
            )

        if global_step_id >= iter_nums:
            break
    with profiler.profile("on_loop_end"):
        self.on_loop_end(
            model=self.model,
            ema_model=self.ema_model,
            optimizer=self.optimizer,
            epoch_id=epoch_id,
            global_step_id=global_step_id,
            device=self.device,
            train_metrics=self.train_metrics,
            val_metrics=self.val_metrics,
            callbacks=self.callbacks,
            storage=self.storage,
        )


@OBJECT_REGISTRY.register
class ModelTrainingPerf:
    """
    Model training speed perf plugin, used to test model training speed.

    Including forward/backward/step, without loading data.

    Args:
        trainer: trainer instance.
        iter_nums: perf iteration nums.
        frequent: log frequent.
        show_loss: show loss metric.
        profiler: checking if there are any bottlenecks during model training.
    """

    def __init__(
        self,
        trainer,
        iter_nums: Optional[int] = 1000,
        frequent: Optional[int] = 1,
        show_loss: Optional[bool] = False,
        profiler: Optional[BaseProfiler] = None,
    ):
        self.trainer = trainer
        self.iter_nums = iter_nums
        self.frequent = frequent
        if profiler is None:
            profiler = PassThroughProfiler()
        self.profiler = profiler

        # TODO(min.du, 0.1): put batch_size to Trainer #
        dataloader = trainer.data_loader
        if hasattr(dataloader, "batch_size"):
            self.batch_size = dataloader.batch_size
        elif isinstance(dataloader, MultitaskLoader):
            loaders = dataloader.loaders
            if isinstance(loaders, dict):
                self.batch_size = sum([loaders[k].batch_size for k in loaders])
            elif isinstance(loaders, list):
                self.batch_size = sum([item.batch_size for item in loaders])
            else:
                raise ValueError
        else:
            raise NotImplementedError(f"{type(loaders)} not supported")

        # don't show loss
        if show_loss:
            # change log_interval to frequent
            pass
        else:
            new_callbacks = []
            for cb in self.trainer.callbacks:
                if not isinstance(cb, (StatsMonitor, MetricUpdater)):
                    new_callbacks.append(cb)
            self.trainer.callbacks = new_callbacks

    def run(self) -> None:
        fit(
            self.trainer,
            self.iter_nums,
            self.frequent,
            self.batch_size,
            self.profiler,
        )
        self.profiler.describe()
