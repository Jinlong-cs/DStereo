# Copyright (c) Horizon Robotics. All rights reserved.
from functools import partial

import numpy as np
import torch

from hat.core.event import EventStorage
from hat.engine.loop_base import PipeBase
from hat.engine.processors import BatchProcessorMixin
from hat.metrics.metric import EvalMetric
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "FakeAuto2dDataset",
    "BoringModel",
    "BoringTraceModel",
    "BoringMetric",
    "BoringTrainer",
    "ToyIterableDataset",
]


@OBJECT_REGISTRY.register
class FakeAuto2dDataset(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __getitem__(self, i):
        data = dict(
            image_name="test.jpg",
        )
        return data

    def __len__(self):
        return 2


class BoringModel(torch.nn.Module):
    def __init__(self):
        super(BoringModel, self).__init__()
        self.conv = torch.nn.Conv2d(3, 128, kernel_size=1)

    def forward(self, data):
        output = dict(
            pred_bboxes=np.asarray(
                [
                    [
                        [0, 0, 100, 100, 0.95],
                    ]
                ]
            ),
        )
        return output


class BoringTraceModel(torch.nn.Module):
    def __init__(self):
        super(BoringTraceModel, self).__init__()
        self.conv = torch.nn.Conv2d(3, 128, kernel_size=1)

    def forward(self, data):
        return data


class BoringBatchProcessor(BatchProcessorMixin):
    def __init__(self):
        super(BoringBatchProcessor, self).__init__()

    def __call__(
        self,
        step_id,
        batch,
        model,
        device,
        optimizer=None,
        storage=None,
        profiler=None,
        batch_begin_callback=None,
        batch_end_callback=None,
        backward_begin_callback=None,
        backward_end_callback=None,
        optimizer_step_begin_callback=None,
        forward_begin_callback=None,
        forward_end_callback=None,
    ):
        batch = dict(img=torch.Tensor([1.0]), label=torch.Tensor([2.0]))
        if batch_begin_callback is not None:
            batch_begin_callback(batch=batch)

        model(batch["img"])

        model_outs = dict(
            predict=torch.Tensor([3.0]), loss=torch.Tensor([4.0])
        )
        if batch_end_callback is not None:
            batch_end_callback(
                batch=batch, losses=model_outs["loss"], model_outs=model_outs
            )


class BoringMetric(EvalMetric):
    def __init__(self, name="accuracy"):
        super(BoringMetric, self).__init__(name)

    def update(self, output):
        pass

    def get(self):
        return [self.name], [0.5]


class BoringTrainer(PipeBase):
    def __init__(
        self,
        model=None,
        data_loader=None,
        num_epochs=3,
        num_steps=None,
        start_epoch=0,
        start_step=0,
        optimizer=None,
        device=None,
        train_metrics=None,
        val_metrics=None,
        callbacks=None,
    ):
        super(BoringTrainer, self).__init__(callbacks)
        self.model = model if model is not None else BoringModel()
        self.ema_model = None
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.start_epoch = start_epoch
        self.start_step = start_step
        self.device = device
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        if data_loader is not None:
            self.data_loader = data_loader
        else:
            dataset = FakeAuto2dDataset()
            self.data_loader = torch.utils.data.DataLoader(
                dataset=dataset, batch_size=1
            )

        self.optimizer = (
            optimizer
            if optimizer
            else torch.optim.Adam(self.model.parameters(), 0.01)
        )
        self.batch_processor = BoringBatchProcessor()
        self.storage = EventStorage()

    def fit(self):
        global_step_id = 0
        epoch_id = self.start_epoch
        self.model.train()
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

        while epoch_id < self.num_epochs:
            self.model.train()
            self.on_epoch_begin(
                model=self.model,
                epoch_id=epoch_id,
                optimizer=self.optimizer,
                global_step_id=global_step_id,
                train_metrics=self.train_metrics,
                val_metrics=self.val_metrics,
                storage=self.storage,
            )

            for step_id, data in enumerate(self.data_loader):
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
                    data,
                    self.model,
                    self.device,
                    optimizer=self.optimizer,
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
                )
                self.on_step_end(
                    epoch_id=epoch_id,
                    step_id=step_id,
                    global_step_id=global_step_id,
                    data_loader=self.data_loader,
                    model=self.model,
                    ema_model=self.ema_model,
                    optimizer=self.optimizer,
                    num_steps=self.num_steps,
                    callbacks=self.callbacks,
                    device=self.device,
                    train_metrics=self.train_metrics,
                    val_metrics=self.val_metrics,
                    storage=self.storage,
                )
                global_step_id += 1

            self.on_epoch_end(
                epoch_id=epoch_id,
                global_step_id=global_step_id,
                model=self.model,
                ema_model=self.ema_model,
                optimizer=self.optimizer,
                num_epochs=self.num_epochs,
                callbacks=self.callbacks,
                device=self.device,
                train_metrics=self.train_metrics,
                val_metrics=self.val_metrics,
                storage=self.storage,
            )
            epoch_id += 1

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


class ToyIterableDataset(torch.utils.data.IterableDataset):
    """
    Toy IterableDataset for test only. For IterableDataset working with
    multiple workers, please refer to IterableDataset docstring.
    """

    def __init__(self, start=0, end=32):
        super(ToyIterableDataset, self).__init__()
        assert start < end
        self.start, self.end = start, end

    def __iter__(self):
        return iter(range(self.start, self.end))
