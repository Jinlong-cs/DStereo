import copy
from typing import Optional

from hat.core.task_sampler import TaskSampler


def build_validation(
    val_model,
    val_batch_processor,
    task_configs,
    vis_callback=None,
    interval_by="step",
    val_interval=5000,
    custom_length: Optional[int] = None,
    inf_loader=False,
):
    val_callbacks = []
    for t in task_configs:
        task_name = t.task_name
        loaders = t.val_dataloader
        val_metric_updater = t.val_metric_updater

        if len(val_metric_updater) == 1:
            val_metric_updaters = [
                copy.deepcopy(val_metric_updater) for _ in loaders
            ]
        else:
            assert len(val_metric_updater) == len(loaders)
            val_metric_updaters = [
                [copy.deepcopy(val_m)] for val_m in val_metric_updater
            ]

        for val_metric_updater, loader in zip(val_metric_updaters, loaders):
            callbacks_in_val = val_metric_updater
            # build the visualization callback in validation
            if vis_callback:
                callbacks_in_val.extend(vis_callback)

            if inf_loader:
                task_sampler = TaskSampler(
                    task_config={task_name: dict(sampling_factor=1)},
                    method="sample_all",
                    shuffle=False,
                    unions=None,
                )
                task_loader = dict(
                    type="MultitaskInfLoader",
                    loaders={task_name: loader},
                    task_sampler=task_sampler,
                    return_task=True,
                    custom_length=custom_length,
                    iter_once=True,
                )
            else:
                task_loader = dict(
                    type="MultitaskLoader",
                    loaders={task_name: loader},
                    return_task=True,
                    mode="validation",
                    custom_length=custom_length,
                    wrap_batch=True,
                )

            # build validation callback with all components
            val_callback = dict(
                type="Validation",
                val_model=val_model,
                data_loader=task_loader,
                batch_processor=val_batch_processor,
                callbacks=callbacks_in_val,
                interval_by=interval_by,
                val_interval=val_interval,
                log_interval=50,
                val_on_train_end=True,
            )
            val_callbacks.append(val_callback)

    return val_callbacks
