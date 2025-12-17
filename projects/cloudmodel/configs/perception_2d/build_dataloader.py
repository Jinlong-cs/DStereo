from hat.core.task_sampler import TaskSampler


def build_multitask_dataloader(task_configs, mode="train", inf_loader=False):
    """This function builds a multitask dataloader for all tasks, each of which
     generally includes a training dataloader, a validation dataloader and
     sometimes a test dataloader. See hat/data/dataloaders/multitask_loader.py.

    Args:
        task_configs: define each task and include task dataloaders.
        mode: specifies what kind of dataloader you want, e.g., "train", "val".

    Returns:
        A multitask dataloader.
    """
    # build task sampler
    task_sampler = TaskSampler(
        task_config={
            t.task_name: dict(sampling_factor=1) for t in task_configs
        },
        method="sample_all",
        shuffle=False,
        unions=None,
    )

    if mode == "train":
        loaders = {t.task_name: t.train_dataloader for t in task_configs}
    elif mode == "val":
        loaders = {t.task_name: t.val_dataloader for t in task_configs}
    elif mode == "test":
        loaders = {t.task_name: t.test_dataloader for t in task_configs}
    else:
        raise ValueError(
            f"mode should be `train`, `val` or `test`, but got {mode}."
        )

    # build MultitaskLoader
    if inf_loader:
        dataloader = dict(
            type="MultitaskInfLoader",
            loaders=loaders,
            task_sampler=task_sampler,
            return_task=True,
        )
    else:
        dataloader = dict(
            type="MultitaskLoader",
            loaders=loaders,
            task_sampler=task_sampler,
            mode="max_size" if mode == "train" else "validation",
            return_task=True,
            custom_length=None,
            wrap_batch=True,
        )

    return dataloader
