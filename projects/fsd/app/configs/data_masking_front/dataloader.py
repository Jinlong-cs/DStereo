from importlib import import_module

from common import multitask

from hat.core.task_sampler import TaskSampler

task_names = [t["name"] for t in multitask]
TASK_CONFIGS = [import_module(t) for t in task_names]

task_sampler_configs = {
    T.task_name: dict(sampling_factor=1) for T in TASK_CONFIGS
}

task_sampler = TaskSampler(
    task_config=task_sampler_configs,
    method="sample_all",
)

loaders = {T.task_name: T.dataloader for T in TASK_CONFIGS}

data_loader = dict(
    type="MultitaskInfLoader",
    loaders=loaders,
    task_sampler=task_sampler,
    return_task=True,
    __build_recursive=False,
)
