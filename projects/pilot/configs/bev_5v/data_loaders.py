from importlib import import_module

from hat.core.task_sampler import TaskSampler
from projects.pilot.configs.bev_5v.common import tasks

task_names = [t["name"] for t in tasks]
TASK_CONFIGS = [import_module(t) for t in task_names]

task_sampler_configs = {
    T.task_name: dict(sampling_factor=1) for T in TASK_CONFIGS
}

imp_task_names = [t["name"] for t in tasks if t.get("important", False)]

task_sampler_configs["chosen_tasks"] = [
    imp_task_names,
    imp_task_names,
    imp_task_names,
    task_names,
]

task_sampler = TaskSampler(
    task_config=task_sampler_configs,
    method="sample_repeat",
)

loaders = {
    i.task_name: i.get_train_dataloader()
    if hasattr(i, "get_train_dataloader")
    else i.data_loader
    for i in TASK_CONFIGS
}

data_loader = dict(
    type="MultitaskInfLoader",
    loaders=loaders,
    task_sampler=task_sampler,
    return_task=True,
    __build_recursive=False,
)
