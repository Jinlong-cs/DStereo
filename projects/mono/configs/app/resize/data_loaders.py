import copy

from common import custom_loader_length
from lib.utils import get_list
from schedule import TASK_CONFIGS, tasks

from hat.core.task_sampler import TaskSampler

task_sampler_configs = {
    T.task_name: dict(sampling_factor=1) for T in TASK_CONFIGS
}

imp_task_names = [t["name"] for t in tasks if t.get("important", False)]
normal_task_names = [t["name"] for t in tasks if not t.get("important", False)]

important_sample_weight = 3

task_sampler_configs["chosen_tasks"] = [
    imp_task_names,
]

if len(normal_task_names) > 0:
    task_sampler_configs["chosen_tasks"] = (
        task_sampler_configs["chosen_tasks"] * important_sample_weight
        + normal_task_names
    )

task_sampler = TaskSampler(
    task_config=task_sampler_configs,
    method="sample_repeat",
)


loaders = {T.task_name: T.data_loader for T in TASK_CONFIGS}

data_loader = dict(
    type="MultitaskInfLoader",
    loaders=loaders,
    task_sampler=task_sampler,
    return_task=True,
    __build_recursive=False,
)

val_loaders = dict()
for config in TASK_CONFIGS:
    if isinstance(config.val_data_loader, list):
        assert len(config.task_name_list) == len(config.val_data_loader)
        for i in range(len(config.val_data_loader)):
            val_loaders[config.task_name_list[i]] = config.val_data_loader[i]
    else:
        val_loaders[config.task_name] = config.val_data_loader

val_task_config = dict()
for config in TASK_CONFIGS:
    if hasattr(config, "task_name_list"):
        for task_name in config.task_name_list:
            val_task_config[task_name] = dict(sampling_factor=1)
    else:
        val_task_config[config.task_name] = dict(sampling_factor=1)

val_task_sampler = TaskSampler(
    task_config=val_task_config, method="sample_all"
)

val_data_loader = dict(
    type="MultitaskInfLoader",
    loaders=val_loaders,
    task_sampler=val_task_sampler,
    return_task=True,
    __build_recursive=False,
    iter_once=True,
    custom_length=custom_loader_length,
)


def wrap_multitask_loader(taskname, loader):
    ts_configs = {
        taskname: dict(sampling_factor=1),
        "chosen_tasks": [[taskname]],
    }
    ts = TaskSampler(
        task_config=ts_configs,
        method="sample_repeat",
    )

    mt_loader = copy.deepcopy(val_data_loader)
    mt_loader["loaders"] = {taskname: loader}
    mt_loader["task_sampler"] = ts
    return mt_loader


aidi_eval_dataloaders = get_list(
    "aidi_eval_loader",
    TASK_CONFIGS,
    merge_list=True,
    wrap_fn=wrap_multitask_loader,
)
