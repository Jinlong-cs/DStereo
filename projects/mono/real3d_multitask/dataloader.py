from importlib import import_module

from common import multitask

from hat.core.task_sampler import TaskSampler

task_names = [t["name"] for t in multitask]
TASK_CONFIGS = [import_module(t) for t in task_names]

task_sampler_configs = {
    "vehicle_detection": dict(sampling_factor=2),
    "face_detection": dict(sampling_factor=1),
    "plate_detection": dict(sampling_factor=1),
}
task_sampler_configs["chosen_tasks"] = [
    task_names,
]
task_sampler = TaskSampler(
    task_config=task_sampler_configs,
    method="sample_one",
)

loaders = {T.task_name: T.dataloader for T in TASK_CONFIGS}

dataloader = dict(
    type="MultitaskInfLoader",
    loaders=loaders,
    task_sampler=task_sampler,
    return_task=True,
    __build_recursive=False,
)
