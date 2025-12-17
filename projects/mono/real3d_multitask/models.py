from importlib import import_module

import torch
from common import multitask

task_names = [t["name"] for t in multitask]
TASK_CONFIGS = [import_module(t) for t in task_names]

inputs = dict(img=torch.zeros((1, 3, 192, 960)))
deploy_inputs = dict(img=torch.zeros((1, 3, 192, 960)))
opt_inputs = dict(
    train=dict(),
    val=dict(),
    test=dict(),
)

model = dict(
    type="MultitaskGraphModel",
    inputs=inputs,
    opt_inputs=opt_inputs["train"],
    task_inputs={T.task_name: T.inputs["train"] for T in TASK_CONFIGS},
    task_modules={T.task_name: T.model for T in TASK_CONFIGS},
    funnel_modules=None,
    flatten_outputs=True,
    lazy_forward=False,
)

deploy_model = dict(
    type="MultitaskGraphModel",
    inputs=inputs,
    # opt_inputs=opt_inputs["test"],
    task_inputs={T.task_name: T.inputs["test"] for T in TASK_CONFIGS},
    task_modules={T.task_name: T.deploy_model for T in TASK_CONFIGS},
    # funnel_modules=None,
    # flatten_outputs=True,
    lazy_forward=False,
)
