from collections import OrderedDict
from copy import deepcopy
from importlib import import_module

import torch
from common import input_size, tasks

task_names = [t["name"] for t in tasks]
TASK_CONFIGS = [import_module(t) for t in task_names]

inputs = dict(img=torch.zeros((1, 3, *input_size)))
deploy_inputs = dict(img=torch.zeros((1, 3, *input_size)))
opt_inputs = dict(
    train=dict(),
    val=dict(),
    test=dict(),
)

traced_inputs = dict(
    img=torch.randn((1, 3, *input_size)),
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

val_decoders = {}
converted_decoders = OrderedDict(
    {
        (tuple(task_names), group): decoder
        for group, (task_names, decoder) in val_decoders.items()
    }
)

val_mode = dict(
    type="MultitaskGraphModel",
    inputs=inputs,
    opt_inputs=opt_inputs["val"],
    task_inputs={T.task_name: T.inputs["val"] for T in TASK_CONFIGS},
    task_modules={T.task_name: T.model for T in TASK_CONFIGS},
    funnel_modules=converted_decoders,
    flatten_outputs=False,
    lazy_forward=False,
)

val_model = deepcopy(val_mode)
