from importlib import import_module

import torch
from common import img_height, img_width, multitask

task_names = [t["name"] for t in multitask]
TASK_CONFIGS = [import_module(t) for t in task_names]

inputs = dict(img=torch.zeros((1, 3, img_height, img_width)))
deploy_inputs = dict(img=torch.zeros((1, 3, img_height, img_width)))
opt_inputs = dict(
    train=dict(),
    val=dict(),
    deploy=dict(),
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

val_model = dict(
    type="MultitaskGraphModel",
    inputs=inputs,
    opt_inputs=opt_inputs["val"],
    task_inputs={T.task_name: T.inputs["val"] for T in TASK_CONFIGS},
    task_modules={T.task_name: T.val_model for T in TASK_CONFIGS},
    funnel_modules=None,
    flatten_outputs=False,
    lazy_forward=False,
)


deploy_model = dict(
    type="MultitaskGraphModel",
    inputs=inputs,
    opt_inputs=opt_inputs["deploy"],
    task_inputs={T.task_name: T.inputs["deploy"] for T in TASK_CONFIGS},
    task_modules={T.task_name: T.deploy_model for T in TASK_CONFIGS},
    funnel_modules=None,
    flatten_outputs=True,
    lazy_forward=False,
)
