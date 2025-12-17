import copy
import json
import os
import random
from collections import OrderedDict

import torch

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.utils.config import Config

cfg_dir = os.path.dirname(__file__)
BASE_CFG = Config.fromfile(os.path.join(cfg_dir, "base.py"))
pipeline_test = BASE_CFG.pipeline_test
training_step = BASE_CFG.training_step
global_desc = BASE_CFG.global_desc

task_name = "task1"
label_name = "task1_label"
num_classes = 1000
batch_size_per_gpu = 2

# 1. task specific graph nodes
nodes = {
    task_name: dict(
        type="OutputModule",
        head=dict(
            type="ToyHead",
            in_channels=64,
            fc_filter=128,
            num_classes=num_classes,
            with_dequant=True,
        ),
        head_parser=dict(type="ToyHeadParser"),
        target=dict(type="ToyTarget"),
        loss=dict(type="ToyLoss"),
        postprocess=None,
        prefix="toy",
    )
}
test_nodes = copy.deepcopy(nodes)
test_nodes[task_name]["head"]["reshape_output"] = False
test_nodes[task_name]["target"] = None
test_nodes[task_name]["loss"] = None
test_nodes[task_name]["postprocess"] = dict(
    type="MultiInputSequential",
    modules=[
        dict(type="ToyPostProcess"),
        dict(
            type="AddDesc",
            per_tensor_desc=[
                # ToyPostProcess with two output tensors
                json.dumps(
                    dict(task=task_name, output_name="output1", **global_desc)
                ),
            ],
        ),
    ],
)

# 2. task specific graph inputs, used to build graph's input variables, i.e.
# the placeholders
inputs = {
    label_name: None  # can be None when GraphModel set `lazy_forward=True`
}


# 3. task specific topology
def topo_builder(nodes, inputs, feats, is_train):
    name2out = OrderedDict()
    out_module = nodes[task_name]
    if is_train:
        label = inputs[label_name]
        name2out[task_name] = out_module(feats, label)
    else:
        name2out[task_name] = out_module(feats)

    return name2out


# 4. task specific data_loader
data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="RandDataset",
        length=16,
        example={
            "img": torch.randn((3, 224, 224)),
            label_name: random.randint(0, num_classes - 1),
        },
        clone=True,
    ),
    # int_infer not use ddp trainer, so sampler should be None
    sampler=dict(type=torch.utils.data.DistributedSampler)
    if training_step != "int_infer"
    else None,
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
)
val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="RandDataset",
        length=10,
        example={
            "img": torch.randn((3, 224, 224)),
            label_name: random.randint(0, num_classes - 1),
        },
        clone=True,
    ),
    # int_infer not use ddp trainer, so sampler should be None
    sampler=dict(type=torch.utils.data.DistributedSampler)
    if training_step != "int_infer"
    else None,
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)

# 5. task specific metric
label_pattern = "^.*%s.*" % label_name
pred_pattern = "^.*%s.*predict" % task_name
metric_updater = dict(
    type="MetricUpdater",
    metrics=[
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
    metric_update_func=update_metric_using_regex(
        per_metric_patterns=[  # corresponding to metrics
            dict(label_pattern=label_pattern, pred_pattern=pred_pattern),
            dict(label_pattern=label_pattern, pred_pattern=pred_pattern),
        ]
    ),
    filter_condition=lambda x: x[1] == task_name,
    step_log_freq=10 if not pipeline_test else 1,
    epoch_log_freq=1,
    log_prefix=task_name,
)
val_metric_updater = copy.deepcopy(metric_updater)
val_metric_updater["log_prefix"] = "Validation " + task_name
