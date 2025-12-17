import copy
from collections import OrderedDict

import torch
from hatbc.workflow.trace import GraphTracer, make_traceable

from hat.utils.apply_func import _as_list
from tests.data.toy_modules import ToyBackbone

GraphTracer.register_basic_types(ToyBackbone)


@make_traceable
def forward_backbone_traceable(img, backbone):
    feats = backbone(img)
    return feats


def forward_backbone(img, backbone):
    feats = backbone(img)
    return feats


def get_topo_builder(is_train, out_feat_idx=None, use_traceable_func=False):
    def _build_topo(nodes, inputs):
        #
        # Topology:
        #
        # (img) -> backbone -> neck -> task1_out_module -> (task1 outputs)
        #                         \ -> task2_out_module -> (task2 outputs)
        #
        name2out = OrderedDict()
        img = inputs["img"]
        backbone = nodes["backbone"]
        if use_traceable_func:
            feats = forward_backbone_traceable(img, backbone)
        else:
            feats = forward_backbone(img, backbone)

        for task in task_names:
            out_module = nodes[task]
            if is_train:
                label_name = task + label_suffix
                label = inputs[label_name]
                name2out[task] = out_module(feats, label)
            else:
                name2out[task] = out_module(feats)

        # special case: output neck feats as tracking feats ONLY when test
        if out_feat_idx is not None:
            for i in _as_list(out_feat_idx):
                name2out["backbone_feat_%d" % i] = feats[i]

        return name2out

    return _build_topo


num_classes = 1000
batch_size = 1
backbone = dict(
    type="ToyBackbone",
    strides=(1, 2, 4, 8, 16, 32),
    channels=(3, 8, 8, 16, 32, 64),
)
out_module = dict(
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
export_out_module = copy.deepcopy(out_module)
export_out_module["target"] = None
export_out_module["loss"] = None
export_out_module["postprocess"] = dict(type="ToyPostProcess")

task_names = ["task1", "task2", "task3"]
label_suffix = "_label"

model = dict(
    type="GraphModel",
    nodes=dict(
        backbone=backbone,
        task1=out_module,
        task2=copy.deepcopy(out_module),
        task3=copy.deepcopy(out_module),
    ),
    inputs=dict(
        img=torch.randn((batch_size, 3, 224, 224)),
        task1_label=torch.randint(0, num_classes, (batch_size,)),
        task2_label=torch.randint(0, num_classes, (batch_size,)),
        task3_label=torch.randint(0, num_classes, (batch_size,)),
    ),
    topology_builder=get_topo_builder(is_train=True),
    lazy_forward=False,
)
test_model = dict(
    type="GraphModel",
    nodes=dict(
        backbone=backbone,
        task1=export_out_module,
        task2=copy.deepcopy(export_out_module),
        task3=copy.deepcopy(export_out_module),
    ),
    inputs=dict(
        # shape may be different from train img
        img=torch.randn((1, 3, 224, 224))
    ),
    topology_builder=get_topo_builder(is_train=False, out_feat_idx=[0]),
    lazy_forward=False,
)
test_traceable_model = dict(
    type="GraphModel",
    nodes=dict(
        backbone=backbone,
        task1=out_module,
        task2=copy.deepcopy(out_module),
        task3=copy.deepcopy(out_module),
    ),
    inputs=dict(
        img=torch.randn((batch_size, 3, 224, 224)),
        task1_label=torch.randint(0, num_classes, (batch_size,)),
        task2_label=torch.randint(0, num_classes, (batch_size,)),
        task3_label=torch.randint(0, num_classes, (batch_size,)),
    ),
    topology_builder=get_topo_builder(is_train=True, use_traceable_func=True),
    lazy_forward=False,
)
