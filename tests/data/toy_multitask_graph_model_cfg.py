import torch


class NaiveStructure(torch.nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        output_module: torch.nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.output_module = output_module

    def forward(self, x):
        feats = self.backbone(x["img"])
        if "label" in x:
            res = self.output_module(feats, x["label"])
        else:
            res = self.output_module(feats)
        return res


class ToyOutModule(torch.nn.Module):
    def __init__(
        self,
        head,
        target=None,
        loss=None,
        postprocess=None,
    ):
        super().__init__()
        self.head = head
        self.target = target
        self.loss = loss
        self.postprocess = postprocess

    def forward(self, x, label=None):
        x = self.head(x)
        output = {}
        if self.loss is not None:
            assert self.target is not None and label is not None
            target = self.target(label)
            output["loss"] = self.loss(x, target)

        if self.postprocess is not None:
            output["pred"] = self.postprocess(x)

        return output


num_classes = 1000
batch_size = 1
backbone = dict(
    type="ToyBackbone",
    strides=(1, 2, 4, 8, 16, 32),
    channels=(3, 8, 8, 16, 32, 64),
    node_name="backbone",
)

head = dict(
    type="ToyHead",
    in_channels=64,
    fc_filter=128,
    num_classes=num_classes,
    with_dequant=True,
    node_name="task_head",
)


def get_task_out_module(task_name, mode):
    return dict(
        type=ToyOutModule,
        head=head,
        target=dict(type="ToyTarget", node_name=f"{task_name}_target")
        if "train" in mode
        else None,
        loss=dict(
            type="ToyLoss",
            node_name=f"{task_name}_loss",
        )
        if "train" in mode
        else None,
        postprocess=dict(
            type="ToyPostProcess",
            node_name=f"{task_name}_postprocess",
        )
        if "train" not in mode
        else None,
    )


def get_task_module(task_name, mode):
    task_out_module = get_task_out_module(task_name, mode)
    return dict(
        type=NaiveStructure,
        backbone=backbone,
        output_module=task_out_module,
    )


task_names = ["task1", "task2", "task3"]
label_suffix = "_label"

model = dict(
    type="MultitaskGraphModel",
    inputs=dict(img=torch.randn((batch_size, 3, 224, 224))),
    task_inputs=dict(
        task1=dict(label=torch.randint(0, num_classes, (batch_size,))),
        task2=dict(label=torch.randint(0, num_classes, (batch_size,))),
        task3=dict(label=torch.randint(0, num_classes, (batch_size,))),
    ),
    task_modules={
        t_name: get_task_module(t_name, "train") for t_name in task_names
    },
    lazy_forward=False,
)

test_model = dict(
    type="MultitaskGraphModel",
    inputs=dict(img=torch.randn((1, 3, 224, 224))),
    task_inputs={t_name: {} for t_name in task_names},
    task_modules={
        t_name: get_task_module(t_name, "test") for t_name in task_names
    },
    lazy_forward=False,
)
