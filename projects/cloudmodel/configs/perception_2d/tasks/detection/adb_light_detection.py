import copy

from projects.cloudmodel.configs.perception_2d.common import (  # backbone,; neck,
    batch_size_per_gpu,
    fast_debug_dataset_number,
    num_workers,
    save_root,
    task_inputs,
    transforms,
    use_all_data,
)
from projects.cloudmodel.configs.perception_2d.data import HorizonDataset
from projects.cloudmodel.configs.perception_2d.data import (
    build_singletask_dataloader as build_dataloader,
)
from projects.cloudmodel.configs.perception_2d.metrics import get_metrics
from projects.cloudmodel.configs.perception_2d.models import (
    build_detector_fcos,
)
from projects.cloudmodel.configs.perception_2d.tasks.all_tasks import AllTasks

# ------------------------------ task setting ------------------------------ #
task = AllTasks.adb_light_detection
task_name = task.task_name
task_loss_weight = 1.0

# ---------------------------- task dataloader ----------------------------- #
dataset = HorizonDataset(projects=["mono", "unknown"], tasks=task)

train_dataloader = build_dataloader(
    dataset=dataset.build_dataset(
        mode="train",
        transforms=transforms["train_det"],
        fast_debug=fast_debug_dataset_number,
        use_all_data=use_all_data,
    ),
    mode="train",
    batch_size_per_gpu=batch_size_per_gpu["train_det"],
    num_workers=num_workers["train"],
)
val_dataloader = build_dataloader(
    dataset=dataset.build_dataset(mode="val", transforms=transforms["val"]),
    mode="val",
    batch_size_per_gpu=batch_size_per_gpu["val_det"],
    num_workers=num_workers["val"],
)
coco_anno_jsons = dataset.get_val_coco_anno_path(task=task, project="unknown")
test_dataloader = build_dataloader(
    dataset=dataset.build_dataset(mode="test", transforms=transforms["test"]),
    mode="test",
    batch_size_per_gpu=batch_size_per_gpu["test"],
    num_workers=num_workers["test"],
)

# ------------------------------ task model -------------------------------- #
# This block defines the task network which is built on one pre-defined network
# in the `models` directory
inputs = dict(
    train=copy.deepcopy(task_inputs["train_det"]),
    val=copy.deepcopy(task_inputs["val_det"]),
    test=copy.deepcopy(task_inputs["test"]),
    infer=dict(),
)

backbone_cfg = {"name": "hat_swin", "arch": "base_w12"}
neck_cfg = {"name": "pafpn"}
stride2channels = {s: 320 for s in [4, 8, 16, 32, 64]}
from projects.cloudmodel.configs.perception_2d.models import (
    build_backbone_neck,
)

backbone, neck = build_backbone_neck(backbone_cfg, neck_cfg, stride2channels)

head_cfg = {
    "in_strides": [4, 8, 16, 32, 64],
    "out_strides": [4, 8, 16, 32, 64],
    "stride2channels": {s: 320 for s in [4, 8, 16, 32, 64]},
    "add_stride": False,
    "stacked_convs": 4,
    "num_classes": 4,
    "cls_name_mapping": {
        0: "head_light",
        1: "tail_light",
        2: "street_light",
        3: "reflection_point",
    },
}


def get_model(
    backbone=backbone,
    neck=neck,
    task_name=task_name,
    task_loss_weight=task_loss_weight,
    mode="train",
):
    # deepcopy is essential, otherwise errors about building backbone will
    # come up
    backbone = copy.deepcopy(backbone["backbone"])
    neck = copy.deepcopy(neck["neck"])
    decoder_transforms = None if mode == "train" else transforms[mode]
    task_model = build_detector_fcos(
        backbone,
        neck,
        task_name,
        task_loss_weight,
        decoder_transforms,
        mode,
        head_cfg,
    )
    return task_model


# ------------------------------ task metrics ------------------------------ #
train_metric_updater, val_metric_updater = get_metrics(
    task_name,
    ann_files=coco_anno_jsons,
    save_root=save_root,
)
