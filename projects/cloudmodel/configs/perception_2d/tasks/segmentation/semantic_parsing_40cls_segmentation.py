import copy

from hat.core.proj_spec.descs import get_default_parsing_desc
from hat.core.proj_spec.parsing import get_default_parsing_labels
from projects.cloudmodel.configs.perception_2d.common import (
    backbone,
    batch_size_per_gpu,
    fast_debug_dataset_number,
    neck,
    num_workers,
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
    build_segmentor_semanticfpn,
)
from projects.cloudmodel.configs.perception_2d.tasks.all_tasks import AllTasks

# ------------------------------ task setting ------------------------------ #
task = AllTasks.semantic_parsing_40cls_segmentation
task_name = task.task_name
task_loss_weight = 1.0
ignore_index = 255
num_classes = 40

parsing_desc = get_default_parsing_desc(desc_id="gl_40")
desc_labels = get_default_parsing_labels("gl_40")

seg_class = [
    "road",
    "sidewalk",
    "vegetation",
    "terrain",
    "pole",
    "traffic_sign",
    "traffic_light",
    "sign_line",
    "lane_marking",
    "person",
    "rider",
    "bicycle",
    "motorcycle",
    "tricycle",
    "car",
    "truck",
    "bus",
    "train",
    "building",
    "fence",
    "sky",
    "traffic_cone",
    "bollard",
    "guide_post",
    "crosswalk_line",
    "traffic_arrow",
    "guide_line",
    "stop_line",
    "slow_down_tria",
    "speed_sign",
    "diamond",
    "bicycle_sign",
    "speedbumps",
    "no_forward",
    "parking_rod",
    "parking_lock",
    "traversable",
    "untraversable",
    "mask",
    "other",
]

class_weight = [
    1.0,
    2.0,
    1.0,
    1.0,
    2.0,
    2.0,
    2.0,
    2.0,
    1.0,
    1.0,
    1.0,
    2.0,
    2.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    2.0,
    2.0,
    1.0,
    1.0,
    1.0,
    2.0,
    2.0,
    1.0,
    2.0,
    2.0,
    1.0,
    2.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
]
# ---------------------------- task dataloader ----------------------------- #
dataset = HorizonDataset(projects=["mono", "pilot"], tasks=task)

train_dataloader = build_dataloader(
    dataset=dataset.build_dataset(
        mode="train",
        transforms=transforms["train_seg"],
        fast_debug=fast_debug_dataset_number,
        use_all_data=use_all_data,
    ),
    mode="train",
    batch_size_per_gpu=batch_size_per_gpu["train_seg"],
    num_workers=num_workers["train"],
)
val_dataloader = build_dataloader(
    dataset=dataset.build_dataset(mode="val", transforms=transforms["val"]),
    mode="val",
    batch_size_per_gpu=batch_size_per_gpu["val_seg"],
    num_workers=num_workers["val"],
)

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
    train=copy.deepcopy(task_inputs["train_seg"]),
    val=copy.deepcopy(task_inputs["val_seg"]),
    test=dict(),
    infer=dict(),
)

head_in_strides = [4, 8, 16]


def get_model(backbone=backbone, neck=neck, task_name=task_name, mode="train"):
    head_cfg = dict(
        task_loss_weight=task_loss_weight,
        ignore_index=ignore_index,
        num_classes=num_classes,
        parsing_desc=parsing_desc,
        class_weight=class_weight,
        in_strides=head_in_strides,
        in_channels=[neck["stride2channels"][s] for s in head_in_strides],
    )
    # deepcopy is essential, otherwise errors about building backbone will
    # come up
    backbone = copy.deepcopy(backbone["backbone"])
    neck = copy.deepcopy(neck["neck"])

    task_model = build_segmentor_semanticfpn(
        backbone,
        neck,
        task_name,
        mode,
        head_cfg,
    )

    return task_model


# ------------------------------ task metrics ------------------------------ #
train_metric_updater, val_metric_updater = get_metrics(
    task_name,
    seg_class=seg_class,
    ignore_index=ignore_index,
)
