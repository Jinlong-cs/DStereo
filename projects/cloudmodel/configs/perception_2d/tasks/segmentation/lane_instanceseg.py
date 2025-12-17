import copy

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
    build_segmentor_solov2,
)
from projects.cloudmodel.configs.perception_2d.tasks.all_tasks import AllTasks

# ------------------------------ task setting ------------------------------ #
task = AllTasks.lane_instanceseg
task_name = task.task_name
task_loss_weight = 1.0


# desc_attributes = dict(
#     type=[
#         "solid",
#         "dashed",
#         "wide_solid",
#         "wide_dashed",
#         "deceleration_lane",
#         "tidal_lane",
#         "mixed",
#         "Road_teeth",
#         # "botts_dots",
#     ],
#     double_line=["no", "yes"],
#     color=["white", "yellow", "green", "blue", "red", "other"],
#     occlusion=[
#         "full_visible",
#         "occluded",
#         "heavily_occluded",
#         "snow_occluded",
#     ],
# )
desc_attributes = dict(
    type=[
        "solid",
        "dashed",
        "wide_solid",
        "wide_dashed",
        "mixed",
        "Road_teeth",
        # "botts_dots",
    ],
    double_line=["no", "yes"],
    color=["white", "yellow", "green", "blue", "red", "other"],
    occlusion=[
        "full_visible",
        "occluded",
        "heavily_occluded",
        "snow_occluded",
    ],
    deceleration_lane=["no", "yes"],
    tidal_lane=["no", "yes"],
)

cls_name_mapping = {
    k: {i: v_i for i, v_i in enumerate(v)} for k, v in desc_attributes.items()
}
attr_name2num = {k: len(v) for k, v in desc_attributes.items()}
num_classes = attr_name2num.pop("type")


# ---------------------------- task dataloader ----------------------------- #
dataset = HorizonDataset(projects=["mono", "pilot"], tasks=task)

train_dataloader = build_dataloader(
    dataset=dataset.build_dataset(
        mode="train",
        transforms=transforms["train_insseg"],
        fast_debug=fast_debug_dataset_number,
        use_all_data=use_all_data,
    ),
    mode="train",
    batch_size_per_gpu=batch_size_per_gpu["train_seg"],
    num_workers=num_workers["train"],
)
val_dataloader = build_dataloader(
    dataset=dataset.build_dataset(
        mode="val", transforms=transforms["val_insseg"]
    ),
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
    train=copy.deepcopy(task_inputs["train_insseg"]),
    val=copy.deepcopy(task_inputs["val_insseg"]),
    test=dict(),
    infer=dict(),
)


def get_model(backbone=backbone, neck=neck, task_name=task_name, mode="train"):
    head_cfg = dict(
        desc_attributes=desc_attributes,
        cls_name_mapping=cls_name_mapping,
        attr_name2num=attr_name2num,
        num_classes=num_classes,
        in_channels=list(neck["stride2channels"].values())[0],
        pos_scale=0.2,
        strides=[4],
        scale_ranges=[(1, 10000)],
        num_grids=[100],
        max_pos_num=100,
        use_ignore_region=True,
    )
    # deepcopy is essential, otherwise errors about building backbone will
    # come up
    backbone = copy.deepcopy(backbone["backbone"])
    neck = copy.deepcopy(neck["neck"])

    task_model = build_segmentor_solov2(
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
    attributes=desc_attributes,
    ignore_index=255,
)
