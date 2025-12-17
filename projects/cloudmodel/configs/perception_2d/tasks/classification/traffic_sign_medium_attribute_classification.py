import copy

import torch

from projects.cloudmodel.configs.perception_2d.common import (
    backbone,
    batch_size_per_gpu,
    fast_debug_dataset_number,
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
    build_attribute_classifier,
)
from projects.cloudmodel.configs.perception_2d.tasks.all_tasks import AllTasks

# ------------------------------ task setting ------------------------------ #
task = AllTasks.traffic_sign_medium_attribute_classification
task_name = task.task_name

task_loss_weight = 1.0

num_classes = 61
attr_type_list = ["cn_medium_type", "occlusion", "ignore", "Inside"]
attr_type_value = [51, 4, 2, 4]
ignore_idx = [56]
confidence_idx = []
occlusion_idx = [54]
type_ignore_bad = True  # False #
ignore_idxs = ignore_idx + confidence_idx + occlusion_idx

name_dict = {
    "cn_medium_type": [
        "I_IndicationRectangle",
        "U_Unknown",
        "Prohibit_I",
        "RouteID_Other",
        "I_IndicationCircle",
        "C_Unknown",
        "PSL_Unknown",
        "P_Unknown1",
        "I_RectangleMisc",
        "ProhibitRemove",
        "TravelSign_BrownWhite",
        "False_Positive",
        "PSL_Other",
        "I_Other_rectangle",
        "GSGW_Other",
        "MSL_Other",
        "TSBW_Other",
        "W_Other",
        "MSL_Unknown",
        "sign_Construction",
        "RSBW_Other",
        "PR_Other",
        "RouteID_Unknown",
        "RouteID",
        "GuideSign_GreenWhite",
        "Min_SpeedLim",
        "I_Other_circle",
        "GSWB_Unknown",
        "TSBW_Unknown",
        "P_ElectricalSpeedLimit",
        "W_Unknown",
        "GSWB_Misc",
        "RoadSign_BlueWhite",
        "I_Unknown_circle",
        "P_Other1",
        "GSWB_Other",
        "C_Other",
        "PR_Unknown",
        "W_Warning",
        "I_Unknown_rectangle",
        "sign_back",
        "GSGW_Misc",
        "P_SpeedLimit",
        "P_Unknow_SpeedLimele",
        "RSBW_Misc",
        "GuideSign_WhiteBlack",
        "W_Misc",
        "GSGW_Unknown",
        "O_Other",
        "P_Other_SpeedLimele",
        "RSBW_Unknown",
    ],
    "occlusion": [
        "full_visible",
        "occluded",
        "heavily_occluded",
        "invisible",
    ],
    "ignore": ["no", "yes"],
    "Inside": ["no", "yes", "outside", "inside"],
}

# 标注返回的json中的关键字
task_key_word = "traffic_sign"


# ---------------------------- task dataloader ----------------------------- #
dataset = HorizonDataset(projects=["unknown"], tasks=task)

train_dataloader = build_dataloader(
    dataset=dataset.build_dataset(
        mode="train",
        transforms=transforms["train_cls"],  # task_transforms,
        fast_debug=fast_debug_dataset_number,
        use_all_data=use_all_data,
    ),
    mode="train",
    batch_size_per_gpu=batch_size_per_gpu["train_cls"],
    num_workers=num_workers["train"],
    is_iterabledataset=True,
)
val_dataloader = build_dataloader(
    dataset=dataset.build_dataset(
        mode="val", transforms=transforms["val_cls"]
    ),
    mode="val",
    batch_size_per_gpu=batch_size_per_gpu["val_cls"],
    num_workers=num_workers["val"],
    is_iterabledataset=True,
)

# ------------------------------ task model -------------------------------- #
# This block defines the task network which is built on one pre-defined network
# in the `models` directory
inputs = dict(
    train=copy.deepcopy(task_inputs["train_cls"]),
    val=copy.deepcopy(task_inputs["val_cls"]),
    test=copy.deepcopy(task_inputs["test"]),
    infer=dict(),
)


def get_model(
    backbone=backbone,
    task_name=task_name,
    mode="train",
    **kwargs,
):
    bn_kwargs = dict(eps=1e-5, momentum=0.1)
    in_channel_list = backbone["stride2channels"]
    backbone = copy.deepcopy(backbone["backbone"])
    neck = dict(
        neck=torch.nn.Identity(),
        stride2channels=in_channel_list,
    )
    task_model = build_attribute_classifier(
        backbone,
        neck,
        task_name,
        bn_kwargs,
        num_classes,
        mode,
        attr_type_list,
        attr_type_value,
        ignore_idxs,
    )
    return task_model


# ------------------------------ task metrics ------------------------------ #
train_metric_updater, val_metric_updater = get_metrics(
    task_name,
    attr_type_list=attr_type_list,
    attr_type_value=attr_type_value,
    ignore_idxs=ignore_idxs,
)
