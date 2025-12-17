import copy

from projects.cloudmodel.configs.perception_2d.common import (
    backbone,
    batch_size_per_gpu,
    fast_debug_dataset_number,
    neck,
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

INF = 1e8

# ------------------------------ task setting ------------------------------ #
task = AllTasks.vehicle_side_detection
task_name = task.task_name
task_loss_weight = 1.0


def rm_target_aug(trans, target_type):
    return [d for d in trans if d.get("type") != target_type]


def update_target_aug(trans, target_type, update_content):
    new_data = []
    for item in trans:
        new_item = item.copy()
        if new_item.get("type") == target_type:
            new_item.update(update_content)
        new_data.append(new_item)
    return new_data


def reset_type(datalist, target_type):
    new_data = []
    for item in datalist:
        new_dict = item.copy()
        new_dict["type"] = target_type
        new_dict.update(extend_ignore_region_into_gtbox=True)
        new_data.append(new_dict)
    return new_data


# ---------------------------- task dataloader ----------------------------- #
dataset = HorizonDataset(projects=["mono"], tasks=task)

# remove RandomFlip
train_transforms = rm_target_aug(transforms["train_det"], "RandomFlip")

# add rm_neg_coords=False to Resize
update_content = dict(rm_neg_coords=False)
updated_train_transforms = update_target_aug(
    train_transforms, "Resize", update_content
)

# add truncate_gt=False to RandomCrop
update_content = dict(truncate_gt=False, rm_neg_coords=False)
updated_train_transforms = update_target_aug(
    updated_train_transforms, "RandomCrop", update_content
)

# set dataset type to VehicleSideDenseboxDataset
train_dataset = reset_type(
    dataset.build_dataset(
        mode="train",
        transforms=updated_train_transforms,
        fast_debug=fast_debug_dataset_number,
        use_all_data=use_all_data,
    ),
    "VehicleSideDenseboxDataset",
)

train_dataloader = build_dataloader(
    dataset=train_dataset,
    mode="train",
    batch_size_per_gpu=batch_size_per_gpu["train_det"],
    num_workers=num_workers["train"],
)
val_dataloader = build_dataloader(
    dataset=reset_type(
        dataset.build_dataset(mode="val", transforms=transforms["val"]),
        "VehicleSideDenseboxDataset",
    ),
    mode="val",
    batch_size_per_gpu=batch_size_per_gpu["val_det"],
    num_workers=num_workers["val"],
)

coco_anno_jsons = dataset.get_val_coco_anno_path(task=task, project="mono")
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
inputs["train"].update(gt_tanalphas=None)


def get_model(
    backbone=backbone,
    neck=neck,
    task_name=task_name,
    task_loss_weight=task_loss_weight,
    mode="train",
):
    # deepcopy is essential, otherwise errors about building backbone will
    # come up
    out_strides = [8, 16, 32, 64, 128]
    channel = 256
    stacked_convs = 4
    backbone = copy.deepcopy(backbone["backbone"])
    neck = copy.deepcopy(neck["neck"])
    decoder_transforms = None if mode == "train" else transforms[mode]
    task_model = build_detector_fcos(
        backbone, neck, task_name, task_loss_weight, decoder_transforms, mode
    )
    vehicle_side_head = dict(
        type="VehicleSideFCOSHead",
        num_classes=1,
        in_strides=[4, 8, 16, 32, 64],
        out_strides=out_strides,
        stride2channels={s: channel for s in [4, 8, 16, 32, 64, 128]},
        upscale_bbox_pred=True,
        feat_channels=channel,
        stacked_convs=stacked_convs,
        use_sigmoid=True,
        share_bn=True,
        dequant_output=True,
        int8_output=False,
        share_conv=True,
        use_plain_conv=True,
        use_gn=True,
        use_scale=True,
        add_stride=True,
        node_name=f"{task_name}_head",
    )
    task_model["head"] = vehicle_side_head

    if mode == "train":
        vehicle_side_target = dict(
            type="DynamicVehicleSideFcosTarget",
            center_sampling=True,
            center_sampling_radius=2.5,
            strides=out_strides,
            cls_out_channels=1,
            background_label=1,
            topK=10,
            loss_cls=dict(
                type="FocalLoss",
                loss_name="cls",
                num_classes=2,
                alpha=0.25,
                gamma=2.0,
                loss_weight=1.0,
                reduction="none",
            ),
            loss_reg=dict(
                type="CIoULoss",
                loss_name="reg",
                loss_weight=3.0,
                reduction="none",
            ),
            node_name=f"{task_name}_target",
        )
        task_model["target"] = vehicle_side_target

        vehicle_side_loss = dict(
            type="VehicleSideFCOSLoss",
            cls_loss=dict(
                type="FocalLoss",
                loss_name="loss_cls",
                num_classes=2,
                alpha=0.25,
                gamma=2.0,
                loss_weight=1.0 * task_loss_weight,
            ),
            reg_bbox_loss=dict(
                type="CIoULoss",
                loss_name="loss_bbox",
                loss_weight=2.0 * task_loss_weight,
            ),
            reg_alpha_loss=dict(
                type="L1Loss",
                loss_name="loss_alpha",
                loss_weight=1.0 * task_loss_weight,
                reduce_weight_shape=True,
                skip_neg_weight=True,
            ),
            centerness_loss=dict(
                type="CrossEntropyLoss",
                use_sigmoid=True,
                loss_name="loss_centerness",
                loss_weight=1.0 * task_loss_weight,
            ),
            node_name=f"{task_name}_loss",
        )
        task_model["loss"] = vehicle_side_loss

    else:
        vehicle_side_postprocess = dict(
            type="MultiInputSequential",
            modules=[
                dict(
                    type="VehicleSideFCOSDecoder",
                    num_classes=1,
                    strides=out_strides,
                    test_cfg=dict(
                        score_thr=0.05,
                        nms_pre=1000,
                        nms=dict(name="nms", iou_threshold=0.45),
                        max_per_img=100,
                    ),
                    nms_use_centerness=True,
                    nms_sqrt=False,
                    transforms=decoder_transforms,
                    inverse_transform_key=["scale_factor"],
                    filter_score_mul_centerness=True,
                    int8_output=False,
                    truncate_bbox=False,
                    node_name=f"{task_name}_postprocess",
                ),
                dict(
                    type="VehicleSideFCOSConverter",
                    task_name=task_name,
                    cls_name_mapping={0: "_".join(task_name.split("_")[:-1])},
                    node_name=f"{task_name}_converter",
                ),
            ],
        )
        task_model["postprocess"] = vehicle_side_postprocess

    return task_model


# ------------------------------ task metrics ------------------------------ #
train_metric_updater, val_metric_updater = get_metrics(
    task_name,
    ann_files=coco_anno_jsons,
    save_root=save_root,
)
