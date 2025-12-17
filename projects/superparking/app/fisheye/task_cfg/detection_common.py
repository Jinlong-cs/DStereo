import torch

from hat.data.collates.collates import collate_2d
from ..common import (
    input_size,
    pipeline_test_dump,
    test_num_workers,
    train_num_workers,
    val_num_workers,
)

mode_to_num_workers = {
    "train": train_num_workers,
    "val": val_num_workers,
    "test": test_num_workers,
}


def get_train_transform(use_mosaic_aug, use_mixup_aug):
    train_transforms = [
        dict(
            type="RandomCrop",
            center_shake=[60, 60, 300, 300],
            center_crop_prob=0.5,
            min_area=-1,
            min_iou=0.3,
            wh_ratio_range=(0.5, 1.0),
            repeat_times=100,
            discriminate_ignore_classes=True,
            without_background=True,
            crop_around_gt=True,
        ),
        dict(
            type="Resize",
            img_scale=input_size,
            keep_ratio=True,
            pad_to_keep_ratio=True,
        ),
        dict(type="RandomFlip", px=0.5),
        dict(type="ToTensor", to_yuv=False),
        dict(type="Pad", divisor=64),
        dict(type="RenameKeys", keys=["imgs|img"]),
        dict(type="DeleteKeys", keys=["before_pad_shape"]),
    ]

    mosaic_cfg = dict(
        type="DetMosaic",
        img_scale=input_size,
        center_ratio_range=(0.75, 1.25),
        p=0.5,
    )

    mixup_cfgs = {
        "yolox": dict(
            type="DetYOLOXMixUp",
            img_scale=input_size,
            p=0.5,
        ),
        "yolov5": dict(
            type="DetYOLOv5MixUp",
            p=0.5,
        ),
    }

    mix_trans_insert_pos = 3
    mix_transform_cfg = None
    if use_mosaic_aug and use_mixup_aug:
        mix_transform_cfg = dict(
            type="RandomSelectOne",
            transforms=[
                mosaic_cfg,
                mixup_cfgs[use_mixup_aug],
            ],
            p_trans=[0.9, 0.1],
            p=1.0,
        )
    elif use_mosaic_aug:
        mix_transform_cfg = mosaic_cfg
    elif use_mixup_aug:
        mix_transform_cfg = mixup_cfgs[use_mixup_aug]

    if mix_transform_cfg:
        train_transforms.insert(mix_trans_insert_pos, mix_transform_cfg)
        train_transforms.insert(
            mix_trans_insert_pos + 1,
            dict(
                type="Resize",
                img_scale=input_size,
                keep_ratio=True,
                pad_to_keep_ratio=True,
            ),
        )
    return train_transforms


val_transforms = [
    dict(
        type="Resize",
        img_scale=input_size,
        keep_ratio=False,
    ),
    dict(type="ToTensor", to_yuv=False),
    dict(type="RenameKeys", keys=["imgs|img"]),
]


def get_2d_detection_dataloader(mode, dataset, batch_size):
    num_workers = eval(f"{mode}_num_workers")
    data_loader = dict(
        type=torch.utils.data.DataLoader,
        dataset=dataset,
        collate_fn=collate_2d,
        batch_size=batch_size[mode],
        num_workers=mode_to_num_workers[mode],
        pin_memory=False,
    )

    if mode == "train":
        target_dataset_type = "DistributedComposeRandomDataset"
        assert (
            dataset["type"] == target_dataset_type
            or dataset["datasets"][0]["type"] == target_dataset_type
        )
        data_loader.update(
            dict(
                persistent_workers=num_workers > 0,
                multiprocessing_context=None
                if pipeline_test_dump
                else "spawn",
            )
        )
    else:
        data_loader.update(
            dict(
                drop_last=False,
            )
        )
    return data_loader
