# This script defines those variables that are used across the entire project.
import os
import time

# --------------------------------- env --------------------------------- #
pipeline_test = os.environ.get("HAT_PIPELINE_TEST", "0") == "1"
# --------------------------------- Cluster --------------------------------- #
is_local_train = not os.path.exists("/running_package")
bucket_root = "/horizon-bucket" if is_local_train else "/bucket/input"
cur_time = time.strftime("%Y%m%d%H%M%S")
save_root = f"./work_dirs/exp_{cur_time}" if is_local_train else "/job_log"


# --------------------------- Define model inputs --------------------------- #
# common inputs to multitask graph model
common_inputs = dict(
    img=None,
    img_id=None,
    img_name=None,
    img_height=None,
    img_width=None,
    img_shape=None,
    scale_factor=None,
    resized_shape=None,
    keep_ratio=None,
    color_space=None,
    layout=None,
    scale=None,
    scale_idx=None,
)

task_inputs = {
    "train_det": dict(
        gt_bboxes=None,
        gt_classes=None,
        crop_offset=None,
    ),
    "train_seg": dict(
        gt_seg=None,
    ),
    "train_cls": dict(
        labels=None,
    ),
    "train_insseg": dict(
        gt_seg=None,
        gt_labels=None,
    ),
    "val_det": dict(
        gt_bboxes=None,
        gt_classes=None,
        orig_img=None,
    ),
    "val_seg": dict(
        gt_seg=None,
        orig_img=None,
        orig_gt_seg=None,
    ),
    "val_insseg": dict(
        gt_seg=None,
        gt_labels=None,
        orig_img=None,
        orig_gt_seg=None,
    ),
    "val_cls": dict(),
    "test": dict(orig_img=None),
}

# -------------------------------- Transforms ------------------------------- #
resize_hw = (640, 1024)
resize_hw_val = (1280, 2048)
resize_hw_test = (1280, 2048)

# These statistics are used for EfficientNet pre-trained weights
mean, std = [123.675, 116.28, 103.53], [58.395, 57.12, 57.375]
# These statistics are used for VargNet pre-trained weights
# mean, std = [128.0, 128.0, 128.0], [128.0, 128.0, 128.0]
# roi model sizes
cls_roi_resize_hw = (224, 224)  # (128, 128)

transforms = {
    "train_det": [
        dict(type="Resize", img_scale=resize_hw, ratio_range=(0.9, 3.5)),
        dict(
            type="RandomCrop",
            size=resize_hw,
            min_area=-1,
            min_iou=0.2,
            crop_around_gt=True,
            repeat_times=10,
            inclusion_rate=0.3,
        ),
        dict(type="RandomFlip", px=0.5),
        dict(type="ToTensor", to_yuv=False),
        dict(type="Normalize", mean=mean, std=std),
        dict(type="Pad", size=resize_hw, pad_val=0),
    ],
    "train_seg": [
        dict(
            type="SegResizeAffine", img_scale=resize_hw, ratio_range=(0.5, 4.0)
        ),
        dict(
            type="SegRandomCrop",
            size=resize_hw,
            cat_max_ratio=0.75,
            ignore_index=255,
        ),
        dict(type="RandomFlip", px=0.5),
        dict(type="ToTensor", to_yuv=False),
        dict(type="Normalize", mean=mean, std=std),
        dict(type="Pad", size=resize_hw, pad_val=0),
    ],
    "train_insseg": [
        dict(type="ReformatLanePolygon"),
        dict(
            type="PolygonToMask",
            filter_emtpy=True,
            replace_gt_seg=True,
            replace_orig_gt_seg=True,
            add_bbox=False,
        ),
        dict(
            type="SegResizeAffine",
            img_scale=resize_hw,
            ratio_range=(0.5, 2.5),
        ),
        dict(
            type="SegRandomCrop",
            size=resize_hw,
            cat_max_ratio=0.999,
            ignore_index=255,
        ),
        dict(
            type="PolygonToMask",
            replace_gt_seg=True,
            replace_orig_gt_seg=False,
        ),
        dict(type="DeleteKeys", keys=["gt_polygons"]),
        dict(type="RandomFlip", px=0.5),
        dict(type="ToTensor", to_yuv=False),
        dict(type="Normalize", mean=mean, std=std),
        dict(type="Pad", size=resize_hw, pad_val=0, seg_pad_val=0),
    ],
    "train_cls": [
        dict(type="RandomFlip", px=0.5),
        dict(
            type="RandomBrightnessContrast",
            brightness_limit=(-0.2, 0.2),  # (-0.4, 0.4)
            contrast_limit=(-0.2, 0.2),  # (-0.4, 0.4)
            brightness_by_max=True,
            p=0.5,
        ),
        dict(
            type="HueSaturationValue",
            hue_range=(-20, 20),
            sat_range=(-30, 30),
            val_range=(-20, 20),
            p=0.5,
        ),
        dict(
            type="RoiTransformCroperAttr",
            model_input_hw=cls_roi_resize_hw[::-1],
            center_crop_prob=0.5,
            rsize_shape=512 if cls_roi_resize_hw[0] == 224 else 256,
            transforms=[
                dict(type="ToTensor", to_yuv=False),
                dict(
                    type="Normalize",
                    mean=[128.0, 128.0, 128.0],
                    std=[128.0, 128.0, 128.0],
                ),
            ],
        ),
    ],
    "val": [
        dict(type="Resize", img_scale=resize_hw_val),
        dict(type="ToTensor", to_yuv=False),
        dict(type="Normalize", mean=mean, std=std),
        dict(type="Pad", size=resize_hw_val, pad_val=0),
    ],
    "val_insseg": [
        dict(type="ReformatLanePolygon"),
        dict(
            type="PolygonToMask",
            replace_gt_seg=True,
            replace_orig_gt_seg=True,
        ),
        dict(type="DeleteKeys", keys=["gt_polygons"]),
        dict(type="Resize", img_scale=resize_hw_val),
        dict(type="ToTensor", to_yuv=False),
        dict(type="Normalize", mean=mean, std=std),
        dict(type="Pad", size=resize_hw_val, pad_val=0, seg_pad_val=0),
    ],
    "val_cls": [
        dict(
            type="RoiTransformCroperAttr",
            model_input_hw=cls_roi_resize_hw[::-1],
            center_crop_prob=1.0,
            rsize_shape=512 if cls_roi_resize_hw[0] == 224 else 256,  # 512,  #
            mode="test",
            use_limit_box=True,
            save_crop_image=False,  # True,
            save_draw_box=True,
            save_crop_image_root="/horizon-bucket/adas/big_model/train_dataset/test_dataset/attribute",
            transforms=[
                dict(type="ToTensor", to_yuv=False),
                dict(
                    type="Normalize",
                    mean=[128.0, 128.0, 128.0],
                    std=[128.0, 128.0, 128.0],
                ),
            ],
        ),
    ],
    "test": [
        dict(type="Resize", img_scale=resize_hw_test),
        dict(type="ToTensor", to_yuv=False),
        dict(type="Normalize", mean=mean, std=std),
        dict(type="Pad", size=resize_hw_test, pad_val=0),
    ],
    "infer": [
        dict(type="Resize", img_scale=resize_hw_test),
        dict(type="ToTensor", to_yuv=False),
        dict(type="Normalize", mean=mean, std=std),
        dict(type="Pad", size=resize_hw_test, pad_val=0),
        dict(type="DeleteKeys", keys=["pad_shape"]),
        dict(type="DeleteKeys", keys=["before_pad_shape"]),
    ],
}

# -------------------------------- Model parts ------------------------------ #
# backbone_cfg = {"name": "hat_efficientnet", "arch": "b5"}
backbone_cfg = {"name": "hat_swin", "arch": "small"}
neck_cfg = {"name": "bifpn", "neck_stack": 3}
stride2channels = {s: 256 for s in [4, 8, 16, 32, 64]}

# import here in case of circular import
from projects.cloudmodel.configs.perception_2d.models import (
    build_backbone_neck,
)

backbone, neck = build_backbone_neck(backbone_cfg, neck_cfg, stride2channels)

# ---------------------------- Training settings ---------------------------- #
optimizer_cfg = {
    "name": "adamw",
    "lr": 0.0002,
    "weight_decay": 1e-2,
    "sync_bn": True,
    # count training by "step" or "epoch"
    "stop_by": "step",
    "num_steps": 80000,
    "num_epochs": None,
}

do_ema = False
do_freeze_bn = False
do_val = True

use_all_data = False
# if you use fast_debug_dataset_number, you need to set use_all_data = False
fast_debug_dataset_number = 1  # see its usage in cloudmodel/data/dataset.py
batch_size_per_gpu = {
    "train_det": 2,
    "train_seg": 3,
    "train_cls": 16,  # 128,
    "val_det": 4,
    "val_seg": 1,
    "val_cls": 16,  # 128,
    "test": 1,
}
num_workers = {"train": 2, "val": 2, "test": 2}

checkpoint_path = backbone.get("checkpoint", None)

resume = False

# ADB light model
# checkpoint_path = "http://svcspawner.bcloud-2nd.hobot.cc/user/homespace/haoyang.zhang/plat_gpu/adb_light_90k_1e-4-20230613_152733/log/big_model_checkpoint-step-62999.pth.tar"
# SD cone model
# checkpoint_path = "http://svcspawner.tcloud.hobot.cc/user/homespace/haoyang.zhang/plat_gpu/cone_5classes_12k_finetune_remove_fps-20230417_153000-COPY3/log/big_model_checkpoint-last.pth.tar"
# checkpoint_path = "http://svcspawner.tcloud.hobot.cc/user/homespace/haoyang.zhang/plat_gpu/cone_5classes_10k_finetune-20230409_111802/log/big_model_checkpoint-step-19999.pth.tar"
# multitask detection
# checkpoint_path = "http://svcspawner.bcloud-2nd.hobot.cc/user/homespace/haoyang.zhang/plat_gpu/multitask-finetue-20k-20230721_150143/log/big_model_checkpoint-last.pth.tar"  # noqa
# lane instance segmentation
# checkpoint_path = "http://svcspawner.tcloud.hobot.cc/user/homespace/naiyu01.gao/plat_gpu/SingleTask-lane-hat_effnetb5-Step400k-B8-SegResizeV2-R960-lr1e-4-2dconvSOLOv2-resume3884891-CudnnOff-20230303_115116/log/big_model_checkpoint-last.pth.tar"  # noqa
# sem40cls parsing
# checkpoint_path = "http://svcspawner.tcloud.hobot.cc/user/homespace/naiyu01.gao/plat_gpu/SingleTask-sem40cls-hatswinb-bifpn-Step200k-B8-R1280_1536-lr1e-5-Upscale2-Feat4816-dwSemFPN-ADRI-FP32-CE1Dice1loss-FreezeBN-20230320_111230/log/big_model_checkpoint-last.pth.tar"  # noqa

# --------------------- Evaluation and inference settings ------------------- #
checkpoint_path_evaluation = checkpoint_path
save_root_evaluation = save_root

data_path_inference = os.path.join(bucket_root, "adas/big_model/tmp/demo")
save_root_inference = save_root
checkpoint_path_inference = checkpoint_path

if pipeline_test:
    optimizer_cfg["num_steps"] = 20
    optimizer_cfg["stop_by"] = "step"
    batch_size_per_gpu = {
        "train_det": 1,
        "train_seg": 1,
        "train_cls": 32,  # 256,
        "val_det": 1,
        "val_seg": 1,
        "val_cls": 32,
        "test": 1,
    }
    num_workers = {"train": 0, "val": 0, "test": 0}
