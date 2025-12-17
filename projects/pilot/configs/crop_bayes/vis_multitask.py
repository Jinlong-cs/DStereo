import os
from copy import deepcopy

import torch
from common import model_checkpoint, model_setting, vis_tasks
from models import val_model
from schedule import get_fuse_patterns_by_stage, train_stages

from hat.core.proj_spec.parsing import colormap
from hat.data.collates.collates import default_collate_v2

root_path = "/"
# clipping: begin
root_path = os.path.join(
    (
        "/horizon-bucket"
        if not os.path.exists("/running_package")
        else "/bucket/input"
    ),
    "matrix2/users/jiaxi.wu/whitebox",
)
# clipping: end

device_ids = [0]
march = "bernoulli2"
qat_mode = "fuse_bn"

val_transforms = [
    dict(type="YUVTurboJPEGDecoder", to_string=True),
    dict(
        type="CropImgPatch",
        static_roi=(0, 0, 1920, 1280),
    ),
    dict(type="ImgBufToYUV444"),
    dict(type="ToTensor", to_yuv=False),
    dict(type="Normalize", mean=128.0, std=128.0),
]

if "rear" in model_setting.lower():
    roi_region = (704, 568, 1216, 760)
elif "side_right" in model_setting.lower():
    roi_region = (1536, 431, 1920, 623)  # right rear
elif "side_left" in model_setting.lower():
    roi_region = (0, 431, 384, 623)  # left rear
val_transforms[1].update(static_roi=roi_region)

colormap = {i: color for i, color in enumerate(colormap)}
colormap[255] = [0, 0, 0]

visualize_callback = dict(
    type="ComposeVisualize",
    callbacks=[
        dict(
            type="DetMultitaskVisualize",
            out_keys=vis_tasks,
            vis_configs=dict(
                vehicle=dict(
                    color=(0, 255, 0),
                    thickness=2,
                    points2=dict(),
                ),
                rear=dict(
                    color=(0, 255, 255),
                    thickness=2,
                ),
                person=dict(
                    color=(255, 0, 0),
                    thickness=2,
                ),
                cyclist=dict(
                    color=(255, 255, 0),
                    thickness=2,
                ),
                default_segmentation=dict(
                    colormap=colormap,
                    alpha=0.7,
                ),
                lane_segmentation=dict(
                    colormap=colormap,
                    alpha=0.7,
                ),
            ),
            save_viz_imgs=True,
        ),
    ],
)

callbacks = [
    visualize_callback,
]

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="FrameDataset",
        img_path=os.path.join(
            root_path, "pilot_data_raw/minitest/test_img.jpg"
        ),
        calib_path=os.path.join(
            root_path, "pilot_data_raw/minitest/test_calib.json"
        ),
        to_rgb=True,
        buf_only=True,
        transforms=val_transforms,
    ),
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    drop_last=False,
    collate_fn=default_collate_v2,
)

base_predictor = dict(
    type="Predictor",
    model=val_model,
    data_loader=data_loader,
    batch_processor=dict(
        type="BasicBatchProcessor",
        need_grad_update=False,
        inverse_transforms=dict(
            type="FrcnnInvTransforms",
            img_transforms=val_transforms,
        ),
    ),
    callbacks=callbacks,
    num_epochs=1,
    share_callbacks=False,
)


for stage in train_stages:
    pre, cur = get_fuse_patterns_by_stage(stage)

    predictor = deepcopy(base_predictor)
    predictor.update(
        model_convert_pipeline=dict(
            type="QATFuseBNConvertPipeline",
            qat_mode="with_bn_reverse_fold",
            pre_stage_fuse_patterns=pre,
            cur_stage_fuse_patterns=cur,
            fuse_part_configs=dict(
                fuse_method="fuse_norm",
                regex=True,
                strict=False,
            ),
            checkpoint_mode="resume",
            checkpoint_configs=dict(
                checkpoint_path=model_checkpoint,
                allow_miss=False,
                ignore_extra=True,
                verbose=1,
            ),
            qconfig_params=None,
        )
    )
    globals()[f"{stage}_predictor"] = predictor
