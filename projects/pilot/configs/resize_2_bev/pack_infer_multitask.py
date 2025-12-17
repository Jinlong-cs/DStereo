import os
from collections import defaultdict
from copy import deepcopy

import torch
from bev_common import (
    camera_view_names,
    organize_data_type,
    per_view_shape,
    raw_image_hw,
    resize_hw,
    spatial_resolution,
    use_distorted_offset,
    vcs_range,
    view_num,
)
from common import model_name, roi_region, vis_tasks_bev
from models import march, val_model  # noqa
from schedule import train_stages

from hat.data.collates.collates import collate_3d

device_ids = [0]

pack_path = "/horizon-bucket/auto_data_4/adas_data_raw/pack/white_changan/H3165_20220507_D/ADAS_20220507-061341_073_$Index.pack"

homo_transforms = {}
for view in camera_view_names:
    view_trans = {
        "Resize": resize_hw,
        "Crop": (roi_region[1], roi_region[0]),
        "ResizeHomo": 1 / 4,
    }
    homo_transforms[view] = view_trans

homo_cfg = dict(
    calib_path=None,
    homo_path=None,
    spatial_resolution=spatial_resolution,
    vcs_range=vcs_range,
    camera_view_names=camera_view_names,
    per_view_shape=per_view_shape,
    norm_homo=True,
    use_distorted_offset=use_distorted_offset,
    homo_transforms=homo_transforms,
)


infer_transforms = [
    dict(
        type="CopyKeys",
        keys=["img|img_ori"],
    ),
    dict(
        type="ANCResize3DV",
        size=resize_hw,
    ),
    dict(
        type="ANCCrop3DV",
        height=[roi_region[3] for _ in range(view_num)],
        width=[roi_region[2] for _ in range(view_num)],
        top=[roi_region[1] for _ in range(view_num)],
        left=[roi_region[0] for _ in range(view_num)],
    ),
    dict(
        type="ANCToTensor3DV",
        to_yuv=True,
        with_color_imgs=False,
        lazy_yuv=True,
        rgb_input=False,
    ),
    dict(
        type="ANCPrepareDataBEV",
        organize_data_type=organize_data_type,
        single_frame=True,
        input_img_sequence=False,
        save_meta_info=True,
    ),
    dict(
        type="GenerateHomo",
        homo_cfg=homo_cfg,
        calib_key="camera_calib",
        camera_view_names=camera_view_names,
        view_shapes=per_view_shape,
    ),
]

callbacks = [
    dict(
        type="BEVMultitaskVisualize",
        out_keys=vis_tasks_bev,
        output_dir=f"./tmp_viz_imgs/{model_name}/",
        bird_eye_size=(
            raw_image_hw[0] // 2,
            raw_image_hw[1] // 2,
        ),
        vcs_range=vcs_range,
        vis_image_layout=dict(
            camera_front=[0, 1],
            camera_front_left=[1, 0],
            bird_eye_view=[1, 1],
            camera_front_right=[1, 2],
            camera_rear_left=[2, 0],
            camera_rear=[2, 1],
            camera_rear_right=[2, 2],
        ),
        vis_configs=dict(
            bev_3d_vehicle=dict(
                colors_map=defaultdict(lambda: (0, 255, 0)),
                score_threshold=0.3,
            ),
            bev_3d_vehicle_cls=dict(
                colors_map=defaultdict(lambda: (0, 255, 0)),
                score_threshold=0.3,
            ),
            bev_3d_pedestrian=dict(
                colors_map=defaultdict(lambda: (0, 255, 0)),
                score_threshold=0.3,
            ),
            bev_3d_cyclist=dict(
                colors_map=defaultdict(lambda: (0, 255, 0)),
                score_threshold=0.3,
            ),
            bev_3d_cyclist_cls=dict(
                colors_map=defaultdict(lambda: (0, 255, 0)),
                score_threshold=0.3,
            ),
        ),
        save_viz_imgs=True,
    ),
]

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="PackDataset",
        pack_path=pack_path,
        pix_format="bgr",
        camera_view_names=camera_view_names,
        camera_calib=True,
        transforms=infer_transforms,
    ),
    collate_fn=collate_3d,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    drop_last=False,
)


base_predictor = dict(
    type="Predictor",
    model=val_model,
    data_loader=data_loader,
    batch_processor=dict(
        type="BasicBatchProcessor",
        need_grad_update=False,
        batch_transforms=[
            dict(
                type="TorchVisionAdapter",
                interface="Normalize",
                mean=128.0,
                std=128.0,
            ),
        ],
    ),
    callbacks=callbacks,
    num_epochs=1,
    share_callbacks=False,
)

for stage in train_stages:
    predictor = deepcopy(base_predictor)
    predictor.update(
        model_convert_pipeline=dict(
            type="FloatQatConvertPipeline",
            qat_mode="fuse_bn",
            checkpoint_mode="resume",
            enable_qat=stage == "qat",
            checkpoint_configs=dict(
                checkpoint_path=os.getenv("HAT_PILOT_MODEL_CHECKPOINT"),
                allow_miss=True,
                ignore_extra=True,
                verbose=1,
            ),
            qconfig_params=None,
        )
    )
    globals()[f"{stage}_predictor"] = predictor
