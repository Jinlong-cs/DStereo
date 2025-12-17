import os

import torch

from hat.utils import Config
from projects.cloudmodel.configs.cloudsparse4d.data.builders import (
    get_view_shape,
)

cfg = Config.fromfile(os.path.join(os.path.dirname(__file__), "config.py"))


def get_train_sparse_transforms(
    model_setting="x3c",
    data_type="unknown",
):
    transforms = [
        dict(
            type="GetCalibParams",
            camera_view_names=cfg.sub_dirs,
            view_shapes=get_view_shape(model_setting),
            return_extra=True,
            homo_noise=dict(
                noise_range=[1.0, 1.0, 1.0, 0.00, 0.00, 0.04],
                noise_type="random",
                noise_view="random",
                noise_prob=0.2,
            ),
        ),
        dict(
            type="MultiViewRecPadView",
            camera_view_names=cfg.sub_dirs,
            view_shapes=get_view_shape(model_setting),
            drop_view=[
                "front_left",
                "front_right",
                "rear_left",
                "rear_right",
                "rear",
                "front_30fov",
            ],
            drop_view_ratio=0.1 if data_type == "pilot" else 0.0,
        ),
        dict(
            type="MultiViewRecTransform",
            category_id_dict=cfg.category_id_dict,
            camera_view_names=cfg.sub_dirs,
        ),
        dict(
            type="VirtualCropCamera",
            virtual_view=cfg.virtual_crop_views,
            crop_roi=cfg.virtual_crop_rois,
            camera_view_names=cfg.sub_dirs,
            prob=0.5,
        ),
        dict(
            type="SplitMultiView",
            category_view=cfg.view2category,
            camera_view_names=cfg.sub_dirs
            + [f"virtual_{_}" for _ in cfg.virtual_crop_views if _ != "front"],
            split_key=[
                "imgs",
                "ori_camera_matrix",
                "ori_distcoeffs",
                "ori_vcs2cam",
                "ori_image_size",
                "T_vcs2cam",
                "camera_matrix",
            ],
        ),
        dict(
            type="SplitMultiViewFlipResizeCrop",
            resize_target_dim=cfg.view2target_dim,
            horizontal_flip_ratio=-1.0,
            keep_ratio=False,
            category_view=cfg.view2category,
        ),
        dict(
            type="SplitMultiViewPhotoMetricDistortion",
            category_view=cfg.view2category,
        ),
        dict(
            type="MultiViewNormalize",
            img_norm_cfg=cfg.img_norm_cfg,
            category_view=cfg.view2category,
        ),
        dict(
            type="SplitMultiViewPadImage",
            size=cfg.view2pad_dim,
            category_view=cfg.view2category,
        ),
        dict(
            type="MultiViewGridMask",
            use_h=True,
            use_w=True,
            rotate=1.0,
            offset=False,
            ratio=0.5,
            mode=1,
            prob=0.7,
            category_view=cfg.view2category,
        ),
        dict(
            type="MVT4DImgFormat",
            category_view=cfg.view2category,
        ),
        dict(
            type="MultiViewRangeFliter",
            point_cloud_range=cfg.point_cloud_range,
        ),
        dict(
            type="Sparse4DAdaptor",
            projection_key="T_vcs2img",
            img_shape_key="img_shape",
            ego_pose_key="T_vcs2global",
            cam_intrinsic_key="camera_matrix",
            category_view=cfg.view2category,
        ),
        dict(
            type="ConvertDataType",
            convert_map=dict(
                gt_bboxes_3d=torch.float32,
                gt_labels_3d=torch.int64,
            ),
        ),
        dict(
            type="SplitMultiViewCollect3D",
            keep_keys=[
                "img",
                "timestamp",
                "projection_mat",
                "image_wh",
                "gt_labels_3d",
                "gt_bboxes_3d",
                "img_metas",
            ],
            img_metas_keys=[
                "timestamp",
                "T_global",
                "T_global_inv",
                "ori_camera_matrix",
                "ori_distcoeffs",
                "ori_vcs2cam",
                "ori_image_size",
                "img_shape",
                "T_vcs2img",
                "T_vcs2global",
                "T_global2vcs",
                "T_local2vcs",
                "T_local2cam",
            ],
            category_view=cfg.view2category,
        ),
    ]
    return transforms


def get_val_sparse_transforms(model_setting="x3c"):
    transforms = [
        dict(
            type="MultiViewRecPadView",
            camera_view_names=cfg.sub_dirs,
            view_shapes=get_view_shape(model_setting),
        ),
        dict(
            type="GetCalibParams",
            camera_view_names=cfg.sub_dirs,
            view_shapes=get_view_shape(model_setting),
            return_extra=True,
        ),
        dict(
            type="MultiViewRecTransform",
            category_id_dict=cfg.category_id_dict,
            camera_view_names=cfg.sub_dirs,
        ),
        dict(
            type="VirtualCropCamera",
            virtual_view=cfg.virtual_crop_views,
            crop_roi=cfg.virtual_crop_rois,
            camera_view_names=cfg.sub_dirs,
        ),
        dict(
            type="SplitMultiView",
            category_view=cfg.view2category,
            camera_view_names=cfg.sub_dirs
            + [f"virtual_{_}" for _ in cfg.virtual_crop_views if _ != "front"],
            split_key=[
                "imgs",
                "ori_camera_matrix",
                "ori_distcoeffs",
                "ori_vcs2cam",
                "ori_image_size",
                "T_vcs2cam",
                "camera_matrix",
            ],
        ),
        dict(
            type="SplitMultiViewFlipResizeCrop",
            resize_target_dim=cfg.view2target_dim,
            horizontal_flip_ratio=-1.0,
            keep_ratio=False,
            category_view=cfg.view2category,
        ),
        dict(
            type="MultiViewNormalize",
            img_norm_cfg=cfg.img_norm_cfg,
            category_view=cfg.view2category,
        ),
        dict(
            type="SplitMultiViewPadImage",
            size=cfg.view2pad_dim,
            category_view=cfg.view2category,
        ),
        dict(
            type="MVT4DImgFormat",
            category_view=cfg.view2category,
        ),
        dict(
            type="MultiViewRangeFliter",
            point_cloud_range=cfg.point_cloud_range,
        ),
        dict(
            type="Sparse4DAdaptor",
            projection_key="T_vcs2img",
            img_shape_key="img_shape",
            ego_pose_key="T_vcs2global",
            cam_intrinsic_key="camera_matrix",
            category_view=cfg.view2category,
        ),
        dict(
            type="ConvertDataType",
            convert_map=dict(
                gt_bboxes_3d=torch.float32,
                gt_labels_3d=torch.int64,
            ),
        ),
        dict(
            type="SplitMultiViewCollect3D",
            keep_keys=[
                "img",
                "timestamp",
                "projection_mat",
                "image_wh",
                "gt_labels_3d",
                "gt_bboxes_3d",
                "img_metas",
                "image_keys",
            ],
            img_metas_keys=[
                "timestamp",
                "T_global",
                "T_global_inv",
                "ori_camera_matrix",
                "ori_distcoeffs",
                "ori_vcs2cam",
                "ori_image_size",
                "img_shape",
                "T_vcs2img",
                "T_vcs2global",
                "T_global2vcs",
                "T_local2vcs",
                "T_local2cam",
            ],
            category_view=cfg.view2category,
        ),
    ]
    return transforms


# train datasets
train_datasets = [
    # pilot
    dict(
        type="multiview",
        file="pilot_galaxy_x3c_multiview_datasets.py",
        transforms=get_train_sparse_transforms("x3c", "pilot"),
        model_setting="x3c",
    ),
    # pilot temporal
    dict(
        type="temporal",
        file="pilot_x3c_multiview_lmdb_datasets.py",
        transforms=get_train_sparse_transforms("x3c", "pilot"),
        model_setting="x3c",
    ),
    # sd
    # TODO: add cloudbev datasets
]

# eval datasets
eval_datasets = [os.getenv("eval_dataset_id", "6042112")]
# eval_datasets = "all" # "sd", "person", "sd_person"
