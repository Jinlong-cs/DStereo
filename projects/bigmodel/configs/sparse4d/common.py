import os

import torch

task_name = "sparse4dv2"

job_suffix = "exp_skip08_75_ignoremask_wokmeans"

os.system(f"rm -rf {os.getenv('HOME')}/.cache/torch/hub/checkpoints")

local_or_remote_debug = False
num_machines = 1 if local_or_remote_debug else 2
num_gpus_per_machine = 2 if local_or_remote_debug else 8

do_experiment = True

enable_temporal_fusion = (
    os.getenv("SPARSE4D_ENABLE_TEMPORAL_FUSION", "1") == "1"
)

is_local_train = not os.path.exists("/running_package")
if is_local_train:
    ckpt_dir = "./tmp_models/%s" % task_name
    train_batch_size = 2
else:
    ckpt_dir = "/job_data/models/%s" % task_name
    train_batch_size = 4
    os.environ["HAT_USE_CHECKPOINT"] = "1"

category_id_dict = {
    1: 0,  # Pedestrian
    2: 1,  # Car        -> Car
    3: 2,  # Cyclist
    4: 1,  # Bus        -> Car
    5: 1,  # Truck      -> Car
    6: 1,  # SpecialCar -> Car
    7: 1,  # Blur       -> Car
    8: -99,  # Other    -> ignore
}  # 'Dontcare' -> Ignore

img_norm_cfg = dict(
    mean=[128.0, 128.0, 128.0], std=[128.0, 128.0, 128.0], to_rgb=False
)

point_cloud_range = [-51.2, -51.2, -3.0, 51.2, 51.2, 5.0]

camera_view_names = [
    "camera_front_left",
    "camera_front_right",
    "camera_rear_left",
    "camera_rear_right",
    "camera_rear",
    "camera_front",
]
sub_dirs = [view.replace("camera_", "") for view in camera_view_names]

num_cameras = len(camera_view_names)


def get_view_shape(model_setting="x3c"):
    if "as33" in model_setting:
        raw_image_hw = (1280, 2048)
    else:
        raw_image_hw = (1280, 1920)
    per_view_shape = {
        "camera_front_left": raw_image_hw,
        "camera_front_right": raw_image_hw,
        "camera_rear_left": raw_image_hw,
        "camera_rear_right": raw_image_hw,
        "camera_rear": raw_image_hw,
        "camera_front": (2160, 3840),
    }
    return per_view_shape


resize_target_dim = (640, 960)
pad_output_dim = (640, 960)


def get_train_sparse_transforms(model_setting="x3c"):

    transforms_ = [
        dict(
            type="GetCalibParams",
            camera_view_names=sub_dirs,
            view_shapes=get_view_shape(model_setting),
        ),
        dict(
            type="MultiViewDrawIgnoreMask",
            pad_value=128,
        ),
        dict(
            type="MultiViewRecPadView",
            camera_view_names=sub_dirs,
            view_shapes=get_view_shape(model_setting),
        ),
        dict(
            type="MultiViewRecTransform",
            category_id_dict=category_id_dict,
            camera_view_names=sub_dirs,
        ),
        dict(
            type="MultiViewFlipResizeCrop",
            resize_target_dim=resize_target_dim,
            horizontal_flip_ratio=-1.0,
            keep_ratio=False,
        ),
        # dict(
        #     type="BBoxRotation",
        #     transform_matrix_key=("T_vcs2img",),
        #     global_key=("T_vcs2global",),
        #     rotation_3d_range=(-22.5, 22.5),
        # ),
        dict(
            type="MultiViewPhotoMetricDistortion",
        ),
        dict(
            type="MultiViewNormalize",
            img_norm_cfg=img_norm_cfg,
        ),
        dict(
            type="MultiViewPadImage",
            size=pad_output_dim,
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
        ),
        dict(
            type="MVT4DImgFormat",
        ),
        dict(
            type="MultiViewRangeFliter",
            point_cloud_range=point_cloud_range,
        ),
        dict(
            type="Sparse4DAdaptor",
            projection_key="T_vcs2img",
            img_shape_key="img_shape",
            ego_pose_key="T_vcs2global",
            cam_intrinsic_key="camera_matrix",
        ),
        dict(
            type="ConvertDataType",
            convert_map=dict(
                gt_bboxes_3d=torch.float32,
                gt_labels_3d=torch.int64,
            ),
        ),
        dict(
            type="MultiViewCollect3D",
            keep_keys=[
                "img",
                "timestamp",
                "projection_mat",
                "image_wh",
                "gt_labels_3d",
                "gt_bboxes_3d",
                "img_metas",
            ],
            img_metas_keys=["timestamp", "T_global", "T_global_inv"],
        ),
    ]

    return transforms_


def get_val_sparse_transforms(model_setting="x3c"):

    transforms_ = [
        dict(
            type="MultiViewRecPadView",
            camera_view_names=sub_dirs,
            view_shapes=get_view_shape(model_setting),
        ),
        dict(
            type="GetCalibParams",
            camera_view_names=sub_dirs,
            view_shapes=get_view_shape(model_setting),
        ),
        dict(
            type="MultiViewRecTransform",
            category_id_dict=category_id_dict,
            camera_view_names=sub_dirs,
        ),
        dict(
            type="MultiViewFlipResizeCrop",
            resize_target_dim=resize_target_dim,
            horizontal_flip_ratio=-1.0,
            keep_ratio=False,
        ),
        dict(
            type="MultiViewNormalize",
            img_norm_cfg=img_norm_cfg,
        ),
        dict(
            type="MultiViewPadImage",
            size=pad_output_dim,
        ),
        dict(
            type="MVT4DImgFormat",
        ),
        dict(
            type="MultiViewRangeFliter",
            point_cloud_range=point_cloud_range,
        ),
        dict(
            type="Sparse4DAdaptor",
            projection_key="T_vcs2img",
            img_shape_key="img_shape",
            ego_pose_key="T_vcs2global",
            cam_intrinsic_key="camera_matrix",
        ),
        dict(
            type="ConvertDataType",
            convert_map=dict(
                gt_bboxes_3d=torch.float32,
                gt_labels_3d=torch.int64,
            ),
        ),
        dict(
            type="MultiViewCollect3D",
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
            ],
        ),
    ]

    return transforms_


def get_dataset_list(rec_path_list, model_setting="x3c", mode="train"):
    if mode == "train":
        get_transforms = get_train_sparse_transforms
    else:
        get_transforms = get_val_sparse_transforms

    per_view_shape = get_view_shape(model_setting)

    dataset_list = [
        dict(
            type="MultiViewRecDataset",
            rec_path=rec_path,
            rec_idx_file=rec_path + ".idx",
            camera_view_names=sub_dirs,
            view_shapes=per_view_shape,
            to_rgb=True,
            decode_img=True,
            transforms=get_transforms(model_setting),
            homo_cfg=None,
        )
        for rec_path in rec_path_list
    ]
    return dataset_list


def get_lmdb_dataset_list(lmdb_path_list, model_setting="x3c", mode="train"):
    if mode == "train":
        get_transforms = get_train_sparse_transforms
    else:
        get_transforms = get_val_sparse_transforms

    dataset_list = [
        dict(
            type="TemporalLmdbDataset",
            idx_path=os.path.join(lmdb_path, "idx"),
            anno_path=os.path.join(lmdb_path, "anno"),
            transforms=[
                dict(
                    type="MVImgLmdbReader",
                    img_path=os.path.join(lmdb_path, "img"),
                    to_rgb=True,
                )
            ]
            + get_transforms(model_setting),
            max_interval=600,
            max_len_in_clip=75,
        )
        for lmdb_path in lmdb_path_list
    ]
    return dataset_list


def get_json_dataset_list(json_path_list, model_setting="x3c", mode="train"):
    if mode == "train":
        get_transforms = get_train_sparse_transforms
    else:
        get_transforms = get_val_sparse_transforms

    dataset_list = [
        dict(
            type="TemporalJsonDataset",
            json_file=json_path["json_file"],
            transforms=[
                dict(
                    type="MVImgJsonReader",
                    img_dir=json_path["img_dir"],
                    to_rgb=True,
                )
            ]
            + get_transforms(model_setting),
            max_interval=600,
            max_len_in_clip=-1,
        )
        for json_path in json_path_list
    ]
    return dataset_list


# -------------------------------- Submit configuration ----------------------------------  # noqa
job_name = "hat_job"
job_name = job_name + "_" + task_name + "_" + job_suffix
job_password = "6150"
framework = "pytorch"
task_label = "HAT_MVT4D"
project_id = "PDT20220001"
# project_id = "LMT20230002"
input_bucket = "matrix,matrix2,auto_eval,adas"
priority = 5
docker_image = "docker.hobot.cc/imagesys/hat:pilot-runtime-cu111-hdflow-20230611-torch1102-aidisdk0111-py38-mmcv142"  # noqa
max_jobtime = 60 if local_or_remote_debug else 10000  # 60 for debug
launcher = "mpi"
upload_folder_name = "k8s_job"
folder_list = [
    "../../hat",
    "../../tools",
    "../../projects/bigmodel",
    "../../projects/pilot",
    # "/horizon-bucket/matrix/users/xuewu.lin/nuscenes_kmeans900.npy",
    "url2IP.py",
]
config_file = "bigmodel/configs/sparse4d/entry.py"
eval_config_file = "bigmodel/configs/sparse4d/eval_entry.py"
job_list = [
    f"python3 -W ignore tools/train.py --config {config_file} --stage float --device-ids "
    + ",".join(list(map(str, range(num_gpus_per_machine)))),
    # f"python3 -W ignore tools/predict.py --config {eval_config_file} --stage float --device-ids "
    # + ",".join(list(map(str, range(num_gpus_per_machine)))),
]
