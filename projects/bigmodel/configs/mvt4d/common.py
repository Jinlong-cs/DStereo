import os

task_name = "mvt4dv2_r50_badcases"

os.system(f"rm -rf {os.getenv('HOME')}/.cache/torch/hub/checkpoints")

local_or_remote_debug = False
num_machines = 1 if local_or_remote_debug else 2
num_gpus_per_machine = 2 if local_or_remote_debug else 8

is_local_train = not os.path.exists("/running_package")
if is_local_train:
    ckpt_dir = "./tmp_models/%s" % task_name
    train_batch_size = 2
else:
    ckpt_dir = "/job_data/models/%s" % task_name
    train_batch_size = 2
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

enable_temporal_fusion = True

camera_view_names = [
    "camera_front_left",
    "camera_front_right",
    "camera_rear_left",
    "camera_rear_right",
    "camera_rear",
    "camera_front",
]
sub_dirs = [view.replace("camera_", "") for view in camera_view_names]


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

num_cameras = len(camera_view_names)

backbone = dict(
    type="ResNet50V2",
    num_classes=1000,
    group_base=8,
    include_top=False,
    extend_features=False,
    bn_kwargs=dict(eps=1e-5, momentum=0.1),
)

img_neck = dict(
    type="FPN",
    in_strides=[2, 4, 8, 16, 32],
    in_channels=[64, 256, 512, 1024, 2048],
    out_strides=[8, 16, 32],
    out_channels=[256, 256, 256],
    bn_kwargs=dict(eps=1e-5, momentum=0.1),
)


bev_h_value = 256
bev_w_value = 256
bev_feat_encoder = dict(
    type="BevFeatEncoder",
    bev_h=bev_h_value,
    bev_w=bev_w_value,
    num_cams=num_cameras,
    bev_num_refs=4,
    num_layers=3,
    pc_range=point_cloud_range,
    max_interval=600.0 / 1e3,
    temporal_layer=dict(
        type="BaseTransformerLayer",
        attn_cfgs=[
            dict(
                type="BevDeformableTemporalAttention",
                dropout=0.1,
                num_levels=1,
                embed_dims=256,
                bev_h=bev_h_value,
                bev_w=bev_w_value,
                qv_cat=False,
            )
        ],
        ffn_cfgs=dict(
            type="FFN",
            embed_dims=256,
            feedforward_channels=512,
            num_fcs=2,
            ffn_drop=0.1,
        ),
        operation_order=(
            "self_attn",
            "norm",
            "ffn",
            "norm",
        ),
    ),
    transformerlayers=dict(
        type="BaseTransformerLayer",
        attn_cfgs=[
            dict(
                type="MultiScaleDeformableAttention",
                num_levels=1,
                embed_dims=256,
            ),
            dict(
                type="BevSpatialCrossAtten",
                pc_range=point_cloud_range,
                num_cams=num_cameras,
                deformable_attention=dict(
                    type="MSDeformableAttention3D",
                    embed_dims=256,
                    num_points=8,
                    num_levels=3,
                ),
                embed_dims=256,
            ),
        ],
        ffn_cfgs=dict(
            type="FFN",
            embed_dims=256,
            feedforward_channels=512,
            num_fcs=2,
            ffn_drop=0.1,
        ),
        operation_order=(
            "self_attn",
            "norm",
            "cross_attn",
            "norm",
            "ffn",
            "norm",
        ),
    ),
    positional_encoding=dict(
        type="LearnedPositionalEncoding",
        num_feats=128,
        row_num_embed=bev_h_value,
        col_num_embed=bev_w_value,
    ),
)


def get_train_transforms(model_setting="x3c"):

    transforms_ = [
        dict(
            type="GetCalibParams",
            camera_view_names=sub_dirs,
            view_shapes=get_view_shape(model_setting),
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
            type="MultiViewCollect3D",
            keep_keys=("img", "gt_labels_3d", "gt_bboxes_3d", "img_metas"),
            img_metas_keys=(
                "img_shape",
                "T_vcs2img",
                "T_vcs2global",
                "T_global2vcs",
                "timestamp",
            ),
        ),
    ]

    return transforms_


def get_val_transforms(model_setting="x3c"):

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
            type="MultiViewCollect3D",
            keep_keys=("img", "gt_labels_3d", "gt_bboxes_3d", "img_metas"),
            img_metas_keys=(
                "img_shape",
                "T_vcs2img",
                "timestamp",
                "T_vcs2global",
                "T_global2vcs",
                "ori_camera_matrix",
                "ori_distcoeffs",
                "ori_vcs2cam",
                "ori_image_size",
            ),
        ),
    ]

    return transforms_


def get_dataset_list(rec_path_list, model_setting="x3c", mode="train"):
    if mode == "train":
        get_transforms = get_train_transforms
    else:
        get_transforms = get_val_transforms

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
        get_transforms = get_train_transforms
    else:
        get_transforms = get_val_transforms

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
            max_len_in_clip=20,
        )
        for lmdb_path in lmdb_path_list
    ]
    return dataset_list


def get_json_dataset_list(json_path_list, model_setting="x3c", mode="train"):
    if mode == "train":
        get_transforms = get_train_transforms
    else:
        get_transforms = get_val_transforms

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
job_name = job_name + "_" + task_name
job_password = "6150"
framework = "pytorch"
task_label = "HAT_MVT4D"
project_id = "PDT20220001"
input_bucket = "matrix,matrix2"
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
    "url2IP.py",
]
config_file = "bigmodel/configs/mvt4d/entry.py"
eval_config_file = "bigmodel/configs/mvt4d/eval_entry.py"
job_list = [
    # train on aidi
    f"python3 -W ignore tools/train.py --config {config_file} --stage float --device-ids "
    + ",".join(list(map(str, range(num_gpus_per_machine)))),
    # eval on aidi
    # f"python3 -W ignore tools/predict.py --config {eval_config_file} --stage float --device-ids "
    # + ",".join(list(map(str, range(num_gpus_per_machine)))),
]
