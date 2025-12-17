import os

import torch
from common import (
    backbone,
    batch_size_per_gpu_face3d,
    in_channels,
    input_hw,
    is_local_train,
    loss_weights,
    num_workers,
    step_log_freq,
)

try:
    import lpips
except ImportError:
    raise ImportError("Please install lpips")


task_name = "face3d"
bucket_root = "/horizon-bucket" if is_local_train else "/bucket/input"
# bucket_root = "/horizon-bucket" if is_local_train else "/bucket/output"
root = os.path.join(bucket_root, "interaction")
deploy_inputs = dict(img=torch.randn((1, 3, input_hw[0], input_hw[1])))

# inputs
inputs = dict(
    pretrain=dict(gt_ldmk=torch.zeros(2, 68, 3)),
    train=dict(
        gt_ldmk=torch.zeros(2, 68, 3),
        gt_img=torch.zeros(2, 3, 256, 256),
        gt_mask=torch.zeros(2, 1, 256, 256),
    ),
    val=dict(),
    test=dict(),
    deploy=dict(),
)

# train data
train_lmdb_image_list = [
    f"{root}/active/face3d/lmdb/zuocang_lmdb/image_lmdb",
]
train_lmdb_mask_list = [
    f"{root}/active/face3d/lmdb/zuocang_lmdb/mask_lmdb",
]
train_ldmk_anno_list = [
    f"{root}/active/face3d/lmdb/zuocang_lmdb/anno_lmdb",
]

# test data
val_lmdb_image_list = [
    [f"{root}/active/face3d/lmdb/gaze_A29_test_lmdb/image_lmdb"],
    [f"{root}/active/face3d/lmdb/gaze_T18_test_lmdb/image_lmdb"],
    [f"{root}/active/face3d/lmdb/gaze_CD569_test_lmdb/image_lmdb"],
]

val_lmdb_mask_list = [
    [f"{root}/active/face3d/lmdb/gaze_A29_test_lmdb/mask_lmdb"],
    [f"{root}/active/face3d/lmdb/gaze_T18_test_lmdb/mask_lmdb"],
    [f"{root}/active/face3d/lmdb/gaze_CD569_test_lmdb/mask_lmdb"],
]

val_lmdb_anno_list = [
    [f"{root}/active/face3d/lmdb/gaze_A29_test_lmdb/anno_lmdb"],
    [f"{root}/active/face3d/lmdb/gaze_T18_test_lmdb/anno_lmdb"],
    [f"{root}/active/face3d/lmdb/gaze_CD569_test_lmdb/anno_lmdb"],
]


# ----------------- model ---------------------------
def get_model(mode):
    return dict(
        type="FaceMtlFace3dModel",
        backbone=backbone,
        head=dict(
            type="FaceMtlFace3dHead",
            # kernel_size=4,
            in_channels=in_channels,
            out_channels=128 if in_channels == 128 else 120,
            node_name=f"{task_name}_head",
        ),
        post_module=dict(
            type="FaceMtlFace3DPostModule",
            flame=dict(
                type="FLAME",
                flame_model_path="face3d_data/new_generic_model.pkl",
                flame_lmk_embedding_path="face3d_data/new_landmark_embedding.npy",
            ),
            flame_tex=dict(
                type="FLAMETex",
                tex_path="face3d_data/FLAME_albedo_from_BFM.npz",
            ),
            renderer=dict(
                type="NVRenderer",
                obj_filename="face3d_data/head_template_mesh.obj",
                render_size=(256, 256),
            ),
            lpips=lpips.LPIPS(
                pretrained=True,
                pnet_rand=True,
                model_path="face3d_data/lpips.pth",
                verbose=True,
                net="vgg",
            ),
            face3d_loss_weight=loss_weights[task_name],
            node_name=f"{task_name}_post_module",
        )
        if mode == "train"
        else None,
        loss_weights={
            "ldmk": 256.0,
            "photo": 10.0,
            "lpips": 30.0,
            "shape_reg": 1e-3,
            "exp_reg": 1e-3,
            "tex_reg": 1e-3,
        },
        deploy=True if mode != "train" else False,
    )


# ----------------- data ---------------------------
train_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Face3dDataset",
        image_path_list=train_lmdb_image_list,
        mask_path_list=train_lmdb_mask_list,
        anno_path_list=train_ldmk_anno_list,
        transforms=[
            dict(type="RandomFlip", px=0.0, py=0.0),
            dict(
                type="RandomRotateCrop",
                rot_prob=1.0,
                rot_angle_range=30.0,
                center_shift_prob=1.0,
                center_shift_range=0.025,
                norm_ratio=1.2,
                norm_method="longside_square",
                norm_jitter_range=0.1,
                net_input_size=(input_hw[0], input_hw[1]),
                net_target_size=(256, 256),
                base_len=200,
            ),
            dict(type="ToTensor"),
            dict(
                type="FaceMtlTransformLabel",
                name=task_name,
            ),
        ],
        stage="finetune",
    ),
    persistent_workers=True,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu_face3d,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)


test_data_loader_list = []
for lmdb_image_list, lmdb_mask_list, lmdb_anno_list in zip(
    val_lmdb_image_list, val_lmdb_mask_list, val_lmdb_anno_list
):
    test_data_loader = dict(
        type=torch.utils.data.DataLoader,
        dataset=dict(
            type="Face3dDataset",
            image_path_list=lmdb_image_list,
            mask_path_list=lmdb_mask_list,
            anno_path_list=lmdb_anno_list,
            transforms=[
                dict(
                    type="RandomRotateCrop",
                    rot_prob=0.0,
                    rot_angle_range=0.0,
                    center_shift_prob=0.0,
                    center_shift_range=0.0,
                    norm_ratio=1.2,
                    norm_method="longside_square",
                    norm_jitter_range=0.0,
                    net_input_size=(input_hw[0], input_hw[1]),
                    net_target_size=(256, 256),
                    base_len=200,
                ),
                dict(type="ToTensor"),
            ],
        ),
        batch_size=batch_size_per_gpu_face3d,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    test_data_loader_list.append(test_data_loader)


# ----------------- metric ---------------------------
train_metrics = [
    dict(type="LossShow", name="Lmk"),
    dict(type="LossShow", name="Photo"),
    dict(type="LossShow", name="Lpips"),
    dict(type="LossShow", name="Shape"),
    dict(type="LossShow", name="Exp"),
    dict(type="LossShow", name="Tex"),
    dict(type="LossShow", name="Loss"),
]

pretrain_metrics = [
    dict(type="LossShow", name="Lmk"),
    dict(type="LossShow", name="Shape"),
    dict(type="LossShow", name="Exp"),
    dict(type="LossShow", name="Tex"),
    dict(type="LossShow", name="Loss"),
]

test_metrics = [
    dict(type="PoseMAE", name=["Roll", "Pitch", "Yaw", "Mae"]),
]


# ----------------- metric ---------------------------
def update_metric(metrics, batch, model_outs):
    if task_name in model_outs:
        model_outs = model_outs[task_name]
        for metric, key in zip(metrics, model_outs):
            metric.update(model_outs[key])


def test_update_metric(metrics, batch, model_outs):
    model_outs = model_outs[task_name]
    # democar visualization
    # metrics[0].update(batch[0]["gt_pose"], model_outs["pred_pose"])

    for metric, key in zip(metrics, model_outs):
        metric.update(batch[0], model_outs[key])


metric_updater = dict(
    type="MetricUpdater",
    metrics=train_metrics,
    metric_update_func=update_metric,
    step_log_freq=step_log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
)

pretrain_metric_updater = dict(
    type="MetricUpdater",
    metrics=pretrain_metrics,
    metric_update_func=update_metric,
    step_log_freq=step_log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
)


test_metric_updater_list = []
for lmdb_image_list in val_lmdb_image_list:
    test_metric_updater = dict(
        type="MetricUpdater",
        metrics=test_metrics,
        metric_update_func=test_update_metric,
        step_log_freq=0,
        epoch_log_freq=1,
        log_prefix="Validation " + task_name + " " + lmdb_image_list[0],
    )
    test_metric_updater_list.append(test_metric_updater)
