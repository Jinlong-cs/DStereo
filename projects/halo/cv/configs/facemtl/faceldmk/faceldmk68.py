import copy
import os

import torch
from common import (
    backbone,
    backbone_type,
    batch_size_per_gpu_ldmk,
    input_hw,
    is_local_train,
    ldmk_in_channel,
    loss_weights,
    num_workers,
    step_log_freq,
)

bucket_root = "/horizon-bucket" if is_local_train else "/bucket/input"
# bucket_root = "/horizon-bucket" if is_local_train else "/bucket/output"
root = os.path.join(bucket_root, "interaction")

NUM_LDMK = 68
LDMK_PAIRS = [
    [0, 16],
    [1, 15],
    [2, 14],
    [3, 13],
    [4, 12],
    [5, 11],
    [6, 10],
    [7, 9],
    [17, 26],
    [18, 25],
    [19, 24],
    [20, 23],
    [21, 22],
    [31, 35],
    [32, 34],
    [36, 45],
    [37, 44],
    [38, 43],
    [39, 42],
    [40, 47],
    [41, 46],
    [48, 54],
    [49, 53],
    [50, 52],
    [61, 63],
    [60, 64],
    [67, 65],
    [58, 56],
    [59, 55],
]
bn_kwargs = dict(
    eps=2e-5, momentum=0.1
)  # because the different implement from pytorch and mxnet, momentum in gluon face is 0.9. # noqa

task_name = "face_ldmk"
STRIDE = 4
INPUT_H = input_hw[0]
INPUT_W = input_hw[1]
TARGET_H = INPUT_H // STRIDE
TARGET_W = INPUT_W // STRIDE
test_inputs = dict(img=torch.randn((1, 3, INPUT_H, INPUT_W)))
deploy_inputs = dict(img=torch.randn((1, 3, INPUT_H, INPUT_W)))
# inputs
inputs = dict(
    train=dict(
        gt_vector_x=torch.zeros(2, 68, TARGET_H),
        gt_vector_y=torch.zeros(2, 68, TARGET_H),
        gt_vector_weight_x=torch.zeros(2, 68, TARGET_H),
        gt_vector_weight_y=torch.zeros(2, 68, TARGET_H),
        loss_weight=torch.zeros(1),
    ),
    val=dict(
        gt_vector_x=torch.zeros(2, 68, TARGET_H),
        gt_vector_y=torch.zeros(2, 68, TARGET_H),
        gt_vector_weight_x=torch.zeros(2, 68, TARGET_H),
        gt_vector_weight_y=torch.zeros(2, 68, TARGET_H),
        loss_weight=torch.zeros(1),
        gt_ldmk=torch.zeros(2, 68, 2),
    ),
    test=dict(
        gt_vector_x=torch.zeros(2, 68, TARGET_H),
        gt_vector_y=torch.zeros(2, 68, TARGET_H),
        gt_vector_weight_x=torch.zeros(2, 68, TARGET_H),
        gt_vector_weight_y=torch.zeros(2, 68, TARGET_H),
        loss_weight=torch.zeros(1),
        gt_ldmk=torch.zeros(2, 68, 2),
    ),
    deploy=dict(),
)


# ----------------- model ---------------------------
if backbone_type == "mixvargenet0.75":
    # mixvargenet0.75 decoder
    decoder = dict(
        type="GroupFPN",
        feature_dim=120,
        in_channels=[24, 48, 64, 128],  # strides:[4, 8, 16, 32]
        min_output_stride=4,
        group_base=16,
        bn_kwargs=bn_kwargs,
        node_name="face_ldmk_decoder",
    )
elif backbone_type == "mixvargenet1.0":
    # mixvargenet1.0 decoder
    decoder = dict(
        type="GroupFPN",
        feature_dim=120,
        in_channels=[32, 64, 96, 160],  # strides:[4, 8, 16, 32]
        min_output_stride=4,
        group_base=16,
        bn_kwargs=bn_kwargs,
        node_name="face_ldmk_decoder",
    )


def get_model(mode):
    if mode == "test":
        mode = "val"
    return dict(
        type="LdmkModel",
        backbone=backbone,
        mode=mode,
        flatten_output=False,
        decoder=decoder,
        vector_head=dict(
            type="LdmkVectorHead",
            in_channels=ldmk_in_channel,
            num_ldmk=NUM_LDMK,
            band_width=1,
            vector_size=(TARGET_H, TARGET_W),
            band_module_type="conv",
            is_train=True if mode == "train" else False,
            loss_func=dict(type="LdmkLoss", loss_type="l2"),
            bn_kwargs=bn_kwargs,
            node_name="face_ldmk_vector_head",
        ),
        coords_head=None,
        feat_stride=STRIDE,
        heatmap_head=None,
        cls_head=None,
        loss_weights={"vector": loss_weights[task_name]},
    )


# ----------------- data ---------------------------
# longside square dataset
train_recs = [
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_WFLW.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_tianjin201806.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_tianjin201806_plus_201808.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_tianjin201806_plus_201807.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_simrgb_office.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_simrgb_office_plus_201807.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_rgbinfra_downstairs.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_rgbinfra_downstairs_plus_201806.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_purchased_2nd_rgbinfra.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_purchased_1st_rgbinfra.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_purchased_1st_rgbinfra_20181127.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_purchased_1st_rgbinfra_20181126.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_pts72_refresh.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_pts72_refresh_plus_201807.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_njsmoke_20181102.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_pts72_refresh_plus_201808.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_njoffice201809.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_neimeng_ruilian_side_20181210.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_own_ruilian_smoke_20181212.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_neimeng_ruilian_front.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_nebulaV2_side1_wechat_20190325.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_nebulaV2_front_wechat_20190325.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_Menpo-3D.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_custom_sbao_yawn_20190701_02.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_custom_sbao_alarm_20190410.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_custom_alarm_20180615.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_carnet_realtest_20181129.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_AFLW2000-3D-Reannotated.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_custom_sbao_yawn_20190701_01.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_300W.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_300WLP.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_300VW-3D.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_300W-Testset-3D.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/train_Umd.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_IR2019_train_02.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_IR2019_train_03.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_IR2019_train_01.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_IR2019_train_00.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_gaze_68pts_train_04.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_gaze_68pts_train_02.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_gaze_68pts_train_01.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_gaze_68pts_train_00.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_gaze_68pts_train_03.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_68pts_guangqi_yawn_train.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_68pts_smoke_train.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_child_68pts_train_00.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_children_68pts_train_01.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_children_68pts_train_02.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_children_68pts_train_03.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_children_68pts_train_04.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_children_68pts_train_05.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_68pts_s202da_train_01.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_68pts_train_01.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_68pts_train_01.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_68pts_train_01.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_68pts_train_02.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_68pts_train_02.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_68pts_train_02.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_68pts_train_03.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_68pts_train_03.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_68pts_train_03.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_01.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_02.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_03.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_04.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_05.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_06.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_07.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_08.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_09.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_10.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_11.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_12.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_13.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_14.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_15.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_16.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_17.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_18.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_19.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_20.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_21.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_22.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_23.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_01.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_02.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_03.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_04.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_05.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_06.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_07.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_08.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_09.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_10.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_11.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_12.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_13.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_14.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_15.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_16.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_17.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_18.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_19.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_20.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_21.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_22.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_23.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_01.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_02.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_03.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_04.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_05.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_06.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_07.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_08.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_09.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_10.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_11.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_12.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_13.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_14.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_15.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_16.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_17.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_18.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_19.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_20.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_21.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_22.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_train_23.rec",  # noqa
]

val_recs = [
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks68_rgb_test.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks68_ir_test.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_test_01.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_rgb_68pts_yawn_test.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_68pts_yawn_test.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_phone_68pts_test_01.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_rgb_68pts_mask_test.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_68pts_gaze_test.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_68pts_emotion_test.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_children_68pts_test_01.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_68pts_sunglasses_test.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks68_rgb_test_clockwise60.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks68_rgb_test_clockwise30.rec",  # noqa
]

val_recs = [
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks68_rgb_test.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks68_ir_test.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_large_pose_68pts_test_01.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_68pts_largepose_test.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_rgb_68pts_yawn_test.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_68pts_yawn_test.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_phone_68pts_test_01.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_rgb_68pts_mask_test.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_68pts_gaze_test.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_ir_68pts_emotion_test.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_children_68pts_test_01.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks_68pts_sunglasses_test.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks68_rgb_test_clockwise60.rec",  # noqa
    f"{root}/face/landmark/68pts/rec/rec_long_square/lmks68_rgb_test_clockwise30.rec",  # noqa
]

val_data_descs = [
    "test_lmks_pose_Lmks68RGBTest_lmks68",
    "test_lmks_pose_Lmks68IRTest_lmks68",
    "lmks_large_pose_68pts_test_01",
    "lmks_ir_68pts_largepose_test",
    "lmks_rgb_68pts_yawn_test",
    "lmks_ir_68pts_yawn_test",
    "lmks_ir_phone_68pts_test_01",
    "lmks_rgb_68pts_mask_test",
    "lmks_ir_68pts_gaze_test",
    "lmks_ir_68pts_emotion_test",
    "lmks_children_68pts_test_01",
    "lmks_68pts_sunglasses_test",
    "lmks68_rgb_test_clockwise60",
    "lmks68_rgb_test_clockwise30",
]


# transform
train_transforms = [
    dict(
        type="CropRecROI",
        crop_type="random",
        target_shape=(INPUT_H, INPUT_W, 3),
        base_roi=[51, 51, 205, 205],
        crop_jitter_range=0.2,
        random_type="gaussian",
    ),
    dict(
        type="RandomFlip",
        px=0.5,
        # flip_ldmk_wo_1=True,  # note: The current master does not have this parameter  # noqa
    ),
    dict(
        type="RandomShiftRotateScale",
        rotate_prob=0.5,
        bounded=True,
        max_rotate_angle=30,
        resize=True,
    ),
    dict(
        type="GaussianNoise",
        prob=0.2,
        mean=0,
        sigma=2,
    ),
    dict(
        type="RandomNoise",
        prob=0.2,
        min=-5,
        max=5,
    ),
    dict(
        type="SaltPepperNoise",
        prob=0.2,
        s_ratio=0.05,
        p_ratio=0.05,
    ),
    dict(
        type="GaussianBlur",
        p=0.2,
        kernel_size_min=3,
        kernel_size_max=9,
        sigma_min=1.0,
        sigma_max=5.0,
    ),
    dict(
        type="MotionBlur",
        p=0.2,
        length_min=2,
        length_max=10,
        angle_min=1,
        angle_max=359,
    ),
    dict(
        type="RandomGray",
        p=0.3,
    ),
    dict(
        type="RandomOcclusion",
        prob=0.5,
        occ_type="whole",
        occ_num=1,
        size_ratio=0.7,
    ),
    dict(
        type="GenerateGaussianVector",
        num_ldmk=NUM_LDMK,
        feat_stride=STRIDE,
        vector_size=(TARGET_W, TARGET_H),
        sigma=2,
        encoding_method="standard",
    ),
    dict(type="ToTensor"),
    dict(
        type="Lighting",
        prob=1.0,
        alphastd=0.2,
    ),
    dict(
        type="TorchVisionAdapter",
        interface="ColorJitter",
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0,
    ),
    dict(
        type="FaceMtlTransformLabel",
        name=task_name,
    ),
]


val_transforms = [
    dict(
        type="CropRecROI",
        crop_type="center",
        target_shape=(INPUT_H, INPUT_W, 3),
        base_roi=[51, 51, 205, 205],
        crop_jitter_range=0.0,
    ),
    dict(
        type="GenerateGaussianVector",
        num_ldmk=NUM_LDMK,
        feat_stride=STRIDE,
        vector_size=(TARGET_H, TARGET_W),
        sigma=2,
        encoding_method="standard",
    ),
    dict(type="ToTensor"),
    dict(
        type="FaceMtlTransformLabel",
        name=task_name,
    ),
]


# datasets
train_datasets = [
    dict(
        type="LdmkRecDataset",
        filename=rec_file,
        num_ldmk=NUM_LDMK,
        use_3d=False,
        task_type="face",
        transforms=train_transforms,
        ldmk_pairs=LDMK_PAIRS,
    )
    for rec_file in train_recs
]

val_datasets = [
    dict(
        type="LdmkRecDataset",
        filename=rec_file,
        data_desc=data_desc_i,
        num_ldmk=NUM_LDMK,
        use_3d=False,
        task_type="face",
        transforms=val_transforms,
        ldmk_pairs=LDMK_PAIRS,
    )
    for rec_file, data_desc_i in zip(val_recs, val_data_descs)
]


# dataloader
train_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type=torch.utils.data.ConcatDataset,
        datasets=train_datasets,
    ),
    persistent_workers=True,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu_ldmk,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)


val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(type=torch.utils.data.ConcatDataset, datasets=val_datasets),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu_ldmk,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)


test_data_loader_list = []
for rec_file, data_desc_i in zip(val_recs, val_data_descs):
    test_loader = dict(
        type=torch.utils.data.DataLoader,
        dataset=dict(
            type="LdmkRecDataset",
            filename=rec_file,
            data_desc=data_desc_i,
            num_ldmk=NUM_LDMK,
            use_3d=False,
            task_type="face",
            transforms=val_transforms,
            ldmk_pairs=LDMK_PAIRS,
        ),
        sampler=dict(type=torch.utils.data.DistributedSampler),
        batch_size=batch_size_per_gpu_ldmk,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_data_loader_list.append(test_loader)


# ----------------- metric ---------------------------
def update_metric(metrics, batch, model_outs):
    if task_name in model_outs:
        model_outs = model_outs[task_name]
        for metric, key in zip(metrics, model_outs):
            metric.update(model_outs[key])


train_metrics = [
    dict(type="LossShow", name="LdmkLoss"),
    dict(type="LossShow", name="loss"),
]


test_metrics = []
for i in range(len(val_data_descs)):
    if "large_pose" in val_data_descs[i] or "largepose" in val_data_descs[i]:
        test_metrics.append(
            dict(
                type="NormalizedMeanError",
                num_ldmk=NUM_LDMK,
                norm_type="SIZE",
                mode="vector",
                decoding_method="diff",
                feat_stride=4,
                name=f"LdmkION_{i}",
            )
        )
    else:
        test_metrics.append(
            dict(
                type="NormalizedMeanError",
                num_ldmk=NUM_LDMK,
                norm_type="ION",
                mode="vector",
                decoding_method="diff",
                feat_stride=4,
                name=f"LdmkION_{i}",
            )
        )


metric_updater = dict(
    type="MetricUpdater",
    metrics=train_metrics,
    metric_update_func=update_metric,
    step_log_freq=step_log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
)


def get_update_metric(data_desc):
    def _update(metrics, _batch, _model_outs):
        _model_outs = _model_outs[task_name]
        batch = copy.deepcopy(_batch)
        model_outs = copy.deepcopy(_model_outs)
        data_descs = batch[0][task_name]["data_desc"]
        gt_ldmk = model_outs["gt_ldmk"]
        pr_x = model_outs["pr_vector_x"]
        pr_y = model_outs["pr_vector_y"]
        imgs = _batch[0]["img"]
        img_list = []
        ldmk_list = []
        pr_x_list = []
        pr_y_list = []
        for i in range(len(data_descs)):
            if data_desc == data_descs[i]:
                img_list.append(imgs[i])
                ldmk_list.append(gt_ldmk[i])
                pr_x_list.append(pr_x[i])
                pr_y_list.append(pr_y[i])
        if len(ldmk_list) != 0:
            data = {
                "img": torch.stack(img_list),
                "gt_ldmk": torch.stack(ldmk_list),
                "pr_vector_x": torch.stack(pr_x_list),
                "pr_vector_y": torch.stack(pr_y_list),
            }
            for metric in metrics:
                metric.update(data)

    return _update


test_metric_updater_list = [
    dict(
        type="MetricUpdater",
        log_prefix=f"Validation {data_desc_i}\n",
        metrics=[test_metrics[i]],
        metric_update_func=get_update_metric(data_desc_i),
        step_log_freq=0,
        epoch_log_freq=1,
    )
    for i, data_desc_i in enumerate(val_data_descs)
]
