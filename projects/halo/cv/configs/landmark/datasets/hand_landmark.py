import os

from hatbc.filestream.bucket.client import get_bucket_client

prefix = "dmpv2://interaction/active/hand/lmdb/hand_kps_3d/"


train_datasets = [
    # external data
    "01_data_shujutang_v2",
    "03_data_shujutang",
    "03_data_shujutang_v2",
    # stereo in car
    "606579_1_stereo_incar_01",
    "606580_1_stereo_incar_01",
    "610592_1_stereo_incar_02",
    "610593_1_stereo_incar_02",
    "611358_1_stereo_incar_train_05",
    "611359_1_stereo_incar_train_06",
    "611360_1_stereo_incar_train_07",
    "611361_1_stereo_incar_train_08",
    "611028_1_stereo_incar_train_09",
    "611029_1_stereo_incar_train_10",
    "611030_1_stereo_incar_train_11",
    "611031_1_stereo_incar_train_12",
    "611032_1_stereo_incar_train_13",
    "611033_1_stereo_incar_train_14",
    "hand_kps_multi_view_in_car_train_cam_15",
    "hand_kps_multi_view_in_car_train_cam_16",
    "hand_kps_multi_view_in_car_train_cam_17",
    "hand_kps_multi_view_in_car_train_cam_18",
    "hand_kps_multi_view_in_car_train_cam_19",
    "hand_kps_multi_view_in_car_train_cam_20",
    "hand_kps_multi_view_in_car_train_cam_21",
    "hand_kps_multi_view_in_car_train_cam_22",
    # stereo_incar_train_01 & stereo_incar_train_02 与上面的重复
    # public
    "cmu_hand_kps_train",
    "coco_wholebody_train_v1.0",
    "cvpr_2019_augments_coco_train",
    "FreiHand_train",
    "FreiHand_train_v2",
    "GANeratedHands_train",
    "RHD_train",
    # AIOT
    "hand_kps_aiot_badcase_train_01",
    "hand_kps_x3_train_01",
    "hand_kps_x3_train_child",
    # C281
    "hand_kps_C281_common_gesture_train_01",
    "hand_kps_C281_common_gesture_train_02",
    "hand_kps_C281_common_gesture_train_03",
    "hand_kps_C281_common_gesture_train_04",
    "hand_kps_C281_common_gesture_train_05",
    "hand_kps_C281_common_gesture_train_06",
    "hand_kps_C281_common_gesture_train_07",
    "hand_kps_C281_common_gesture_train_08",
    "hand_kps_C281_finger-rot_gesture_train_01",
    "hand_kps_C281_special_gesture_train_01",
    "hand_kps_C281_special_gesture_train_02",
    "hand_kps_C281_special_gesture_train_03",
    "hand_kps_C281_special_gesture_train_04",
    # CD569
    "hand_kps_CA_supplement_train",
    "hand_kps_cd569_badcase_01",
    "hand_kps_cd569_badcase_train_02",
    "hand_kps_cd569_badcase_train_03",
    "hand_kps_cd569_normal_01",
    "hand_kps_in_car_rgb_train_18k",
    "hand_kps_in_car_rgb_train_part1",
    "hand_kps_in_car_rgb_train_part2",
    # E300
    "hand_kps_E300_common_gesture_train_01",
    "hand_kps_E300_common_gesture_train_02",
    # guangqi
    "hand_kps_guangqi_train_01",
    # H9
    "hand_kps_h9_ir_train_01",
    "hand_kps_h9_ir_train_02",
    "hand_kps_h9_ir_train_03",
    "hand_kps_h9_ir_train_04",
    "hand_kps_h9_ir_train_05",
    # T18
    "hand_kps_T18_common_gesture_train_01",
    "hand_kps_T18_common_gesture_train_02",
    "hand_kps_T18_common_gesture_train_03",
    "hand_kps_T18_common_gesture_train_04",
    # ToF
    "hand_kps_tof_ir_train_01",
    # multi-view in lab
    "multi_cam_matting_train_01",
    "multi_cam_matting_train_02",
    "multi_cam_matting_train_03",
    "multi_cam_matting_train_04",
    "multi_cam_matting_train_05",
    "multi_cam_matting_train_06",
    "multi_cam_matting_train_07",
    "multi_cam_matting_train_08",
    # realsense
    "realsense_rgbd_matting_train_02",
]

test_datasets = [
    # public
    "cmu_hand_kps_test",
    "coco_wholebody_val_v1.0",
    "GANeratedHands_test",
    "RHD_test",
    # AIOT
    "hand_kps_x3_test_01",
    "hand_kps_x3_test_02",
    "hand_kps_x3_test_03",
    "hand_kps_x3_test_child",
    # C281
    "hand_kps_C281_common_gesture_test_01",
    "hand_kps_C281_common_gesture_test_02",
    "hand_kps_C281_common_gesture_test_03",
    "hand_kps_C281_common_gesture_test_04",
    "hand_kps_C281_special_gesture_test_01",
    # CD569
    "hand_kps_cd569_badcase_test",
    "hand_kps_cd569_badcase_test_02",
    "hand_kps_in_car_rgb_test",
    "hand_kps_in_car_test_static",
    "hand_kps_rgb_test_dark_and_blur",
    # guangqi
    "hand_kps_guangqi_a11_test",
    "hand_kps_guangqi_a13_test",
    "hand_kps_guangqi_a18_test",
    "hand_kps_guangqi_a20_test",
    "hand_kps_guangqi_a29_test",
    "hand_kps_guangqi_a55_test",
    "hand_kps_guangqi_a57_test",
    "hand_kps_guangqi_a60_test",
    # H9
    "hand_kps_h9_ir_test_01",
    "hand_kps_h9_ir_test_02",
    # T18
    "hand_kps_T18_common_gesture_test_01"
    # multi-camera in lab
    "multi_cam_matting_test_01",
    "multi_cam_matting_test_04"
    # realsense
    "realsense_rgbd_matting_test_02",
    # stereo in car
    "stereo_incar_test_01",
]


datasets = {}
for key in train_datasets:
    datasets[key] = {}
    datasets[key]["image_lmdb"] = os.path.join(prefix, key, "image_lmdb")
    datasets[key]["anno_lmdb"] = os.path.join(prefix, key, "anno_lmdb")
for key in test_datasets:
    datasets[key] = {}
    datasets[key]["image_lmdb"] = os.path.join(prefix, key, "image_lmdb")
    datasets[key]["anno_lmdb"] = os.path.join(prefix, key, "anno_lmdb")


def get_datasets(names):
    bucket_client = get_bucket_client()
    image_lmdb_list = []
    anno_lmdb_list = []
    for name in names:
        image_lmdb_list.append(
            bucket_client.url_to_local(datasets[name]["image_lmdb"])
        )
        anno_lmdb_list.append(
            bucket_client.url_to_local(datasets[name]["anno_lmdb"])
        )
    return {
        "image_lmdb_list": image_lmdb_list,
        "anno_lmdb_list": anno_lmdb_list,
    }
