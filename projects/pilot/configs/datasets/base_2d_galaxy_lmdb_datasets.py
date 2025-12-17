from easydict import EasyDict

from hat.utils.logger import rank_zero_info

root = "dmpv2://matrix2"


def append_data_not_exists_in_galaxy(
    input_datapaths, output_datapaths, weight_append_ratio
):
    """
    1. Only process 2d data.
    2. Append data exists in input_datapaths and not exists in output_dataths to
    output_datapths;
    3. weight_append_ratio: if value of key <=0, the data of the key will not be appended.
    4. Format:
    weight_append_ratio: dict(
        class: ratio
    )
    """
    for key in output_datapaths:
        assert key in input_datapaths, f"{key} not in input_datapaths"
        assert key in weight_append_ratio
        ratio = weight_append_ratio[key]
        if ratio <= 0:
            rank_zero_info(f"skip all class {key} data")
            continue
        input_train_paths = input_datapaths[key].train_data_paths
        output_train_paths = output_datapaths[key].train_data_paths
        for data_path in input_train_paths:
            anno_path = data_path.data_path
            if anno_path not in [
                data_path.data_path for data_path in output_train_paths
            ]:
                data_path.sample_weight *= ratio
                rank_zero_info(
                    f"append and reset weight to {data_path.sample_weight}: {anno_path}"
                )
                output_datapaths[key].train_data_paths.append(data_path)


# pilot3.0 base 2d data
pilot3_day_datapaths = {
    "rear_plate_detection": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/data/pilot/plate/train_220319/0233/train",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/data/pilot/plate/train_220319/x3c/train",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/data/pilot/plate/train_220319/unknow/train",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_rear_plate_5v_CW-A23GB-A114-L_day_2pe/20220504-102122/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_rear_plate_5v_CO-OX3GB-A100-L_night_2pe/20220504-103634/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_rear_plate_5v_CW-A23GB-A100-L_night_2pe/20220504-103906/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_rear_plate_5v_CO-OX3GB-A100-L_day_2pe/20220504-103937/data",
                "sample_weight": 16,
            },
        ]
    },
    "person_face_detection": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_all_all_2pe/20220720-214115/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_all_all_2pe/20220720-205236/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_c385-5v_all_2pe/20220720-174226/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_c385-5v_all_2pe/20220720-174425/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_as33-5v_all_2pe/20220720-174221/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_cc02-5v_all_2pe/20220720-174714/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_maxfa-5v_all_2pe/20220720-180427/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_c385-5v_all_2pe/20220721-202711/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_as33-5v_all_2pe/20220809-212818/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_cc02-5v_all_2pe_2022-07-20_2022-08-09/20220810-114649/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_maxfa-5v_all_2pe_2022-07-20_2022-08-09/20220810-114506/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_pilot-pdt-5v_all_2pe_2022-07-20_2022-08-09/20220810-115032/data",
                "sample_weight": 16,
            },
        ]
    },
    "plate": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/jinchuan01.xiao/data/plate/0820/v220305/4pe/train",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/jinchuan01.xiao/data/plate/unknow/v220305/4pe/train",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/jinchuan01.xiao/data/plate/wissen/v220305/4pe/train",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/jinchuan01.xiao/data/plate/x3c/v220305/4pe/train",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/jinchuan01.xiao/data/plate/plate_v211108/train/train",
                "sample_weight": 1,
            },
        ]
    },
    "face": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_all_all_4pe/20220715-201117/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_all_all_4pe/20220715-184010/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_all_all_4pe/20220715-162048/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_c385-5v_all_4pe/20220715-181100/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_c385-5v_all_4pe/20220726-155303/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_c385-5v_all_4pe/20220726-143048/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_as33-5v_all_4pe/20220726-143247/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_cc02-5v_all_4pe/20220726-144014/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_maxfa-5v_all_4pe/20220726-151851/data",
                "sample_weight": 1,
            },
        ]
    },
    "person_occlusion_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/person_occlusion_classification/cls_person_occlusion_v1-20190810-train/data",
                "sample_weight": 16,
            },
        ]
    },
    "vehicle_occlusion_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_cls_data/cls_subbox_vehicle_full_occlusion_v2-20200825-train_old/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_cls_data/cls_subbox_vehicle_full_occlusion_v2-20201117-vehicle_full_0323/data",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_cls_data/cls_subbox_vehicle_full_occlusion_v2-20200621-vehicle_full_0220/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_cls_data/cls_vehicle_full_fp_v3-train_other/data",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_cls_data/cls_vehicle_full_fp-vehicle_full_fp-data_3900/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_cls_data/cls_vehicle_full_fp_v3-20201120-train_0323/data",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_occ_classification/pilot3.0_0233_data_before_0406_vehicle_occ_cls/data",
                "sample_weight": 16,
            },
        ]
    },
    "cyclist_classification": {"train_data_paths": []},
    "cyclist_kps": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/cyclist_kps/gluon_rec_width12/lf_new/data.train",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/2pe_cyclist_kps/cyc_kps_zhengwei/20220121-132406/data",
                "sample_weight": 4,
            },
        ]
    },
    "person_head_detection": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/person_head_detection/white_changan_20190212_min_head_height12/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/person_head_detection/id1-4_head_min_height_22/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/person_head_detection/id5-8_head_min_height_22/data",
                "sample_weight": 1.5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/person_head_detection/id24-27_head_min_height_22/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/2pe_person_head_detection/example_pack_densebox_ped_head_data_20220121_124814_0233_all/20220121-141027/data",
                "sample_weight": 4,
            },
        ]
    },
    "person_pose_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/person_pose_classification/cls_person_pose_v2_upsample-20190810-train/data",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/2pe_person_pose_classification/example_pack_densebox_ped_pose_data_20220121_122405_0233_all/20220121-140616/data",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/2pe_person_pose_classification/Pilot_person_pose_5v_CAMERA_CW_A23GB_A114_L_day/20220623-163511/data",
                "sample_weight": 10,
            },
        ]
    },
    "vehicle_light_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_light/full_1025/vehicle_light_train_1025_full",
                "sample_weight": 32,
            },
        ]
    },
    "vehicle_category": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_category_classification/pilot3.0_vehicle_8cls_senyun_day/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_category_classification/pilot3.0_vehicle_8cls_weissen_day/data",
                "sample_weight": 1,
            },
        ]
    },
    "person_age_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/person_age_classification/cls_person_age_haokai_v5_clean_Adult-20191012-train/data",
                "sample_weight": 30,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/person_age_classification/cls_person_age_haokai_v5_clean_Child-20191012-train/data",
                "sample_weight": 2,
            },
        ]
    },
    "person_orientation_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/person_orientation_classification/cls_person_orientation_v11_upsample-20191204_1-train/data",
                "sample_weight": 32,
            },
        ]
    },
    "cyclist": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/majiajia.data/CC02_V5.0/Pilot_cyclist_5v_CO-OX3GB-A100-L_day/20220611-151613/data",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/han.shen.data/cyc_data/20190522-remove-truck/20190522-remove-truck",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/han.shen.data/cyc_data/cyc_20190918",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/zhengwei.data/cyc_data/cyc_20200613_cn_390_person_tain_history",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/mono_0323_data/user/kaijun01.zhang/data/data_2020/0323_10635_quad_cyc_20200723/0323_10635_quad_cyc_20200723",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/mono_0323_data/user/kaijun01.zhang/data/data_2020/0323_0220_cyc_20201119/0323_0220_cyc_20201119",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/cyclist_detection/20210915_sensor0233_train_id_4218-9655_part6_day_upsample3_gt63_remove_empty_up300pix_30_heavyocc_normal/train",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/cyclist_detection/20220121-190855-x3c-all/data",
                "sample_weight": 30,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/20220510/cyclist/day/20220511-142409/data",
                "sample_weight": 30,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/cyclist_detection/Pilot_cyclist_5v_CAMERA_CW_A23GB_A114_L_day_5w/20220804-134412/data",
                "sample_weight": 24,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/majiajia.data/NonTarget_exp/exp5/Pilot_cyclist_5v__all/20220712-043858/data",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20220907-200731/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20220907-200731/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/cyclist_detection/fp/Veh_Head/Pilot_cyclist_5v_CW-A23GB-A114-L_day_fp/20220926-150141/data",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221011-224648/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221124-230709/data",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221127-200817/data",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Jira_Cam_5V/20221014-200602/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Jira_Cam_5V/20221127-194742/data",
                "sample_weight": 2,
            },
        ]
    },
    "person": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/majiajia.data/rare_pose_exp/Pilot_person_5v_CW-A23GB-A100-L_day/20220602-180009/data",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/majiajia.data/rare_pose_exp/Pilot_person_5v_CW-A23GB-A100-L_day/20220602-180009/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/majiajia.data/CC02_V5/Pilot_person_5v_CO-OX3GB-A100-L_day/20220611-170110/data",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/han.shen.data/ped_data/ped_20181210_train_12cam_update",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/han.shen.data/ped_data/ped_20190605_train_part_2_update",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/han.shen.data/ped_data/ped_20180814_train_gray_update",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/han.shen.data/ped_data/ped_20190918",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/zhengwei.data/ped_data/ped_20200613_cn_390_person_tain_history",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/han.shen.data/ped_data/ped_street_people_data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/mono_0323_data/user/kaijun01.zhang/data/data_2020/0323_0220_ped_20201119/0323_0220_ped_20201119",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/20210915_sensor0233_train_id_4218-9655_part6_day_upsample3_gt63_remove_empty/train",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/20211119-153000-big-person/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/Pilot_person_5v_CAMERA_CW_A23GB_A100_L_day_big_person/20220620-201707/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/20220214-031030-0233-toll-gate/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/20220329-151248-x3c-tunnel/data",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/tunnel_ped_by_zhongpeng/data",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/Pilot_person_5v_CAMERA_CW_A23GB_A114_L_day_5w/20220804-122458/data",
                "sample_weight": 24,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/20220121-192622-x3c-all/data",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/20220510/person/day/20220511-142910/data",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/majiajia.data/NonTarget_exp/exp5/Pilot_person_5v__all/20220708-063813/data",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20220907-200310/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/fp/Veh_Head/Pilot_person_5v_CW-A23GB-A114-L_day_fp/20220926-152912/data",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221011-223223/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Jira_Cam_5V/20221014-201823/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/Pilot_person_5v_all_all_Child/20221015-081347/data",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221124-233317/data",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221127-201047/data",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Jira_Cam_5V/20221127-194559/data",
                "sample_weight": 2,
            },
        ]
    },
    "vehicle": {
        "train_data_paths": [
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/newdata_rec/densebox.train.basenewv1",
            #     "sample_weight": 10,
            #     "partition": "base",
            # },
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/newdata_rec/densebox.train.12camnewv1",
            #     "sample_weight": 10,
            #     "partition": "base",
            # },
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/newdata_rec/densebox.train.newdatav2",
            #     "sample_weight": 2,
            #     "partition": "base",
            # },
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/vehicle_full_data_1080_cam3900/det_vehicle_full_v1_0-20200527_v1/train.mayv9",
            #     "sample_weight": 5,
            #     "partition": "base",
            # },
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/vehicle_full_data_1080/det_vehicle_full_v1_0-20200102_v1/train",
            #     "sample_weight": 5,
            #     "partition": "base",
            # },
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/mono_0323_data/user/yonggang.yang/data/vehicle_full_data_0323/det_vehicle_full_v1_0-20201118_v1/train",
            #     "sample_weight": 5,
            #     "partition": "base",
            # },
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/mono_0323_data/user/yonggang.yang/data/vehicle_full_data_1080/det_vehicle_full_v1_0-20200706_v1/train",
            #     "sample_weight": 5,
            #     "partition": "base",
            # },
            # # mcp全量白天全车数据 约19w张
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot3.0_0233_data_before_0714_day/train",
            #     "sample_weight": 20,
            #     "partition": "base",
            # },
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection_tmp/pilot3.0_0233_data_before_0406_bigtruck_new/data",
            #     "sample_weight": 5,
            #     "partition": "base",
            # },
            # # 白天 韦森 10w图
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_local/20210827-195610/data",
            #     "sample_weight": 10,
            #     "partition": "base",
            # },
            # # 白天 韦森  8.7w图
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_local_local/20210928-120519/data",
            #     "sample_weight": 10,
            #     "partition": "base",
            # },
            # full_vehicle_badcase_self_camera_local
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/badcase/full_vehicle_badcase_self_camera_local/20210827-212337/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # tunnel badcase 1.5w
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_badcase_local/20211019-194438/data",
                "sample_weight": 6,
                "partition": "difficult_scene",
            },
            # 韦森 day 3.3w
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_trace/20211124-163642/data",
                "sample_weight": 3,
                "partition": "base",
            },
            # X3C 3.3w
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_trace/20211129-163640/data",
                "sample_weight": 3,
                "partition": "base",
            },
            # X3C白天32246
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_x3c/20211222-124627/data",
                "sample_weight": 4,
                "partition": "base",
            },
            # 0233badcase白天9859
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_badcase/20211222-121533/data",
                "sample_weight": 2,
                "partition": "difficult_scene",
            },
            # 0233badcase白天6w
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_night_x3c/20220121-114854/data",
                "sample_weight": 3,
                "partition": "difficult_scene",
            },
            # cc02 白天 21745
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_data_5v_32393_2.1w_CAMERA_CO_OX3GB_A100_L_TIME_DAY_day/20220406-125044/data",
                "sample_weight": 2,
                "partition": "base",
            },
            # x3c 白天 45265
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_DG4W_data_5v_CAMERA_CO_OX3GB_D100_L_TIME_DAY_day/20220401-184015/data",
                "sample_weight": 4,
                "partition": "base",
            },
            # 逆光 7304
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_LIGHT07_data_5v_CAMERA_CP_OX3GB_D100_L_TIME_DAY_day/20220401-164602/data",
                "sample_weight": 2,
                "partition": "difficult_scene",
            },
            # x3c 白天 10763
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_29384_10763_CAMERA_CP_OX3GB_D100_L_TIME_DAY_day/20220426-101418/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # CW_A23GB_A114_L day 2.2W
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_data_5v_as33_25469_2.2w_CAMERA_CW_A23GB_A114_L_TIME_DAY_day/20220413-111602/data",
                "sample_weight": 3,
                "partition": "base",
            },
            # 0233WISSEN day 2.2W
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/polit3.0_v9.0_0315_0233_day/20220316-142817/data",
                "sample_weight": 3,
                "partition": "base",
            },
            # mono  白天夜间 大车异型 86868
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/det_vehicle_full_v1_0-20220407_v1/train",
                "sample_weight": 2,
                "partition": "special_vehicle",
            },
            # CW-A23GB-A100-L day 5.6k
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/pilot_vehicle_1v_CW-A23GB-A100-L_day/20220518-142747/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # CW-A23GB-A100-L 大车 35940
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_BIG_CAR_DAY_day/20220621-202410/data",
                "sample_weight": 4,
                "partition": "special_vehicle",
            },
            # CO_OX3GB_A100_L 大车 3219
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/pilot_vehicle_5v_CAMERA_CO_OX3GB_A100_L_TIME_BIG_CAR_DAY_day/20220621-192421/data",
                "sample_weight": 1,
                "partition": "special_vehicle",
            },
            # CW-A23GB-A100-L day 快递车数据
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_day/20220617-131419/data",
                "sample_weight": 1,
                "partition": "special_vehicle",
            },
            # CW_A23GB_A114_L 常规 白天 10842
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A114_L_day_normal/20220706-134742/data",
                "sample_weight": 2,
                "partition": "base",
            },
            # CW_A23GB_A114_L 逆光 白天 1900
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A114_L_day/20220621-161421/data",
                "sample_weight": 1,
                "partition": "difficult_scene",
            },
            # 各LTC小车挡大车
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_day_overlap_vehicle_Cam_5V/20220826-161108/data",
                "sample_weight": 0.5,
                "partition": "special_vehicle",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_day_overlap_vehicle_Cam_5V/20220826-163619/data",
                "sample_weight": 0.5,
                "partition": "special_vehicle",
            },
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_pilot_pdt_5v_CW-A23GB-A100-L_train_day_overlap_vehicle_Cam_5V/20220826-163951/data",
            #     "sample_weight": 0,
            # },
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_maxfa_5v_CP-OX3GB-D100-L_train_day_overlap_vehicle_Cam_5V/20220826-163258/data",
            #     "sample_weight": 0,
            # },
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_cc02_5v_CO-OX3GB-A100-L_train_day_overlap_vehicle_Cam_5V/20220826-162631/data",
            #     "sample_weight": 0,
            # },
            # 逆光 5694
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_pilot_pdt_5v_CW-A23GB-A100-L_train_day_glare_Cam_5V/20220729-121524/data",
                "sample_weight": 2,
                "partition": "difficult_scene",
            },
            # maxfa day special car 10259
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_54665_10259_CAMERA_CP_OX3GB_D100_L_TIME_DAY_day/20220707-160146/data",
                "sample_weight": 2,
                "partition": "special_vehicle",
            },
            # 全量数据快递车 7w
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_DAY_day/20220711-121816/data",
                "sample_weight": 1,
                "partition": "special_vehicle",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_DAY_day/20220711-121723/data",
                "sample_weight": 1,
                "partition": "special_vehicle",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_DAY_day/20220711-134219/data",
                "sample_weight": 4,
                "partition": "special_vehicle",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_DAY_day/20220711-160123/data",
                "sample_weight": 4,
                "partition": "special_vehicle",
            },
            # mono 三轮车2M
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_1v_OV10652_TIME_DAY_day/20220713-053741/data",
                "sample_weight": 7,
                "partition": "special_vehicle",
            },
            # mono 三轮车8M
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_1v_OV10652_TIME_DAY_day/20220713-061904/data",
                "sample_weight": 11,
                "partition": "special_vehicle",
            },
            # c385 6802
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20220902-200336/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # 环境误检 AS33 白天 1912
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_AS33_SCENE-FP_day/20221115-182150/data",
                "sample_weight": 4,
                "partition": "difficult_scene",
            },
            # 白天异型车 8010
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_maxfa_5v_CP-OX3GB-D100-L_train_day_special_vehicle_Cam_5V/20220926-103720/data",
                "sample_weight": 1,
                "partition": "special_vehicle",
            },
            # c385 白天 11114
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221012-122031/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # c385 JIRA-badcase 700
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CO-OX3GB-D100-L_JIRA-badcase_all/20221230-150000",
                "sample_weight": 5,
                "partition": "jira",
            },
            # c385白天jiracase异型车 182
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_special_vehicle_Cam_5V/20221009-193855/data",
                "sample_weight": 1,
                "partition": "jira",
            },
            # GALAXY 环境误检 白天 1512
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_day/20221115-190526/data",
                "sample_weight": 2,
                "partition": "difficult_scene",
            },
            # AS33 jira-badcase 白天/夜晚 369
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_AS33_JIRA-badcase_all/20221110-192849/data",
                "sample_weight": 1,
                "partition": "jira",
            },
            # c385 白天异型车 241
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_special_vehicle_Cam_5V/20221103-203708/data",
                "sample_weight": 2,
                "partition": "special_vehicle",
            },
            # c385 白天异型车 241
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CO_OX3GB_D100_L_TIME_DAY_SPECIAL_CAR_all/20221122-103953/data",
                "sample_weight": 2,
                "partition": "special_vehicle",
            },
            # GALAXY 环境误检 白天 531
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_day/20221230-151426",
                "sample_weight": 2,
                "partition": "difficult_scene",
            },
            # maxfa 白天异型车 17352
            {
                "data_path": f"{root}/multicam_pilot/data/train/vehicle_detection/project_maxfa_5v_CP-OX3GB-D100-L_train_day_special_vehicle_Cam_5V/20230320_111109",
                "sample_weight": 2,
                "partition": "special_vehicle",
            },
        ]
    },
    "vehicle_truncation_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_truncation_classification/pilot3.0_0233_data_before_0406_vehicle_truncation_cls/data",
                "sample_weight": 32,
            },
        ]
    },
    "semantic_parsing": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/data/default_parsing/train_data/mono3.0/16.2c/data_all_0220model_2020-06-22/parsing-720p10635-train-anno16.2_59070",
                "sample_weight": 7,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/data/default_parsing/train_data/mono3.0/16.2c/train-1080p-id20220324_62910",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_2872_20210804_anno16.2",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_950_20220527_anno16.2_day",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_2912_20220111_anno16.2",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_4876_20220415_anno16.2",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_side_roundview_num_15836_20220111_anno16.2_sy_day",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_rear_roundview_num_4671_20220111_anno16.2_sy_day",
                "sample_weight": 2.5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_side_roundview_num_13566_20220111_anno16.2_ws_day",
                "sample_weight": 8.5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_rear_roundview_num_3457_20220111_anno16.2_ws_day",
                "sample_weight": 2.5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_side_roundview_num_16089_20220111_anno16.2_ws_night",
                "sample_weight": 9,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_rear_roundview_num_3504_20220111_anno16.2_ws_night",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4987_20220111_anno16.2_x3c_night",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4300_20220214_anno16.2_x3c_night",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4060_20220428_anno16.2_x3c_night",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4500_20220111_anno16.2_x3c_day",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_10228_20220214_anno16.2_x3c_day",
                "sample_weight": 9,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_1010_20220428_anno16.2_x3c_day",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_2671_20220513_anno16.2_cc02_day",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_1202_20220602_anno16.2_as33_day",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_1660_20220825_anno16.2_as33_day/20220825-140857/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_4247_anno16.2_as33_day/data",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_176_anno16.2_c385_day_86907/20221011-011836/data",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_318_anno16.2_c385_day_86913/20221011-012023/data",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_604_anno16.2_c385_day_88317/20221011-010935/data",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_618_anno16.2_c385_day_88356/20221011-010444/data",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_2770_anno16.2_c385_day_90719/20221011-022443/data",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_1357_anno16.2_c385_day_20221123/data",
                "sample_weight": 4,
            },
        ]
    },
    "lane_parsing": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_18772_20210419_5_2classes",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_8571_20210524_5_2classes",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_24305_20210711_5_2classes",
                "sample_weight": 12,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_num_17311_20210711_5_2classes",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_28019_20210803_5_2classes",
                "sample_weight": 12,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_num_21736_20210803_5_2classes",
                "sample_weight": 12,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_71524_20210825_5_2classes",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_num_13711_20210825_5_2classes",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_55922_20211018_5_2classes",
                "sample_weight": 15,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_num_36387_20211018_5_2classes",
                "sample_weight": 15,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_x3c_num_160_20211129_5_2classes",
                "sample_weight": 0.1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_wissen_num_78742_20211128_5_2classes",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_x3c_num_22101_20211129_5_2classes",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_wissen_num_4549_20211129_5_2classes",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_x3c_num_47972_20211220_5_2classes",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_x3c_num_16073_20211220_5_2classes",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_x3c_num_64275_20220118_5_2classes",
                "sample_weight": 30,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_x3c_num_27923_20220118_5_2classes",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_x3c_num_14440_20220214_5_2classes",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_x3c_num_2924_20220214_5_2classes",
                "sample_weight": 0.2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_day_x3c_num_24851_20220328_5_2classes",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_night_x3c_num_5356_20220328_5_2classes",
                "sample_weight": 0.2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_day_x3c_D100L_num_17734_20220424_5_2classes",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_day_x3c_CO_OX3GB_A100L_num_7589_20220505_5_2classes",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_night_x3c_CO_OX3GB_A100L_num_745_20220505_5_2classes",
                "sample_weight": 0.5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_0233_CW_A23GB_A114L_num_4481_20220525_5_2classes",
                "sample_weight": 2.3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_0233_CW_A23GB_A114L_num_5460_20220525_5_2classes",
                "sample_weight": 2.5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_0233_CW_A23GB_A114L_num_6271_20220525_5_2classes",
                "sample_weight": 3.1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_0233_CW_A23GB_A114L_num_6551_20220608_5_2classes",
                "sample_weight": 3.3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_0233_CW_A23GB_A114L_num_1995_20220621_5_2classes",
                "sample_weight": 0.5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_220829/lane_parsing_train_day_badcase_as33/20220727-223306/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_221141/lane_parsing_train_day_badcase_as33/20220727-222539/data",
                "sample_weight": 0.4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_221458/lane_parsing_train_day_badcase_as33/20220727-222511/data",
                "sample_weight": 0.2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_221812/lane_parsing_train_day_badcase_as33/20220727-222648/data",
                "sample_weight": 0.1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_222305/lane_parsing_train_day_special_as33/20220727-223226/data",
                "sample_weight": 0.1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_222623/lane_parsing_train_day_special_as33/20220727-223356/data",
                "sample_weight": 0.06,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_222930/lane_parsing_train_day_special_as33/20220727-223812/data",
                "sample_weight": 0.1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_223832/lane_parsing_train_day_normal_as33/20220727-224554/data",
                "sample_weight": 0.09,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_224147/lane_parsing_train_day_normal_as33/20220727-224901/data",
                "sample_weight": 0.06,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_224502/lane_parsing_train_day_normal_as33/20220727-225403/data",
                "sample_weight": 0.2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221012_000716/pilot_lane_parsing_annoset/20221012-010546/data",
                "sample_weight": 2.0,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221012_005334/pilot_lane_parsing_annoset/20221012-012546/data",
                "sample_weight": 2.0,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221012_012136/pilot_lane_parsing_annoset/20221012-015617/data",
                "sample_weight": 1.0,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221012_014836/pilot_lane_parsing_annoset/20221012-022211/data",
                "sample_weight": 1.0,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221126_123750/pilot_lane_parsing_annoset/20221126-132031/data",
                "sample_weight": 2.0,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221126_125221/pilot_lane_parsing_annoset/20221126-132842/data",
                "sample_weight": 2.0,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221126_130644/pilot_lane_parsing_annoset/20221126-133734/data",
                "sample_weight": 1.0,
            },
        ]
    },
    "vehicle_wheel_kps": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_kps/vehicle_2_kps/vehicle_2_kps_all_20201021.train",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_kps_detection/release_rec/vehicle_2_kps_add0323cutin_20201125",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_day_0233/20220119-212808/data",
                "sample_weight": 50,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_day_X3C/20220119-122143/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_day_X3C/20220224-131626/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_CAMERA_X3C_day/20220315-163131/data",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_CAMERA_X3C_day/20220425-161851/data",
                "sample_weight": 1,
            },
        ]
    },
    "vehicle_ground_line": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_day_0233/20220119-221935/data",
                "sample_weight": 50,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_day_X3C/20220119-125418/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_day_X3C/20220224-133404/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_CAMERA_X3C_day/20220315-163208/data",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_CAMERA_X3C_day/20220425-161825/data",
                "sample_weight": 1,
            },
        ]
    },
    "vehicle_flank": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehFlank/kVehFlank_Cam_5V/20220706-213158/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehFlank/kVehFlank_ar0233-RGGB_train_Cam_5V/20220706-205154/data",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehFlank/kVehFlank_ar0233-WISSEN_train_Cam_5V/20220706-165341/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehFlank/kVehFlank_ar0233-WISSEN_train_day_big_vehicle_Cam_5V/20220706-123556/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehFlank/kVehFlank_x3c_train_day_Cam_5V/20220706-112001/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehFlank/kVehFlank_project_cc02_5v_CO-OX3GB-A100-L_train_day_big_vehicle_Cam_5V/20220706-102652/data",
                "sample_weight": 1,
            },
        ]
    },
    "vehicle_wheel_detection": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_wheel_detection/det_subbox_vehicle_wheel_belongto_clean-veh_wheel-20210407_train_clean_323_all_trans_pb_rec/train",
                "sample_weight": 171,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_wheel_detection/MCP_wheel_detection_day_0927/train",
                "sample_weight": 85,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehWheelBBox2D/kVehWheelBBox2D_train_day_big_vehicle_Cam_5V_Cam_5V/20220617-193224/data",
                "sample_weight": 2,
            },
        ]
    },
    "rear": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v3-20190726-train_base/data",
                "sample_weight": 8,
                "partition": "base",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v3-20190726-train_gray/data",
                "sample_weight": 6,
                "partition": "base",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v3-20190726-train_night/data",
                "sample_weight": 3,
                "partition": "base",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v6-20200624-train_390_5200_day_FrontRear/data",
                "sample_weight": 2,
                "partition": "base",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v6-20200624-train_390_5200_day_NonFrontRear/data",
                "sample_weight": 3,
                "partition": "base",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v6-20200624-train_390_5200_night/data",
                "sample_weight": 2,
                "partition": "base",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/mono_0323_data/user/ni.jiang/data/vehicle_rear_data/det_vehicle_rear_v8-20201119-train_part_4_0323_0220_exclude_20200601_exclude_part33/data",
                "sample_weight": 12,
                "partition": "base",
            },
            # 维森 白天 330660
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_day_0233/20220119-012241/data",
                "sample_weight": 50,
                "partition": "base",
            },
            # X3C 白天 12906
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_day_X3C/20220224-113552/data",
                "sample_weight": 2,
                "partition": "base",
            },
            # X3C 白天 15688
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CAMERA_X3C_day/20220315-163248/data",
                "sample_weight": 2,
                "partition": "base",
            },
            # X3C 白天 13505
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CAMERA_X3C_day/20220401-140046/data",
                "sample_weight": 2,
                "partition": "base",
            },
            # AS33 day 24985
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/pilot_rear_5v_AS33_day/20220412-220556/data",
                "sample_weight": 3,
                "partition": "base",
            },
            # PDT 大车 day 3205
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/pilot_rear_5v_CW_A23GB_A100_L_big_car_day/20220412-212012/data",
                "sample_weight": 1,
                "partition": "badcase",
            },
            # PROJECT_CC02_5V 白天 4910
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_PROJECT_CC02_5V_day/20220407-195106/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # PROJECT_CC02_5V 白天 8526 annoset(16130)
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_PROJECT_CC02_5V_day/20220510-214104/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # PROJECT_AS33_5V 白天 2113
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CW-A23GB-A114-L_day/20220510-103100/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # PROJECT_c385_5V 白天 4879
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_rear/pilot_rear_5v_CO-OX3GB-D100-L_day/20221127-140301/data",
                "sample_weight": 1,
                "partition": "base",
            },
        ]
    },
    "rear_occlusion_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_data/vehicle_rear_occlusion_cls_data/cls_subbox_rear_occlusion_v8-20190719-train_base_night_gray-cn93-cn750-model-score0.18/data",
                "sample_weight": 12,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_data/vehicle_rear_occlusion_cls_data/cls_subbox_rear_occlusion_v8-20200903-train_part_4_0323_ispDefault_2280x1080_exclude_part33-cn750-cn759/data",
                "sample_weight": 18,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_data/vehicle_rear_occlusion_cls_data/cls_rear_neg_v4-20200901-fp_train/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_day_0233/20220119-041133/data",
                "sample_weight": 50,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_day_X3C/20220119-120507/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_day_X3C/20220224-123226/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_CAMERA_X3C_day/20220316-102102/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_CAMERA_X3C_day/20220401-160904/data",
                "sample_weight": 2,
            },
        ]
    },
    "rear_part_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_data/vehicle_rear_cls_data/cls_vehicle_part_3cls_v1-old/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_data/vehicle_rear_cls_data/cls_vehicle_part_3cls_v1-hard/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_data/vehicle_rear_cls_data/cls_vehicle_part_3cls_v1-res/data",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_data/vehicle_rear_cls_data/cls_vehicle_part_3cls_v1-0323/data",
                "sample_weight": 9,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_day_0233/20220119-210054/data",
                "sample_weight": 50,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_day_X3C/20220119-120509/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_day_X3C/20220224-130525/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_CAMERA_X3C_day/20220315-163229/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_rear_part_classification/pilot_rear_part_5v_CAMERA_CW_A23GB_A114_L_day_bigcar/20220822-093413/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_rear_part_classification/pilot_rear_part_5v_CO-OX3GB-D100-L_day/20221127-145644/data",
                "sample_weight": 1,
            },
        ]
    },
}

pilot3_night_datapaths = {
    "rear_plate_detection": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/data/pilot/plate/train_220319/0233/train",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/data/pilot/plate/train_220319/x3c/train",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/data/pilot/plate/train_220319/unknow/train",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_rear_plate_5v_CW-A23GB-A114-L_day_2pe/20220504-102122/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_rear_plate_5v_CO-OX3GB-A100-L_night_2pe/20220504-103634/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_rear_plate_5v_CW-A23GB-A100-L_night_2pe/20220504-103906/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_rear_plate_5v_CO-OX3GB-A100-L_day_2pe/20220504-103937/data",
                "sample_weight": 16,
            },
        ]
    },
    "person_face_detection": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_all_all_2pe/20220720-214115/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_all_all_2pe/20220720-205236/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_c385-5v_all_2pe/20220720-174226/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_c385-5v_all_2pe/20220720-174425/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_as33-5v_all_2pe/20220720-174221/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_cc02-5v_all_2pe/20220720-174714/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_maxfa-5v_all_2pe/20220720-180427/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_c385-5v_all_2pe/20220721-202711/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_as33-5v_all_2pe/20220809-212818/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_cc02-5v_all_2pe_2022-07-20_2022-08-09/20220810-114649/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_maxfa-5v_all_2pe_2022-07-20_2022-08-09/20220810-114506/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_pilot-pdt-5v_all_2pe_2022-07-20_2022-08-09/20220810-115032/data",
                "sample_weight": 16,
            },
        ]
    },
    "person_occlusion_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/person_occlusion_classification/cls_person_occlusion_v1-20190810-train/data",
                "sample_weight": 16,
            },
        ]
    },
    "vehicle_occlusion_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_cls_data/cls_subbox_vehicle_full_occlusion_v2-20200825-train_old/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_cls_data/cls_subbox_vehicle_full_occlusion_v2-20201117-vehicle_full_0323/data",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_cls_data/cls_subbox_vehicle_full_occlusion_v2-20200621-vehicle_full_0220/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_cls_data/cls_vehicle_full_fp_v3-train_other/data",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_cls_data/cls_vehicle_full_fp-vehicle_full_fp-data_3900/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_cls_data/cls_vehicle_full_fp_v3-20201120-train_0323/data",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_occ_classification/pilot3.0_0233_data_before_0406_vehicle_occ_cls/data",
                "sample_weight": 16,
            },
        ]
    },
    "cyclist_classification": {"train_data_paths": []},
    "cyclist_kps": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/cyclist_kps/gluon_rec_width12/lf_new/data.train",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/2pe_cyclist_kps/cyc_kps_zhengwei/20220121-132406/data",
                "sample_weight": 4,
            },
        ]
    },
    "person_head_detection": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/person_head_detection/white_changan_20190212_min_head_height12/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/person_head_detection/id1-4_head_min_height_22/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/person_head_detection/id5-8_head_min_height_22/data",
                "sample_weight": 1.5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/person_head_detection/id24-27_head_min_height_22/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/2pe_person_head_detection/example_pack_densebox_ped_head_data_20220121_124814_0233_all/20220121-141027/data",
                "sample_weight": 4,
            },
        ]
    },
    "person_pose_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/person_pose_classification/cls_person_pose_v2_upsample-20190810-train/data",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/2pe_person_pose_classification/example_pack_densebox_ped_pose_data_20220121_122405_0233_all/20220121-140616/data",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/2pe_person_pose_classification/Pilot_person_pose_5v_CAMERA_CW_A23GB_A114_L_night/20220706-113735/data",
                "sample_weight": 10,
            },
        ]
    },
    "vehicle_light_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_light/full_1025/vehicle_light_train_1025_full",
                "sample_weight": 32,
            },
        ]
    },
    "vehicle_category": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_category_classification/pilot3.0_vehicle_8cls_senyun_night/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_category_classification/pilot3.0_vehicle_8cls_weissen_night/data",
                "sample_weight": 3,
            },
        ]
    },
    "person_age_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/person_age_classification/cls_person_age_haokai_v5_clean_Adult-20191012-train/data",
                "sample_weight": 30,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/person_age_classification/cls_person_age_haokai_v5_clean_Child-20191012-train/data",
                "sample_weight": 2,
            },
        ]
    },
    "person_orientation_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/person_orientation_classification/cls_person_orientation_v11_upsample-20191204_1-train/data",
                "sample_weight": 32,
            },
        ]
    },
    "cyclist": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/majiajia.data/CC02_V5/Pilot_cyclist_5v_CO-OX3GB-A100-L_night/20220611-163932/data",
                "sample_weight": 40,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/han.shen.data/cyc_data/20190522-remove-truck/20190522-remove-truck",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/han.shen.data/cyc_data/cyc_20190918",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/zhengwei.data/cyc_data/cyc_20200613_cn_390_person_tain_history",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/mono_0323_data/user/kaijun01.zhang/data/data_2020/0323_10635_quad_cyc_20200723/0323_10635_quad_cyc_20200723",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/mono_0323_data/user/kaijun01.zhang/data/data_2020/0323_0220_cyc_20201119/0323_0220_cyc_20201119",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/cyclist_detection/20210915_sensor0233_train_id_4218-9655_part6_day_upsample3_gt63_remove_empty_up300pix_30_heavyocc_normal/train",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/cyclist_detection/20220121-190855-x3c-all/data",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/20220510/cyclist/night/20220511-124542/data",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/cyclist_detection/Pilot_cyclist_5v_CAMERA_CW_A23GB_A114_L_night/20220624-203955/data",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/majiajia.data/NonTarget_exp/exp6/Pilot_cyclist_5v_all_night/20220713-225739/data",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_Cam_5V/20221128-104916/data",
                "sample_weight": 3,
            },
            # AS33 vehicle glare annoset(89252) 4274
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_AS33_glare_fp_train_night/20230131-114216/data",
                "sample_weight": 1,
            },
        ]
    },
    "person": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/majiajia.data/CC02_V5/Pilot_person_5v_CO-OX3GB-A100-L_night/20220611-164225/data",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/han.shen.data/ped_data/ped_20181210_train_12cam_update",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/han.shen.data/ped_data/ped_20190605_train_part_2_update",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/han.shen.data/ped_data/ped_20180814_train_gray_update",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/han.shen.data/ped_data/ped_20190918",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/zhengwei.data/ped_data/ped_20200613_cn_390_person_tain_history",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/han.shen.data/ped_data/ped_street_people_data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/mono_0323_data/user/kaijun01.zhang/data/data_2020/0323_0220_ped_20201119/0323_0220_ped_20201119",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/20210915_sensor0233_train_id_4218-9655_part6_day_upsample3_gt63_remove_empty/train",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/20211119-153000-big-person/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/20220121-192622-x3c-all/data",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/tunnel_ped_by_zhongpeng/data",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/20220329-151248-x3c-tunnel/data",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/20220510/person/night/20220511-124631/data",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/Pilot_person_5v_CAMERA_CW_A23GB_A114_L_day_5w/20220826-121734/data",
                "sample_weight": 24,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/Pilot_person_5v_CAMERA_CW_A23GB_A100_L_night_big_person/20220620-201624/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/majiajia.data/NonTarget_exp/exp6/Pilot_person_5v_all_night/20220713-233017/data",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_Cam_5V/20221128-104636/data",
                "sample_weight": 3,
            },
            # AS33 vehicle glare annoset(88986) 4274
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_AS33_glare_fp_train_night/20230130-121601/data",
                "sample_weight": 4,
            },
        ]
    },
    "vehicle": {
        "train_data_paths": [
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/newdata_rec/densebox.train.basenewv1",
            #     "sample_weight": 10,
            #     "partition": "base",
            # },
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/newdata_rec/densebox.train.12camnewv1",
            #     "sample_weight": 10,
            #     "partition": "base",
            # },
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/newdata_rec/densebox.train.newdatav2",
            #     "sample_weight": 2,
            #     "partition": "base",
            # },
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/vehicle_full_data_1080_cam3900/det_vehicle_full_v1_0-20200527_v1/train.mayv9",
            #     "sample_weight": 5,
            #     "partition": "base",
            # },
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/vehicle_full_data_1080/det_vehicle_full_v1_0-20200102_v1/train",
            #     "sample_weight": 5,
            #     "partition": "base",
            # },
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/mono_0323_data/user/yonggang.yang/data/vehicle_full_data_0323/det_vehicle_full_v1_0-20201118_v1/train",
            #     "sample_weight": 5,
            #     "partition": "base",
            # },
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/mono_0323_data/user/yonggang.yang/data/vehicle_full_data_1080/det_vehicle_full_v1_0-20200706_v1/train",
            #     "sample_weight": 5,
            #     "partition": "base",
            # },
            # # mcp全量白天全车数据 约19w张
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot3.0_0233_data_before_0714_day/train",
            #     "sample_weight": 20,
            #     "partition": "base",
            # },
            # # 白天 韦森 10w图
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_local/20210827-195610/data",
            #     "sample_weight": 10,
            #     "partition": "base",
            # },
            # # 白天 韦森  8.7w图
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_local_local/20210928-120519/data",
            #     "sample_weight": 10,
            #     "partition": "base",
            # },
            # full_vehicle_badcase_self_camera_local
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/badcase/full_vehicle_badcase_self_camera_local/20210827-212337/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # tunnel badcase 1.5w
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_badcase_local/20211019-194438/data",
                "sample_weight": 6,
                "partition": "difficult_scene",
            },
            # 韦森 day 3.3w
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_trace/20211124-163642/data",
                "sample_weight": 3,
                "partition": "base",
            },
            # 0233 夜晚 全量数据 222100
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_night_0233/20220119-002848/data",
                "sample_weight": 60,
                "partition": "base",
            },
            # X3C 3.3w
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_trace/20211129-163640/data",
                "sample_weight": 3,
                "partition": "base",
            },
            # X3C夜间19185
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_night_x3c/20211222-132550/data",
                "sample_weight": 3,
                "partition": "base",
            },
            # X3C白天32246
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_x3c/20211222-124627/data",
                "sample_weight": 4,
                "partition": "base",
            },
            # 0233badcase白天9859
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_badcase/20211222-121533/data",
                "sample_weight": 2,
                "partition": "base",
            },
            # 0233badcase白天6w
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_night_x3c/20220121-114854/data",
                "sample_weight": 3,
                "partition": "base",
            },
            # cc02 白天 21745
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_data_5v_32393_2.1w_CAMERA_CO_OX3GB_A100_L_TIME_DAY_day/20220406-125044/data",
                "sample_weight": 2,
                "partition": "base",
            },
            # cc02 夜晚 0.68w
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_data_5v_32394_0.68w_CAMERA_CO_OX3GB_A100_L_TIME_NIGHT_night/20220406-124214/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # x3c 白天 45265
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_DG4W_data_5v_CAMERA_CO_OX3GB_D100_L_TIME_DAY_day/20220401-184015/data",
                "sample_weight": 4,
                "partition": "base",
            },
            # x3c 白天 10763
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_29384_10763_CAMERA_CP_OX3GB_D100_L_TIME_DAY_day/20220426-101418/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # CW_A23GB_A114_L day 2.2W
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_data_5v_as33_25469_2.2w_CAMERA_CW_A23GB_A114_L_TIME_DAY_day/20220413-111602/data",
                "sample_weight": 3,
                "partition": "base",
            },
            # 0233WISSEN day 2.2W
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/polit3.0_v9.0_0315_0233_day/20220316-142817/data",
                "sample_weight": 3,
                "partition": "base",
            },
            # mono  白天夜间 大车异型 86868
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/det_vehicle_full_v1_0-20220407_v1/train",
                "sample_weight": 2,
                "partition": "special_vehicle",
            },
            # CW-A23GB-A100-L dark night 31121
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/kVehBBox2D/kVehBBox2D_CW-A23GB-A100-L_train_night_dark_night/20220610-160750/data",
                "sample_weight": 4,
                "partition": "difficult_scene",
            },
            # CP-OX3GB-D100-L dark night 2763
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/kVehBBox2D/kVehBBox2D_project_maxfa_5v_CP-OX3GB-D100-L_train_night_dark_night/20220610-144900/data",
                "sample_weight": 1,
                "partition": "difficult_scene",
            },
            # CW-A23GB-A100-L dark night 2243
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_dark_night/20220610-144130/data",
                "sample_weight": 1,
                "partition": "difficult_scene",
            },
            # CW-A23GB-A100-L night 标识牌
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_day/20220620-181555/data",
                "sample_weight": 1,
                "partition": "difficult_scene",
            },
            # CCO2 night 6661
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_33333_6661_CAMERA_CO_OX3GB_A100-L_TIME_NIGHT_night/20220527-145352/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # CCO2 night 9964
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_32658_9964_CAMERA_CO_OX3GB_A100-L_TIME_NIGHT_night/20220527-145938/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # CCO2 night 12284
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_34231_12284_CAMERA_CO_OX3GB_A100-L_TIME_NIGHT_night/20220527-150019/data",
                "sample_weight": 2,
                "partition": "base",
            },
            # CW-A23GB-A100-L 大车 白天 35940
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_BIG_CAR_DAY_day/20220621-202410/data",
                "sample_weight": 4,
                "partition": "special_vehicle",
            },
            # CO_OX3GB_A100_L 大车 白天 3219
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/pilot_vehicle_5v_CAMERA_CO_OX3GB_A100_L_TIME_BIG_CAR_DAY_day/20220621-192421/data",
                "sample_weight": 1,
                "partition": "special_vehicle",
            },
            # CW-A23GB-A100-L 大车 夜晚 11187
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_BIG_CAR_night/20220629-121819/data",
                "sample_weight": 16,
                "partition": "special_vehicle",
            },
            # CW-A23GB-A100-L 大车 白天 8902
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A114_L_day_bigcar/20220704-185847/data",
                "sample_weight": 1,
                "partition": "special_vehicle",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_maxfa_5v_CP-OX3GB-D100-L_train_night_badcase_dark_night_Cam_5V/20220709-175425/data",
                "sample_weight": 1,
                "partition": "difficult_scene",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_badcase_dark_night_Cam_5V/20220709-173429/data",
                "sample_weight": 3,
                "partition": "difficult_scene",
            },
            # CO-OX3GB-A100-L 大车 夜晚 2416
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CO-OX3GB-A100-L_TIME_BIG_CAR_night/20220725-143019/data",
                "sample_weight": 2,
                "partition": "special_vehicle",
            },
            # CW_A23GB_A114_L 常规 白天 10842
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A114_L_day_normal/20220706-134742/data",
                "sample_weight": 2,
                "partition": "base",
            },
            # CW_A23GB_A114_L 低照 夜间 2243
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A114_L_night_dark_0736/20220726-165515/data",
                "sample_weight": 1,
                "partition": "difficult_scene",
            },
            # CW-A23GB-A100-L 大车 夜晚 3751
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A114_L_night_bigcar_0715/20220715-134225/data",
                "sample_weight": 1,
                "partition": "special_vehicle",
            },
            # 各LTC小车挡大车
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_day_overlap_vehicle_Cam_5V/20220826-161108/data",
                "sample_weight": 0.5,
                "partition": "special_vehicle",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_day_overlap_vehicle_Cam_5V/20220826-163619/data",
                "sample_weight": 0.5,
                "partition": "special_vehicle",
            },
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_pilot_pdt_5v_CW-A23GB-A100-L_train_day_overlap_vehicle_Cam_5V/20220826-163951/data",
            #     "sample_weight": 0,
            # },
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_maxfa_5v_CP-OX3GB-D100-L_train_day_overlap_vehicle_Cam_5V/20220826-163258/data",
            #     "sample_weight": 0,
            # },
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_cc02_5v_CO-OX3GB-A100-L_train_day_overlap_vehicle_Cam_5V/20220826-162631/data",
            #     "sample_weight": 0,
            # },
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_maxfa_5v_CP-OX3GB-D100-L_train_night_overlap_vehicle_Cam_5V/20220721-123722/data",
            #     "sample_weight": 0,
            # },
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_overlap_vehicle_Cam_5V/20220721-124308/data",
            #     "sample_weight": 0,
            # },
            # {
            #     "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_cc02_5v_CO-OX3GB-A100-L_train_night_overlap_vehicle_Cam_5V/20220721-124751/data",
            #     "sample_weight": 0,
            # },
            # maxfa day special car 10259
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_54665_10259_CAMERA_CP_OX3GB_D100_L_TIME_DAY_day/20220707-160146/data",
                "sample_weight": 2,
                "partition": "special_vehicle",
            },
            # as33 night special car 7858
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_73388_73389_73390_7858_CAMERA_CW_A23GB_A114_L_night/20220726-181251/data",
                "sample_weight": 2,
                "partition": "special_vehicle",
            },
            # AS33 night glare 9091
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_64258_64255_65416_9091_CAMERA_CW_A23GB_A114_L_TIME_NIGHT_night/20220707-143748/data",
                "sample_weight": 2,
                "partition": "difficult_scene",
            },
            # 全量数据快递车 7w
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_DAY_day/20220711-121816/data",
                "sample_weight": 1,
                "partition": "special_vehicle",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_DAY_day/20220711-121723/data",
                "sample_weight": 1,
                "partition": "special_vehicle",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_DAY_day/20220711-134219/data",
                "sample_weight": 4,
                "partition": "special_vehicle",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_DAY_day/20220711-160123/data",
                "sample_weight": 4,
                "partition": "special_vehicle",
            },
            # mono 三轮车2M
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_1v_OV10652_TIME_DAY_day/20220713-053741/data",
                "sample_weight": 7,
                "partition": "special_vehicle",
            },
            # mono 三轮车8M
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_1v_OV10652_TIME_DAY_day/20220713-061904/data",
                "sample_weight": 11,
                "partition": "special_vehicle",
            },
            # CW-A23GB-A100-L 标识牌 夜晚 2248
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_DARK_NIGHT_night/20220731-090659/data",
                "sample_weight": 1,
                "partition": "difficult_scene",
            },
            # 环境误检 AS33 白天 1912
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_AS33_SCENE-FP_day/20221115-182150/data",
                "sample_weight": 4,
                "partition": "difficult_scene",
            },
            # 环境误检 AS33 夜晚 2973
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_AS33_SCENE-FP_night/20221115-182152/data",
                "sample_weight": 4,
                "partition": "difficult_scene",
            },
            # c385 夜晚 6881
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_Cam_5V/20221012-114409/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # c385 白天 11114
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221012-122031/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # c385 JIRA-badcase 700
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CO-OX3GB-D100-L_JIRA-badcase_all/20221230-150000",
                "sample_weight": 5,
                "partition": "jira",
            },
            # AS33夜晚眩光jiracase 528
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_glare_Cam_5V/20221009-193824/data",
                "sample_weight": 1,
                "partition": "jira",
            },
            # GALAXY 环境误检 白天 1512
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_day/20221115-190526/data",
                "sample_weight": 2,
                "partition": "difficult_scene",
            },
            # GALAXY 环境误检 夜晚 682
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_night/20221115-190527/data",
                "sample_weight": 2,
                "partition": "difficult_scene",
            },
            # AS33 jira-badcase 白天/夜晚 369
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_AS33_JIRA-badcase_all/20221110-192849/data",
                "sample_weight": 1,
                "partition": "jira",
            },
            # c385 夜晚眩光 373
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_glare_Cam_5V/20221103-204258/data",
                "sample_weight": 2,
                "partition": "difficult_scene",
            },
            # as33 夜晚眩光 夜晚异型车 共871
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_special_vehicle_Cam_5V/20221103-204024/data",
                "sample_weight": 2,
                "partition": "special_vehicle",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_glare_Cam_5V/20221103-205303/data",
                "sample_weight": 2,
                "partition": "difficult_scene",
            },
            # as33 眩光 4528
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_glare_Cam_5V/20221114-194733/data",
                "sample_weight": 1,
                "partition": "difficult_scene",
            },
            # as33 眩光trigger-score 2272
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_AS33_night-glareV4_all/20221205-170615/data",
                "sample_weight": 1,
                "partition": "difficult_scene",
            },
            # GALAXY 环境误检 白天 531
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_day/20221230-151426",
                "sample_weight": 2,
                "partition": "difficult_scene",
            },
            # GALAXY 环境误检 夜晚 692
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_night/20221230-161307",
                "sample_weight": 2,
                "partition": "difficult_scene",
            },
            # galaxy 环境误检 夜晚 1201
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_night/20230221-104721",
                "sample_weight": 1,
                "partition": "difficult_scene",
            },
            # AS33 夜晚眩光 8159
            {
                "data_path": f"{root}/multicam_pilot/data/train/vehicle_detection/project_as33_5v_CW-A23GB-A114-L_train_night_glare_Cam_5V/20230317_202437",
                "sample_weight": 1,
                "partition": "difficult_scene",
            },
            # EV52 环境误检 白天 763
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_EV52_SCENE-FP_day/20230413-153012",
                "sample_weight": 0.5,
                "partition": "difficult_scene",
            },
            # EV52 环境误检 夜晚 767
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_EV52_SCENE-FP_night/20230413-161119",
                "sample_weight": 0.5,
                "partition": "difficult_scene",
            },
        ]
    },
    "vehicle_truncation_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_truncation_classification/pilot3.0_0233_data_before_0406_vehicle_truncation_cls/data",
                "sample_weight": 32,
            },
        ]
    },
    "semantic_parsing": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/data/default_parsing/train_data/mono3.0/16.2c/data_all_0220model_2020-06-22/parsing-720p10635-train-anno16.2_59070",
                "sample_weight": 7,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/data/default_parsing/train_data/mono3.0/16.2c/train-1080p-id20220324_62910",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_2872_20210804_anno16.2",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_950_20220527_anno16.2_day",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_2912_20220111_anno16.2",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_4876_20220415_anno16.2",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_side_roundview_num_15836_20220111_anno16.2_sy_day",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_rear_roundview_num_4671_20220111_anno16.2_sy_day",
                "sample_weight": 2.5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_side_roundview_num_13566_20220111_anno16.2_ws_day",
                "sample_weight": 8.5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_rear_roundview_num_3457_20220111_anno16.2_ws_day",
                "sample_weight": 2.5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_side_roundview_num_16089_20220111_anno16.2_ws_night",
                "sample_weight": 13,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_rear_roundview_num_3504_20220111_anno16.2_ws_night",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4987_20220111_anno16.2_x3c_night",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4300_20220214_anno16.2_x3c_night",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4060_20220428_anno16.2_x3c_night",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4500_20220111_anno16.2_x3c_day",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_10228_20220214_anno16.2_x3c_day",
                "sample_weight": 9,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_1010_20220428_anno16.2_x3c_day",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_2671_20220513_anno16.2_cc02_day",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_1202_20220602_anno16.2_as33_day",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_1660_20220825_anno16.2_as33_day/20220825-140857/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_4247_anno16.2_as33_day/data",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_1358_anno16.2_c385_night_91711/20221011-011708/data",
                "sample_weight": 4,
            },
        ]
    },
    "lane_parsing": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_18772_20210419_5_2classes",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_8571_20210524_5_2classes",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_24305_20210711_5_2classes",
                "sample_weight": 12,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_num_17311_20210711_5_2classes",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_28019_20210803_5_2classes",
                "sample_weight": 12,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_num_21736_20210803_5_2classes",
                "sample_weight": 12,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_71524_20210825_5_2classes",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_num_13711_20210825_5_2classes",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_55922_20211018_5_2classes",
                "sample_weight": 15,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_num_36387_20211018_5_2classes",
                "sample_weight": 15,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_x3c_num_160_20211129_5_2classes",
                "sample_weight": 0.1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_wissen_num_78742_20211128_5_2classes",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_x3c_num_22101_20211129_5_2classes",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_wissen_num_4549_20211129_5_2classes",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_x3c_num_47972_20211220_5_2classes",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_x3c_num_16073_20211220_5_2classes",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_x3c_num_64275_20220118_5_2classes",
                "sample_weight": 30,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_x3c_num_27923_20220118_5_2classes",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_x3c_num_14440_20220214_5_2classes",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_x3c_num_2924_20220214_5_2classes",
                "sample_weight": 0.4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_day_x3c_num_24851_20220328_5_2classes",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_night_x3c_num_5356_20220328_5_2classes",
                "sample_weight": 0.4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_day_x3c_D100L_num_17734_20220424_5_2classes",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_day_x3c_CO_OX3GB_A100L_num_7589_20220505_5_2classes",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_night_x3c_CO_OX3GB_A100L_num_745_20220505_5_2classes",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_0233_CW_A23GB_A114L_num_4481_20220525_5_2classes",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_0233_CW_A23GB_A114L_num_5460_20220525_5_2classes",
                "sample_weight": 5.0,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_0233_CW_A23GB_A114L_num_6271_20220525_5_2classes",
                "sample_weight": 3.1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_0233_CW_A23GB_A114L_num_6551_20220608_5_2classes",
                "sample_weight": 3.3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_0233_CW_A23GB_A114L_num_1995_20220621_5_2classes",
                "sample_weight": 2.0,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_220829/lane_parsing_train_day_badcase_as33/20220727-223306/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_221141/lane_parsing_train_day_badcase_as33/20220727-222539/data",
                "sample_weight": 0.4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_221458/lane_parsing_train_day_badcase_as33/20220727-222511/data",
                "sample_weight": 0.2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_221812/lane_parsing_train_day_badcase_as33/20220727-222648/data",
                "sample_weight": 0.1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_222305/lane_parsing_train_day_special_as33/20220727-223226/data",
                "sample_weight": 0.1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_222623/lane_parsing_train_day_special_as33/20220727-223356/data",
                "sample_weight": 0.06,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_222930/lane_parsing_train_day_special_as33/20220727-223812/data",
                "sample_weight": 0.1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_223832/lane_parsing_train_day_normal_as33/20220727-224554/data",
                "sample_weight": 0.09,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_224147/lane_parsing_train_day_normal_as33/20220727-224901/data",
                "sample_weight": 0.06,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220727_224502/lane_parsing_train_day_normal_as33/20220727-225403/data",
                "sample_weight": 0.2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221012_000716/pilot_lane_parsing_annoset/20221012-010546/data",
                "sample_weight": 1.0,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221012_005334/pilot_lane_parsing_annoset/20221012-012546/data",
                "sample_weight": 1.0,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221012_012136/pilot_lane_parsing_annoset/20221012-015617/data",
                "sample_weight": 2.0,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221012_014836/pilot_lane_parsing_annoset/20221012-022211/data",
                "sample_weight": 2.0,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221126_123750/pilot_lane_parsing_annoset/20221126-132031/data",
                "sample_weight": 1.0,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221126_125221/pilot_lane_parsing_annoset/20221126-132842/data",
                "sample_weight": 1.0,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221126_130644/pilot_lane_parsing_annoset/20221126-133734/data",
                "sample_weight": 2.0,
            },
        ]
    },
    "vehicle_wheel_kps": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_kps/vehicle_2_kps/vehicle_2_kps_all_20201021.train",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_kps_detection/release_rec/vehicle_2_kps_add0323cutin_20201125",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_day_0233/20220119-212808/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_night_0233/20220120-114207/data",
                "sample_weight": 40,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_day_X3C/20220119-122143/data",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_night_X3C/20220119-123010/data",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_day_X3C/20220224-131626/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_CAMERA_X3C_day/20220315-163131/data",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_wheel_kps/pilot_vehicle_kps2_5v_CAMERA_X3C_day/20220425-161851/data",
                "sample_weight": 1,
            },
        ]
    },
    "vehicle_ground_line": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_day_0233/20220119-221935/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_night_0233/20220120-115549/data",
                "sample_weight": 40,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_day_X3C/20220119-125418/data",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_night_X3C/20220119-141510/data",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_day_X3C/20220224-133404/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_CAMERA_X3C_day/20220315-163208/data",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_ground_line/pilot_vehicle_flank_5v_CAMERA_X3C_day/20220425-161825/data",
                "sample_weight": 1,
            },
        ]
    },
    "vehicle_flank": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehFlank/kVehFlank_Cam_5V/20220706-213158/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehFlank/kVehFlank_ar0233-RGGB_train_Cam_5V/20220706-205154/data",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehFlank/kVehFlank_ar0233-WISSEN_train_Cam_5V/20220706-165341/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehFlank/kVehFlank_ar0233-WISSEN_train_day_big_vehicle_Cam_5V/20220706-123556/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehFlank/kVehFlank_x3c_train_day_Cam_5V/20220706-112001/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehFlank/kVehFlank_project_cc02_5v_CO-OX3GB-A100-L_train_day_big_vehicle_Cam_5V/20220706-102652/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehFlank/kVehFlank_x3c_train_night_Cam_5V/20220706-163624/data",
                "sample_weight": 6,
            },
        ]
    },
    "vehicle_wheel_detection": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_wheel_detection/det_subbox_vehicle_wheel_belongto_clean-veh_wheel-20210407_train_clean_323_all_trans_pb_rec/train",
                "sample_weight": 171,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_wheel_detection/MCP_wheel_detection_day_0927/train",
                "sample_weight": 85,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehWheelBBox2D/kVehWheelBBox2D_train_day_big_vehicle_Cam_5V_Cam_5V/20220617-193224/data",
                "sample_weight": 2,
            },
        ]
    },
    "rear": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v3-20190726-train_base/data",
                "sample_weight": 8,
                "partition": "base",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v3-20190726-train_gray/data",
                "sample_weight": 6,
                "partition": "base",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v3-20190726-train_night/data",
                "sample_weight": 3,
                "partition": "base",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v6-20200624-train_390_5200_day_FrontRear/data",
                "sample_weight": 2,
                "partition": "base",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v6-20200624-train_390_5200_day_NonFrontRear/data",
                "sample_weight": 3,
                "partition": "base",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v6-20200624-train_390_5200_night/data",
                "sample_weight": 2,
                "partition": "base",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/mono_0323_data/user/ni.jiang/data/vehicle_rear_data/det_vehicle_rear_v8-20201119-train_part_4_0323_0220_exclude_20200601_exclude_part33/data",
                "sample_weight": 12,
                "partition": "base",
            },
            # 0233 白天 330660
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_day_0233/20220119-012241/data",
                "sample_weight": 10,
                "partition": "base",
            },
            # 0233 夜晚 81783
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_night_0233/20220118-233709/data",
                "sample_weight": 40,
                "partition": "base",
            },
            # X3C 白天 84312
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_day_X3C/20220119-111048/data",
                "sample_weight": 5,
                "partition": "base",
            },
            # X3C 夜晚 24160
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_night_X3C/20220119-105348/data",
                "sample_weight": 8,
                "partition": "base",
            },
            # X3C 白天 12906
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_day_X3C/20220224-113552/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # X3C 夜晚 22000
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_night_X3C/20220224-115939/data",
                "sample_weight": 8,
                "partition": "base",
            },
            # X3C 白天 15688
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CAMERA_X3C_day/20220315-163248/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # X3C 夜晚 7612
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CAMERA_X3C_night/20220315-164230/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # X3C 白天 13505
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CAMERA_X3C_day/20220401-140046/data",
                "sample_weight": 2,
                "partition": "base",
            },
            # PROJECT_CC02_5V 白天 4910
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_PROJECT_CC02_5V_day/20220407-195106/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # PROJECT_CC02_5V 夜晚 4758
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_PROJECT_CC02_5V_night/20220407-195117/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # CAMERA_CW_A23GB_A100_L 夜晚 9739 annoset(12296)
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CAMERA_CW_A23GB_A100_L_night/20220425-140822/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # PROJECT_AS33_5V 夜晚 8872
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_AS33_night/20220412-213815/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # PROJECT_PDT 夜晚 大车 1306
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CW_A23GB_A100_L_big_car_night/20220412-212057/data",
                "sample_weight": 1,
                "partition": "badcase",
            },
            # PROJECT_CC02_5V 白天 8526 annoset(16130)
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_PROJECT_CC02_5V_day/20220510-214104/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # PROJECT_CC02_5V 夜晚 2159 annoset(16131)
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_PROJECT_CC02_5V_night/20220510-213158/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # PROJECT_AS33_5V 白天 2113
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CW-A23GB-A114-L_day/20220510-103100/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # PROJECT_AS33_5V 夜晚 1014
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CW-A23GB-A114-L_night/20220510-115328/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # CAMERA_X3C_5V 夜晚 1.9K
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/pilot_rear_5v_CAMERA_X3C_night/20220621-162154/data",
                "sample_weight": 1,
                "partition": "base",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear/det_vehicle_rear_v10-20211229-10652-split_special_and_BigTrucks/data",
                "sample_weight": 10,
                "partition": "badcase",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear/det_vehicle_rear_v10-20220129-jira_and_close_vehicle_data/data",
                "sample_weight": 10,
                "partition": "badcase",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear/det_vehicle_rear_v8-0220_0323_10635_bigcar-20210917/data",
                "sample_weight": 10,
                "partition": "badcase",
            },
            # CW_A23GB_A114_L 夜晚 眩光 2015
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_rear/pilot_rear_5v_CAMERA_CW_A23GB_A114_L_night_glare_0831/20220831-181141/data",
                "sample_weight": 1,
                "partition": "badcase",
            },
            # PROJECT_c385_5V 白天 4879
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_rear/pilot_rear_5v_CO-OX3GB-D100-L_day/20221127-140301/data",
                "sample_weight": 1,
                "partition": "base",
            },
        ]
    },
    "rear_occlusion_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_data/vehicle_rear_occlusion_cls_data/cls_subbox_rear_occlusion_v8-20190719-train_base_night_gray-cn93-cn750-model-score0.18/data",
                "sample_weight": 12,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_data/vehicle_rear_occlusion_cls_data/cls_subbox_rear_occlusion_v8-20200903-train_part_4_0323_ispDefault_2280x1080_exclude_part33-cn750-cn759/data",
                "sample_weight": 18,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_data/vehicle_rear_occlusion_cls_data/cls_rear_neg_v4-20200901-fp_train/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_day_0233/20220119-041133/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_night_0233/20220119-020318/data",
                "sample_weight": 40,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_day_X3C/20220119-120507/data",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_night_X3C/20220119-113231/data",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_day_X3C/20220224-123226/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_night_X3C/20220224-125801/data",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_CAMERA_X3C_day/20220316-102102/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_CAMERA_X3C_night/20220315-162936/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_occlusion_classification/pilot_rear_occ_5v_CAMERA_X3C_day/20220401-160904/data",
                "sample_weight": 2,
            },
        ]
    },
    "rear_part_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_data/vehicle_rear_cls_data/cls_vehicle_part_3cls_v1-old/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_data/vehicle_rear_cls_data/cls_vehicle_part_3cls_v1-hard/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_data/vehicle_rear_cls_data/cls_vehicle_part_3cls_v1-res/data",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_data/vehicle_rear_cls_data/cls_vehicle_part_3cls_v1-0323/data",
                "sample_weight": 9,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_day_0233/20220119-210054/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_night_0233/20220119-203134/data",
                "sample_weight": 40,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_day_X3C/20220119-120509/data",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_night_X3C/20220119-115855/data",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_day_X3C/20220224-130525/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_CAMERA_X3C_day/20220315-163229/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_CAMERA_X3C_night/20220315-163054/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_part_classification/pilot_rear_part_5v_CAMERA_X3C_night/20220425-142431/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_rear_part_classification/pilot_rear_part_5v_CAMERA_CW_A23GB_A114_L_day_bigcar/20220822-093413/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_rear_part_classification/pilot_rear_part_5v_CO-OX3GB-D100-L_day/20221127-145644/data",
                "sample_weight": 1,
            },
        ]
    },
}

# galaxy base 2d data
galaxy_datapaths = {
    "rear_plate_detection": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/data/pilot/plate/train_220319/0233/train",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/data/pilot/plate/train_220319/x3c/train",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/data/pilot/plate/train_220319/unknow/train",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_rear_plate_5v_CW-A23GB-A114-L_day_2pe/20220504-102122/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_rear_plate_5v_CO-OX3GB-A100-L_night_2pe/20220504-103634/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_rear_plate_5v_CW-A23GB-A100-L_night_2pe/20220504-103906/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_rear_plate_5v_CO-OX3GB-A100-L_day_2pe/20220504-103937/data",
                "sample_weight": 16,
            },
        ]
    },
    "person_face_detection": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_all_all_2pe/20220720-214115/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_all_all_2pe/20220720-205236/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_c385-5v_all_2pe/20220720-174226/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_c385-5v_all_2pe/20220720-174425/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_as33-5v_all_2pe/20220720-174221/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_cc02-5v_all_2pe/20220720-174714/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_maxfa-5v_all_2pe/20220720-180427/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_c385-5v_all_2pe/20220721-202711/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_as33-5v_all_2pe/20220809-212818/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_cc02-5v_all_2pe_2022-07-20_2022-08-09/20220810-114649/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_maxfa-5v_all_2pe_2022-07-20_2022-08-09/20220810-114506/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/users/yuanzhuo.peng/tmp/pilot_person_face_5v_pilot-pdt-5v_all_2pe_2022-07-20_2022-08-09/20220810-115032/data",
                "sample_weight": 16,
            },
        ]
    },
    "person_occlusion_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/person_occlusion_classification/cls_person_occlusion_v1-20190810-train/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedOcclusion/kPedOcclusion_project_galaxy_5v_train_day_Cam_Rear/20220622-171219/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedOcclusion/kPedOcclusion_project_galaxy_5v_train_night_Cam_Side/20220622-171224/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedOcclusion/kPedOcclusion_project_galaxy_5v_train_day_Cam_Side/20220622-171019/data",
                "sample_weight": 1,
            },
        ]
    },
    "vehicle_occlusion_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_cls_data/cls_subbox_vehicle_full_occlusion_v2-20200825-train_old/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_cls_data/cls_subbox_vehicle_full_occlusion_v2-20201117-vehicle_full_0323/data",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_cls_data/cls_subbox_vehicle_full_occlusion_v2-20200621-vehicle_full_0220/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_cls_data/cls_vehicle_full_fp_v3-train_other/data",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_cls_data/cls_vehicle_full_fp-vehicle_full_fp-data_3900/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_full_cls_data/cls_vehicle_full_fp_v3-20201120-train_0323/data",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_occ_classification/pilot3.0_0233_data_before_0406_vehicle_occ_cls/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehOcclusion/kVehOcclusion_project_galaxy_5v_train_day_Cam_Rear/20220622-171638/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehOcclusion/kVehOcclusion_project_galaxy_5v_train_night_Cam_Side/20220622-171803/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehOcclusion/kVehOcclusion_project_galaxy_5v_train_day_Cam_Side/20220622-171251/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehOcclusion/pilot_vehicle_5v_CAMERA_GALAXY_JIRA-badcase_all/20230220-191850",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehOcclusion/pilot_vehicle_5v_CAMERA_GALAXY_normal_day/20230221-144555",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehOcclusion/pilot_vehicle_5v_CAMERA_GALAXY_normal_night/20230221-183002",
                "sample_weight": 15,
            },
        ]
    },
    "cyclist_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/Pilot_cyclist_category_5v_CAMERA_CW_A23GB_A114_L_DAY_day/20220926-105914/data",
                "sample_weight": 7,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/Pilot_cyclist_category_5v_CAMERA_CW_A23GB_A114_L_NIGHT_night/20220926-122940/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/Pilot_cyclist_category_5v_CAMERA_CW_A23GB_A114_L_DAY_day/20221012-135628/data",
                "sample_weight": 13,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/Pilot_cyclist_category_5v_CAMERA_CW_A23GB_A114_L_NIGHT_night/20221012-174742/data",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/Pilot_cyclist_category_5v_CAMERA_CO-OX3GB-D100-L_DAY_day/20221012-185013/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/Pilot_cyclist_category_5v_CAMERA_CO-OX3GB-D100-L_NIGHT_night/20221012-195248/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/Pilot_cyclist_category_5v_CAMERA_CP-OX3GB-D100-L_DAY_day/20221012-211105/data",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/Pilot_cyclist_category_5v_CAMERA_CP-OX3GB-D100-L_NIGHT_night/20221013-012831/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/Pilot_cyclist_category_5v_CAMERA_CO-OX3GB-A100-L_DAY_day/20221013-025417/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/Pilot_cyclist_category_5v_CAMERA_CO-OX3GB-A100-L_NIGHT_night/20221013-022108/data",
                "sample_weight": 3,
            },
        ]
    },
    "cyclist_kps": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/cyclist_kps/gluon_rec_width12/lf_new/data.train",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/2pe_cyclist_kps/cyc_kps_zhengwei/20220121-132406/data",
                "sample_weight": 4,
            },
        ]
    },
    "person_pose_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/person_pose_classification/cls_person_pose_v2_upsample-20190810-train/data",
                "sample_weight": 32,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/2pe_person_pose_classification/example_pack_densebox_ped_pose_data_20220121_122405_0233_all/20220121-140616/data",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedPose/kPedPose_project_galaxy_5v_train_day_Cam_Rear/20220622-171352/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedPose/kPedPose_project_galaxy_5v_train_night_Cam_Side/20220622-171245/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedPose/kPedPose_project_galaxy_5v_train_day_Cam_Side/20220622-171024/data",
                "sample_weight": 1,
            },
        ]
    },
    "vehicle_category": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_category_classification/pilot3.0_vehicle_8cls_senyun_day/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_category_classification/pilot3.0_vehicle_8cls_weissen_day/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehCategory/kVehCategory_project_galaxy_5v_train_day_Cam_Rear/20220622-171442/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehCategory/kVehCategory_project_galaxy_5v_train_night_Cam_Side/20220622-171219/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehCategory/kVehCategory_project_galaxy_5v_train_day_Cam_Side/20220622-171417/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehCategory/kVehCategory_project_galaxy_5v_train_day_Cam_Side/20221209-173155/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehCategory/kVehCategory_project_galaxy_5v_train_night_Cam_Side/20221209-172748/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehCategory/pilot_vehicle_5v_CAMERA_GALAXY_JIRA-badcase_all/20230220-190609",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehCategory/pilot_vehicle_5v_CAMERA_GALAXY_normal_day/20230221-144554",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehCategory/pilot_vehicle_5v_CAMERA_GALAXY_normal_night/20230221-182952",
                "sample_weight": 15,
            },
        ]
    },
    "person_orientation_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/person_orientation_classification/cls_person_orientation_v11_upsample-20191204_1-train/data",
                "sample_weight": 32,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedOrientation/kPedOrientation_project_galaxy_5v_train_night_Cam_Side/20220622-171155/data",
                "sample_weight": 1,
            },
        ]
    },
    "cyclist": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220622-145223/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220622-145424/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Side/20220622-150123/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_night_Cam_Side/20220622-145741/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220719-115017/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Side/20220719-102656/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220806-002734/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Side/20220805-233757/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Side/kCycBBox2D_project_galaxy_5v_train_day_Cam_Side/20220928-170533/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_night_Cam_Side/kCycBBox2D_project_galaxy_5v_train_night_Cam_Side/20220928-155907/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Rear/kCycBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220928-172750/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_night_Cam_Rear/kCycBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220928-160112/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Side/20221028-222207/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Rear/20221028-235854/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_night_Cam_Side/20221028-231600/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_night_Cam_Rear/20221028-232324/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/jianhang.he.data/Cutin_exp/debug/Pilot_cyclist_5v_all_night_cutin/20220928-194559/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/jianhang.he.data/Cutin_exp/debug/Pilot_cyclist_5v_all_night_cutin/20220928-200950/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_night_JIRA_badcase_Cam_All/20221028-230453/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_all_Cam_Rear/20221119-132909/data",
                "sample_weight": 0.25,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_all_Cam_Side/20221119-134819/data",
                "sample_weight": 0.25,
            },
            # galaxy jira day annoset(72636) 130
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_JIRA_badcase_Cam_All/20221119-142707/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_night_JIRA_badcase_Cam_All/20221119-143020/data",
                "sample_weight": 1.5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_all_Cam_Side/20221212-133412/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_all_Cam_Rear/20221212-134154/data",
                "sample_weight": 1,
            },
            # galaxy glare jira annoset(88777) 3863
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_CO-OX3GB-D100-L_train_night_jira_Cam_5V/20230128-143617/data",
                "sample_weight": 4,
            },
            # galaxy night rear; glare; fp; hand select; annoset(89129) 2329
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_train_night_Cam_Rear/20230130-211816/data",
                "sample_weight": 3,
            },
            # galaxy night rear; glare; no prediction; hand select; annoset(89149) 7893
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_train_night_Cam_Rear/20230130-221935/data",
                "sample_weight": 1,
            },
            # galaxy cyclist fn; jira; annoset(100351) 300
            {
                "data_path": f"{root}/multicam_pilot/data/train/cyclist_detection/kCycBBox2D_project_galaxy_5v_train_day_JIRA_FN_Cam_All/20230324-142145",
                "sample_weight": 1,
            },
            # galaxy x03 FP jira, 342
            {
                "data_path": f"{root}/multicam_pilot/data/train/cyclist_detection/kCycBBox2D_project_galaxy_5v_train_day_Badcase_FP_Cam_Side/20230523-135119",
                "sample_weight": 1,
            },
            # galaxy v17 jira
            {
                "data_path": f"{root}/multicam_pilot/data/train/cyclist_detection/kCycBBox2D_project_galaxy_5v_train_all_Jira_Cam_All/20230808-135242",
                "sample_weight": 1,
            },
            # galaxy v18 Jira
            {
                "data_path": f"{root}/multicam_pilot/data/train/cyclist_detection/kCycBBox2D_project_galaxy_5v_train_all_Jira_Cam_All/20230926-165636",
                "sample_weight": 2,
            },
        ]
    },
    "person": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220622-145832/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220622-145002/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Side/20220622-150324/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_night_Cam_Side/20220622-150103/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220719-114353/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Side/20220719-102053/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Side/20220805-235334/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220805-234732/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Side/kPedBBox2D_project_galaxy_5v_train_day_Cam_Side/20220928-171508/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_night_Cam_Side/kPedBBox2D_project_galaxy_5v_train_night_Cam_Side/20220928-155340/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Rear/kPedBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220928-170750/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_night_Cam_Rear/kPedBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220928-155540/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Side/20221028-213019/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Rear/20221028-215546/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_night_Cam_Side/20221028-225802/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_night_Cam_Rear/20221028-230726/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_all_Cam_Rear/20221119-132747/data",
                "sample_weight": 0.25,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_all_Cam_Side/20221119-134707/data",
                "sample_weight": 0.25,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_JIRA_badcase_Cam_All/20230101-212215/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_night_JIRA_badcase_Cam_All/20221119-142917/data",
                "sample_weight": 1.5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_all_Cam_Side/20221212-132212/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_all_Cam_Rear/20221212-132630/data",
                "sample_weight": 1,
            },
            # galaxy glare jira annoset(88776) 3863
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_CO-OX3GB-D100-L_train_night_jira_Cam_5V/20230128-143411/data",
                "sample_weight": 4,
            },
            # galaxy night rear; glare; fp; hand select; annoset(89125) 2329
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_train_night_Cam_Rear/20230130-211620/data",
                "sample_weight": 6,
            },
            # galaxy night rear; glare; no prediction; hand select; annoset(89148) 7893
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_train_night_Cam_Rear/20230130-221528/data",
                "sample_weight": 1,
            },
            # galaxy x03 hook FP jira, 200
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_JIRA_badcase_Cam_Rear/20230421-193911",
                "sample_weight": 1,
            },
            # galaxy x03 FP jira, 342
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/kPedBBox2D_project_galaxy_5v_train_day_Badcase_FP_Cam_Side/20230523-153107",
                "sample_weight": 1,
            },
            # galaxy x03 v16 Pillar FP Side day
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/kPedBBox2D_project_galaxy_5v_train_day_Pillar_FP_Cam_Side/20230630-181048",
                "sample_weight": 2,
            },
            # galaxy x03 v16 Pillar FP Rear day
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/kPedBBox2D_project_galaxy_5v_train_day_Pillar_FP_Cam_Rear/20230630-182030",
                "sample_weight": 2,
            },
            # galaxy x03 v16 Vehicle FP Side day
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/kPedBBox2D_project_galaxy_5v_train_day_Vehicle_FP_Cam_Side/20230630-182318",
                "sample_weight": 1.5,
            },
            # galaxy x03 v16 Vehicle FP Rear day
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/kPedBBox2D_project_galaxy_5v_train_day_Vehicle_FP_Cam_Rear/20230630-182415",
                "sample_weight": 0.5,
            },
            # galaxy x03 v16 Vehicle FP Side night
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/kPedBBox2D_project_galaxy_5v_train_night_Vehicle_FP_Cam_Side/20230630-183659",
                "sample_weight": 1.5,
            },
            # galaxy x03 v16 Vehicle FP Rear night
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/kPedBBox2D_project_galaxy_5v_train_night_Vehicle_FP_Cam_Rear/20230630-183736",
                "sample_weight": 0.5,
            },
            # galaxy v17 jira
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/kPedBBox2D_project_galaxy_5v_train_all_Jira_Cam_All/20230808-135200",
                "sample_weight": 0.5,
            },
            # galaxy v18 Pillar fp
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/kPedBBox2D_project_galaxy_5v_train_day_Pillar_Cam_Rear/20230926-181033",
                "sample_weight": 0.5,
            },
            # galaxy v18 Pillar fp
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/kPedBBox2D_project_galaxy_5v_train_night_Pillar_Cam_Rear/20230926-180702",
                "sample_weight": 2,
            },
            # galaxy v18 Pillar fp
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/kPedBBox2D_project_galaxy_5v_train_night_Pillar_Cam_Side/20230926-172905",
                "sample_weight": 2,
            },
            # galaxy v18 Pillar fp
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/kPedBBox2D_project_galaxy_5v_train_day_Pillar_Cam_Side/20230926-171938",
                "sample_weight": 1,
            },
            # galaxy v18 vehicle fp
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/kPedBBox2D_project_galaxy_5v_train_day_Vehicle_Cam_Rear/20230926-180419",
                "sample_weight": 0.5,
            },
            # galaxy v18 vehicle fp
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/kPedBBox2D_project_galaxy_5v_train_night_Vehicle_Cam_Rear/20230926-180102",
                "sample_weight": 0.5,
            },
            # galaxy v18 vehicle fp
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/kPedBBox2D_project_galaxy_5v_train_night_Vehicle_Cam_Side/20230926-175244",
                "sample_weight": 1.5,
            },
            # galaxy v18 vehicle fp
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/kPedBBox2D_project_galaxy_5v_train_day_Vehicle_Cam_Side/20230926-174859",
                "sample_weight": 1.5,
            },
            # galaxy v18 glare
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/kPedBBox2D_project_galaxy_5v_train_night_Glare_Cam_Rear/20230926-175908",
                "sample_weight": 0.5,
            },
            # galaxy v18 glare
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/kPedBBox2D_project_galaxy_5v_train_night_Glare_Cam_Side/20230926-175558",
                "sample_weight": 0.5,
            },
            # galaxy v18 jira
            {
                "data_path": f"{root}/multicam_pilot/data/train/person_detection/kPedBBox2D_project_galaxy_5v_train_all_Jira_Cam_All/20230926-165758",
                "sample_weight": 3,
            },
        ]
    },
    "vehicle": {
        "train_data_paths": [
            # galaxy day rear annoset(25957) 5672
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220622-150758/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # galaxy night rear annoset(25954) 1473
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220622-145944/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # galaxy day side annoset(25956) 6439
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_day_Cam_Side/20220622-150945/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # galaxy night side annoset(25952) 2168
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_night_Cam_Side/20220622-145829/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # galaxy day rear annoset(39574) 13564
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220719-114309/data",
                "sample_weight": 2,
                "partition": "base",
            },
            # galaxy day side annoset(39575) 10593
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_day_Cam_Side/20220719-111957/data",
                "sample_weight": 2,
                "partition": "base",
            },
            # galaxy night side annoset(39555) 1319
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_night_Cam_Side/20220719-115156/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # galaxy night rear annoset(39540) 1345
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220719-115248/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # galaxy day side annoset(47642) 3300
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_day_Cam_Side/20220805-225859/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # galaxy day rear annoset(47644) 5276
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220805-230512/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # galaxy night rear annoset(47646) 3160
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220805-230658/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # galaxy night side annoset(47647) 2065
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_night_Cam_Side/20220805-230919/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # c385 v2.0 63351
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_350703_CAMERA_CO_OX3GB_D100_L_TIME_ALL_all/20220818-202734/data",
                "sample_weight": 6,
                "partition": "base",
            },
            # galaxy day side annoset(61056) 25547
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_day_Cam_Side/20221001-093910/data",
                "sample_weight": 3,
                "partition": "base",
            },
            # galaxy night rear annoset(61067) 6701
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_night_Cam_Rear/20221001-093514/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # galaxy day rear annoset(61065) 15397
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_day_Cam_Rear/20221001-092648/data",
                "sample_weight": 2,
                "partition": "base",
            },
            # galaxy night side annoset(61061) 10114
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_night_Cam_Side/20221001-092213/data",
                "sample_weight": 2,
                "partition": "base",
            },
            # galaxy JIRA-badcase annoset(66486) 5772
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_JIRA-badcase_all/20230607-214830",
                "sample_weight": 12,
                "partition": "jira",
            },
            # galaxy JIRA-badcase-hard 785
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_JIRA-badcase_hard_all/20230807-110620",
                "sample_weight": 6,
                "partition": "jira",
            },
            # galaxy 白天 异型车 3863
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SPECIAL_VEHICLE_day/20230616-123700",
                "sample_weight": 1,
                "partition": "special_vehicle",
            },
            # galaxy 夜晚 异型车 1842
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SPECIAL_VEHICLE_night/20230616-123154",
                "sample_weight": 1,
                "partition": "special_vehicle",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_day/20230919-183815",
                "sample_weight": 1,
                "partition": "difficult_scene",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_night/20230919-184402",
                "sample_weight": 1,
                "partition": "difficult_scene",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_glare_night/20230920-145601",
                "sample_weight": 1,
                "partition": "difficult_scene",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SPECIAL_VEHICLE_day/20230920-142653",
                "sample_weight": 0.1,
                "partition": "special_vehicle",
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SPECIAL_VEHICLE_night/20230920-143843",
                "sample_weight": 0.1,
                "partition": "special_vehicle",
            },
        ]
    },
    "vehicle_truncation_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_truncation_classification/pilot3.0_0233_data_before_0406_vehicle_truncation_cls/data",
                "sample_weight": 32,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehTruncation/kVehTruncation_project_galaxy_5v_train_day_Cam_Rear/20220622-171728/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehTruncation/kVehTruncation_project_galaxy_5v_train_day_Cam_Side/20220622-171503/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehTruncation/kVehTruncation_project_galaxy_5v_train_night_Cam_Side/20220622-171245/data",
                "sample_weight": 1,
            },
        ]
    },
    "semantic_parsing": {
        "train_data_paths": [
            # mono
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/data/default_parsing/train_data/mono3.0/16.2c/data_all_0220model_2020-06-22/parsing-720p10635-train-anno16.2_59070",
                "sample_weight": 7,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/data/default_parsing/train_data/mono3.0/16.2c/train-1080p-id20220324_62910",
                "sample_weight": 8,
            },
            # sensing day (special for large person)
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_2872_20210804_anno16.2",
                "sample_weight": 1,
            },
            # sensing day (rain)
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_950_20220527_anno16.2_day",
                "sample_weight": 1,
            },
            # wissen day&night (special for merge truck bottom)
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_2912_20220111_anno16.2",
                "sample_weight": 1,
            },
            # as33 day&night (snow)
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_4876_20220415_anno16.2",
                "sample_weight": 2,
            },
            # sensing day
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_side_roundview_num_15836_20220111_anno16.2_sy_day",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_rear_roundview_num_4671_20220111_anno16.2_sy_day",
                "sample_weight": 2.5,
            },
            # wissen day
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_side_roundview_num_13566_20220111_anno16.2_ws_day",
                "sample_weight": 8.5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_rear_roundview_num_3457_20220111_anno16.2_ws_day",
                "sample_weight": 2.5,
            },
            # wissen night
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_side_roundview_num_16089_20220111_anno16.2_ws_night",
                "sample_weight": 9,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_rear_roundview_num_3504_20220111_anno16.2_ws_night",
                "sample_weight": 2,
            },
            # maxfa night
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4987_20220111_anno16.2_x3c_night",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4300_20220214_anno16.2_x3c_night",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4060_20220428_anno16.2_x3c_night",
                "sample_weight": 4,
            },
            # maxfa day
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_4500_20220111_anno16.2_x3c_day",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_10228_20220214_anno16.2_x3c_day",
                "sample_weight": 9,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_1010_20220428_anno16.2_x3c_day",
                "sample_weight": 1,
            },
            # cc02 day
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_2671_20220513_anno16.2_cc02_day",
                "sample_weight": 4,
            },
            # as33 day
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x2048_train_pinhole_5cam_roundview_num_1202_20220602_anno16.2_as33_day",
                "sample_weight": 2,
            },
            # galaxy day
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_2138_anno16.2_galaxy/20220621-165237/data",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_3383_anno16.2_galaxy_day/20220718-231244/data",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_1601_anno16.2_galaxy_day/20220809-203210/data",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_1137_anno16.2_galaxy_day/20220909-193537_new/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_2428_anno16.2_galaxy_day/20220930-170957/data",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_2390_anno16.2_galaxy_day/20221028-174147/data",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_3693_anno16.2_galaxy_day/20221210-001652/data",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_1364_anno16.2_galaxy_day/20230126-021406/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/train/default_segmentation/pilot_galaxy_x02_driving_day_unknown_unknown_unknown_data_mining_manual_label_unknown_default_segmentation/20230522_212728",
                "sample_weight": 1,  # num 174 (x03 159, jira 15)
            },
            # galaxy night
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_496_anno16.2_galaxy/20220621-151513/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_558_anno16.2_galaxy_night/20220719-154904/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_1316_anno16.2_galaxy_night/20220909-215318/data",
                "sample_weight": 2.5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_1015_anno16.2_galaxy_night/20220930-162403/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/semantic_parsing/pilot3_1280x1920_train_pinhole_5cam_roundview_num_1207_anno16.2_galaxy_night/20221209-234006/data",
                "sample_weight": 2,
            },
        ]
    },
    "lane_parsing": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_18772_20210419_5_2classes",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_3500_20210419_5_2classes",
                "sample_weight": 1.5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_8571_20210524_5_2classes",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_3902_20210524_5_2classes",
                "sample_weight": 1.5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_24305_20210711_5_2classes",
                "sample_weight": 12,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_6291_20210711_5_2classes",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_num_1899_20210711_5_2classes",
                "sample_weight": 0.5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_num_17311_20210711_5_2classes",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_28019_20210803_5_2classes",
                "sample_weight": 12,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_1721_20210803_5_2classes",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_num_5598_20210803_5_2classes",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_num_21736_20210803_5_2classes",
                "sample_weight": 12,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_71524_20210825_5_2classes",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_20735_20210825_5_2classes",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_num_3512_20210825_5_2classes",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_num_13711_20210825_5_2classes",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_num_55922_20211018_5_2classes",
                "sample_weight": 15,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_num_13475_20211018_5_2classes",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_num_6000_20211018_5_2classes",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_num_36387_20211018_5_2classes",
                "sample_weight": 15,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_x3c_num_4387_20211129_5_2classes",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_x3c_num_160_20211129_5_2classes",
                "sample_weight": 0.1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_wissen_num_78742_20211128_5_2classes",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_wissen_num_653_20211129_5_2classes",
                "sample_weight": 0.1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_wissen_num_15689_20211129_5_2classes",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_x3c_num_22101_20211129_5_2classes",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_wissen_num_4549_20211129_5_2classes",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_x3c_num_12116_20211220_5_2classes",
                "sample_weight": 12,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_x3c_num_2802_20211220_5_2classes",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_x3c_num_47972_20211220_5_2classes",
                "sample_weight": 20,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_x3c_num_16073_20211220_5_2classes",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_x3c_num_16876_20220118_5_2classes",
                "sample_weight": 12,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_x3c_num_6284_20220118_5_2classes",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_x3c_num_64275_20220118_5_2classes",
                "sample_weight": 30,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_x3c_num_27923_20220118_5_2classes",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_x3c_num_2909_20220214_5_2classes",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_x3c_num_673_20220214_5_2classes",
                "sample_weight": 0.1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_x3c_num_14440_20220214_5_2classes",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_x3c_num_2924_20220214_5_2classes",
                "sample_weight": 0.2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_day_x3c_num_24851_20220328_5_2classes",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_night_x3c_num_5356_20220328_5_2classes",
                "sample_weight": 0.2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_rear_day_x3c_num_6785_20220328_5_2classes",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_rear_night_x3c_num_5356_20220328_5_2classes",
                "sample_weight": 0.2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_day_x3c_D100L_num_17734_20220424_5_2classes",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_rear_day_x3c_D100L_num_5286_20220424_5_2classes",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_day_x3c_CO_OX3GB_A100L_num_7589_20220505_5_2classes",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_rear_day_x3c_CO_OX3GB_A100L_num_2629_20220505_5_2classes",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_side_night_x3c_CO_OX3GB_A100L_num_745_20220505_5_2classes",
                "sample_weight": 0.5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x1920_train_pinhole_rear_night_x3c_CO_OX3GB_A100L_num_166_20220505_5_2classes",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_0233_CW_A23GB_A114L_num_4481_20220525_5_2classes",
                "sample_weight": 2.3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_0233_CW_A23GB_A114L_num_1777_20220525_5_2classes",
                "sample_weight": 1.0,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_night_0233_CW_A23GB_A114L_num_5460_20220525_5_2classes",
                "sample_weight": 2.5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_night_0233_CW_A23GB_A114L_num_2461_20220525_5_2classes",
                "sample_weight": 1.0,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_side_day_0233_CW_A23GB_A114L_num_6271_20220525_5_2classes",
                "sample_weight": 3.1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/train/parsing/lane_parsing/pilot3_1280x2048_train_pinhole_rear_day_0233_CW_A23GB_A114L_num_2193_20220525_5_2classes",
                "sample_weight": 1.1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220624_145658/pilot_lane_parsing_annoset/20220624-151527/data",
                "sample_weight": 0.6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220624_145604/pilot_lane_parsing_annoset/20220624-150814/data",
                "sample_weight": 0.5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220624_145632/pilot_lane_parsing_annoset/20220624-154111/data",
                "sample_weight": 2.2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220624_145532/pilot_lane_parsing_annoset/20220624-151249/data",
                "sample_weight": 0.5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220718_172921/pilot_lane_parsing_annoset/20220718-180136/data",
                "sample_weight": 0.7,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220718_173829/pilot_lane_parsing_annoset/20220718-175945/data",
                "sample_weight": 0.6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220718_173357/pilot_lane_parsing_annoset/20220718-175942/data",
                "sample_weight": 0.5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220718_172304/pilot_lane_parsing_annoset/20220718-182428/data",
                "sample_weight": 2.3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220808_145734/pilot_lane_parsing_annoset/20220808-152052/data",
                "sample_weight": 0.5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220808_150429/pilot_lane_parsing_annoset/20220808-151403/data",
                "sample_weight": 0.2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220915_115417/pilot_lane_parsing_annoset/20220915-121824/data",
                "sample_weight": 0.1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220915_125452/pilot_lane_parsing_annoset/20220915-134505/data",
                "sample_weight": 1.1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220915_141227/pilot_lane_parsing_annoset/20220915-143020/data",
                "sample_weight": 0.7,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20220915_140510/pilot_lane_parsing_annoset/20220915-142425/data",
                "sample_weight": 0.9,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221010_003115/pilot_lane_parsing_annoset/20221010-010602/data",
                "sample_weight": 0.45,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221009_235934/pilot_lane_parsing_annoset/20221010-004456/data",
                "sample_weight": 1.0,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221028_152259/pilot_lane_parsing_annoset/20221028-153922/data",
                "sample_weight": 0.2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/parsing/multicam_pilot/data/tmp/lane_parsing_packing_data_publish_annoset_20221009_225819/pilot_lane_parsing_annoset/20221010-002433/data",
                "sample_weight": 0.2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/lane_parsing/lane_parsing_packing_data_publish_annoset_20230220_101921/pilot_lane_parsing_annoset/20230220-110946",
                "sample_weight": 0.2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/lane_parsing/lane_parsing_packing_data_publish_annoset_20230220_105408/pilot_lane_parsing_annoset/20230220-112500",
                "sample_weight": 0.2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/lane_parsing/lane_parsing_packing_data_publish_annoset_20230220_111927/pilot_lane_parsing_annoset/20230220-113312",
                "sample_weight": 0.2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/lane_parsing/lane_parsing_packing_data_publish_annoset_20230220_113145/pilot_lane_parsing_annoset/20230220-114103",
                "sample_weight": 0.2,
            },
        ]
    },
    "vehicle_wheel_kps": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_kps/vehicle_2_kps/vehicle_2_kps_all_20201021.train",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_kps_detection/release_rec/vehicle_2_kps_add0323cutin_20201125",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/pilot_vehicle_kps2_5v_day_0233/20220119-212808/data",
                "sample_weight": 50,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/pilot_vehicle_kps2_5v_day_X3C/20220119-122143/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/pilot_vehicle_kps2_5v_day_X3C/20220224-131626/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/pilot_vehicle_kps2_5v_CAMERA_X3C_day/20220315-163131/data",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/pilot_vehicle_kps2_5v_CAMERA_X3C_day/20220425-161851/data",
                "sample_weight": 1,
            },
        ]
    },
    "vehicle_ground_line": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/pilot_vehicle_flank_5v_day_0233/20220119-221935/data",
                "sample_weight": 50,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/pilot_vehicle_flank_5v_day_X3C/20220119-125418/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/pilot_vehicle_flank_5v_day_X3C/20220224-133404/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/pilot_vehicle_flank_5v_CAMERA_X3C_day/20220315-163208/data",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/pilot_vehicle_flank_5v_CAMERA_X3C_day/20220425-161825/data",
                "sample_weight": 1,
            },
        ]
    },
    "vehicle_flank": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehFlank/kVehFlank_Cam_5V/20220706-213158/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehFlank/kVehFlank_ar0233-RGGB_train_Cam_5V/20220706-205154/data",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehFlank/kVehFlank_ar0233-WISSEN_train_Cam_5V/20220706-165341/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehFlank/kVehFlank_ar0233-WISSEN_train_day_big_vehicle_Cam_5V/20220706-123556/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehFlank/kVehFlank_x3c_train_day_Cam_5V/20220706-112001/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehFlank/kVehFlank_project_cc02_5v_CO-OX3GB-A100-L_train_day_big_vehicle_Cam_5V/20220706-102652/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehFlank/kVehFlank_x3c_train_night_Cam_5V/20220706-163624/data",
                "sample_weight": 6,
            },
        ]
    },
    "vehicle_wheel_detection": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_wheel_detection/det_subbox_vehicle_wheel_belongto_clean-veh_wheel-20210407_train_clean_323_all_trans_pb_rec/train",
                "sample_weight": 171,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_wheel_detection/MCP_wheel_detection_day_0927/train",
                "sample_weight": 85,
            },
        ]
    },
    "rear": {
        "train_data_paths": [
            # galaxy day rear annoset(25953) 2618
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220622-150345/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # galaxy night rear annoset(26076) 535
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220622-171850/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # galaxy day side annoset(25955) 2993
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Side/20220622-150745/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # galaxy night side annoset(26074) 733
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Side/20220622-172038/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # galaxy night side annoset(39573) 1426
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220719-115310/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # galaxy day rear annoset(39564) 13286
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220719-114719/data",
                "sample_weight": 2,
                "partition": "base",
            },
            # galaxy day side annoset(39539) 4225
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Side/20220719-111918/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # galaxy night side annoset(47638) 1188
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Side/20220805-224901/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # galaxy night rear annoset(47641) 2115
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220805-230106/data",
                "sample_weight": 2,
                "partition": "base",
            },
            # galaxy day side annoset(47648) 8140
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Side/20220805-233211/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # galaxy day rear annoset(47651) 12466
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220806-000857/data",
                "sample_weight": 2,
                "partition": "base",
            },
            # PROJECT_C385_5V 泊车 6319 annoset(19300)
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_PROJECT_C385_5V_PARKING_all/20220524-122038/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # PROJECT_C385_5V 泊车 11926
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_day/20220621-011046/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # PROJECT_C385_5V 泊车
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_rear/pilot_rear_5v_CO-OX3GB-D100-L_day/20220812-004728/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # galaxy day side annoset(61058) 23870
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Side/20221001-094341/data",
                "sample_weight": 3,
                "partition": "base",
            },
            # galaxy day rear annoset(61064) 17728
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Rear/20221001-093816/data",
                "sample_weight": 2,
                "partition": "base",
            },
            # galaxy night rear annoset(61066) 4031
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Rear/20221001-093022/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # galaxy night side annoset(61062) 5948
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Side/20221001-092532/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # galaxy day side contain parking annoset(66501) 5055
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Side/20221027-205819/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # galaxy night side contain parking annoset(66516) 1854
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Side/20221027-205945/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # galaxy day rear contain parking annoset(66517) 3499
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Rear/20221027-210957/data",
                "sample_weight": 1,
                "partition": "base",
            },
            # galaxy night rear contain parking annoset(66520) 1271
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Rear/20221027-211254/data",
                "sample_weight": 1,
                "partition": "base",
            },
        ]
    },
    "rear_occlusion_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_data/vehicle_rear_occlusion_cls_data/cls_subbox_rear_occlusion_v8-20190719-train_base_night_gray-cn93-cn750-model-score0.18/data",
                "sample_weight": 12,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_data/vehicle_rear_occlusion_cls_data/cls_subbox_rear_occlusion_v8-20200903-train_part_4_0323_ispDefault_2280x1080_exclude_part33-cn750-cn759/data",
                "sample_weight": 18,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_data/vehicle_rear_occlusion_cls_data/cls_rear_neg_v4-20200901-fp_train/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/pilot_rear_occ_5v_day_0233/20220119-041133/data",
                "sample_weight": 50,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/pilot_rear_occ_5v_day_X3C/20220119-120507/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/pilot_rear_occ_5v_day_X3C/20220224-123226/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/pilot_rear_occ_5v_CAMERA_X3C_day/20220316-102102/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/pilot_rear_occ_5v_CAMERA_X3C_day/20220401-160904/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearOcclusion/kVehRearOcclusion_project_galaxy_5v_train_day_Cam_Rear/20220622-171123/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearOcclusion/kVehRearOcclusion_project_galaxy_5v_train_day_Cam_Side/20220622-171126/data",
                "sample_weight": 1,
            },
        ]
    },
    "rear_part_classification": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_data/vehicle_rear_cls_data/cls_vehicle_part_3cls_v1-old/data",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_data/vehicle_rear_cls_data/cls_vehicle_part_3cls_v1-hard/data",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_data/vehicle_rear_cls_data/cls_vehicle_part_3cls_v1-res/data",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_data/vehicle_rear_cls_data/cls_vehicle_part_3cls_v1-0323/data",
                "sample_weight": 9,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/pilot_rear_part_5v_day_0233/20220119-210054/data",
                "sample_weight": 50,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/pilot_rear_part_5v_day_X3C/20220119-120509/data",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/pilot_rear_part_5v_day_X3C/20220224-130525/data",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/pilot_rear_part_5v_CAMERA_X3C_day/20220315-163229/data",
                "sample_weight": 2,
            },
        ]
    },
}

pilot3_day_weight_ratio = dict(
    rear_plate_detection=-1,
    person_face_detection=-1,
    vehicle=1,
    vehicle_occlusion_classification=-1,
    vehicle_category=-1,
    vehicle_truncation_classification=-1,
    vehicle_wheel_kps=-1,
    vehicle_ground_line=-1,
    vehicle_flank=1,
    vehicle_wheel_detection=-1,
    rear=1,
    rear_occlusion_classification=-1,
    rear_part_classification=-1,
    person=1,
    person_occlusion_classification=-1,
    person_pose_classification=-1,
    person_orientation_classification=-1,
    cyclist=1,
    cyclist_classification=-1,
    cyclist_kps=-1,
    semantic_parsing=-1,
    lane_parsing=-1,
)
pilot3_night_weight_ratio = dict(
    rear_plate_detection=-1,
    person_face_detection=-1,
    vehicle=1,
    vehicle_occlusion_classification=-1,
    vehicle_category=-1,
    vehicle_truncation_classification=-1,
    vehicle_wheel_kps=-1,
    vehicle_ground_line=-1,
    vehicle_flank=1,
    vehicle_wheel_detection=-1,
    rear=1,
    rear_occlusion_classification=-1,
    rear_part_classification=-1,
    person=1,
    person_occlusion_classification=-1,
    person_pose_classification=-1,
    person_orientation_classification=-1,
    cyclist=1,
    cyclist_classification=-1,
    cyclist_kps=-1,
    semantic_parsing=-1,
    lane_parsing=-1,
)

pilot3_day_datapaths = EasyDict(pilot3_day_datapaths)
pilot3_night_datapaths = EasyDict(pilot3_night_datapaths)
galaxy_datapaths = EasyDict(galaxy_datapaths)
append_data_not_exists_in_galaxy(
    input_datapaths=pilot3_day_datapaths,
    output_datapaths=galaxy_datapaths,
    weight_append_ratio=pilot3_day_weight_ratio,
)
append_data_not_exists_in_galaxy(
    input_datapaths=pilot3_night_datapaths,
    output_datapaths=galaxy_datapaths,
    weight_append_ratio=pilot3_night_weight_ratio,
)

datapaths = galaxy_datapaths
